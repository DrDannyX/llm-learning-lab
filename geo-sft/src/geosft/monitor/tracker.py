"""Training monitoring: structured metrics, live console, divergence alarms.

A loss number scrolling past at 10 Hz teaches you nothing. What you actually
need to see while a fine-tune runs is:

* is validation loss still falling, or has it turned up? (overfitting)
* is the gap between train and val loss widening? (memorisation)
* is the loss flat from step 1? (LR too low, or nothing is trainable)
* is it NaN? (LR too high)
* how much memory and how much time is left?

This tracker answers all five, writes every metric to JSONL so runs are
comparable after the fact, and plots the curves at the end.
"""
from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path

from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.table import Table

console = Console()

SPARK = "▁▂▃▄▅▆▇█"


def sparkline(values: list[float], width: int = 40) -> str:
    if not values:
        return ""
    vals = values[-width:]
    lo, hi = min(vals), max(vals)
    if not math.isfinite(lo) or not math.isfinite(hi) or hi - lo < 1e-9:
        return SPARK[0] * len(vals)
    return "".join(SPARK[int((v - lo) / (hi - lo) * (len(SPARK) - 1))] for v in vals)


@dataclass
class RunTracker:
    """Backend-agnostic: the MLX callback and the HF callback both feed this."""

    run_dir: Path
    total_iters: int
    #: consecutive validations with no improvement before we shout
    patience: int = 3
    quiet: bool = False

    train_hist: list[tuple[int, float]] = field(default_factory=list)
    val_hist: list[tuple[int, float]] = field(default_factory=list)
    best_val: float = math.inf
    best_iter: int = -1
    since_improve: int = 0
    alerts: list[str] = field(default_factory=list)
    _start: float = field(default_factory=time.perf_counter)
    _live: Live | None = None
    _last: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.run_dir = Path(self.run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self._fh = (self.run_dir / "metrics.jsonl").open("a")
        if not self.quiet:
            self._live = Live(self._render(), console=console, refresh_per_second=4)
            self._live.start()

    # ---------------- ingestion ----------------
    def log_train(self, info: dict) -> None:
        it = int(info.get("iteration", 0))
        loss = float(info.get("train_loss", float("nan")))
        self.train_hist.append((it, loss))
        self._last = {**info, "kind": "train"}
        self._emit({"kind": "train", "wall_s": round(time.perf_counter() - self._start, 2), **info})

        if not math.isfinite(loss):
            self._alert(f"iter {it}: loss is {loss} -- diverged. Lower the learning rate.")
        elif len(self.train_hist) >= 12:
            first = [l for _, l in self.train_hist[:5]]
            recent = [l for _, l in self.train_hist[-5:]]
            if abs(sum(recent) / 5 - sum(first) / 5) < 1e-3:
                self._alert(
                    f"iter {it}: train loss flat since the start. Check that adapters "
                    f"are attached and the LR is not ~0."
                )
        self._refresh()

    def log_val(self, info: dict) -> None:
        it = int(info.get("iteration", 0))
        loss = float(info.get("val_loss", float("nan")))
        self.val_hist.append((it, loss))
        self._last = {**self._last, "val_loss": loss}
        self._emit({"kind": "val", "wall_s": round(time.perf_counter() - self._start, 2), **info})

        if loss < self.best_val - 1e-4:
            self.best_val, self.best_iter, self.since_improve = loss, it, 0
        else:
            self.since_improve += 1
            if self.since_improve >= self.patience:
                self._alert(
                    f"iter {it}: val loss has not improved for {self.since_improve} "
                    f"evals (best {self.best_val:.3f} @ {self.best_iter}). "
                    f"Overfitting -- stop here and use the best checkpoint."
                )
        # train/val divergence is the clearer memorisation signal
        if self.train_hist:
            gap = loss - self.train_hist[-1][1]
            if gap > 0.5:
                self._alert(f"iter {it}: val exceeds train by {gap:.2f} -- memorising.")
        self._refresh()

    # ---------------- presentation ----------------
    def _alert(self, msg: str) -> None:
        if msg not in self.alerts:
            self.alerts.append(msg)

    def _emit(self, row: dict) -> None:
        self._fh.write(json.dumps(row) + "\n")
        self._fh.flush()

    def _render(self) -> Panel:
        t = Table.grid(padding=(0, 2))
        t.add_column(style="dim"); t.add_column()
        it = self.train_hist[-1][0] if self.train_hist else 0
        elapsed = time.perf_counter() - self._start
        eta = (elapsed / it * (self.total_iters - it)) if it else 0.0

        t.add_row("iteration", f"{it} / {self.total_iters}")
        if self.train_hist:
            t.add_row("train loss", f"{self.train_hist[-1][1]:.4f}")
            t.add_row("", sparkline([l for _, l in self.train_hist]))
        if self.val_hist:
            t.add_row("val loss", f"{self.val_hist[-1][1]:.4f}")
            t.add_row("", sparkline([l for _, l in self.val_hist]))
            t.add_row("best val", f"{self.best_val:.4f} @ iter {self.best_iter}")
            if self.train_hist:
                t.add_row("val - train", f"{self.val_hist[-1][1] - self.train_hist[-1][1]:+.3f}")
        for key, lbl, fmt in (
            ("learning_rate", "lr", "{:.2e}"),
            ("tokens_per_second", "tokens/s", "{:.0f}"),
            ("peak_memory", "peak mem GB", "{:.2f}"),
            ("trained_tokens", "tokens seen", "{:,.0f}"),
        ):
            if key in self._last:
                t.add_row(lbl, fmt.format(self._last[key]))
        t.add_row("elapsed", f"{elapsed/60:.1f} min")
        t.add_row("eta", f"{eta/60:.1f} min")

        body = [t]
        if self.alerts:
            at = Table.grid(padding=(0, 1))
            at.add_column(style="yellow")
            for a in self.alerts[-4:]:
                at.add_row(f"! {a}")
            body.append(at)
        return Panel(Group(*body), title=f"[bold]{self.run_dir.name}[/bold]",
                     border_style="cyan")

    def _refresh(self) -> None:
        if self._live is not None:
            self._live.update(self._render())

    # ---------------- teardown ----------------
    def close(self) -> dict:
        if self._live is not None:
            self._live.stop()
        self._fh.close()
        summary = {
            "run": self.run_dir.name,
            "iters_done": self.train_hist[-1][0] if self.train_hist else 0,
            "final_train_loss": self.train_hist[-1][1] if self.train_hist else None,
            "final_val_loss": self.val_hist[-1][1] if self.val_hist else None,
            "best_val_loss": None if self.best_val == math.inf else self.best_val,
            "best_val_iter": self.best_iter,
            "wall_minutes": round((time.perf_counter() - self._start) / 60, 2),
            "alerts": self.alerts,
        }
        (self.run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
        self.plot()
        console.print(Panel(json.dumps(summary, indent=2), title="run summary",
                            border_style="green"))
        return summary

    def plot(self) -> Path | None:
        if not self.train_hist:
            return None
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.plot(*zip(*self.train_hist), label="train", lw=1.5)
        if self.val_hist:
            ax.plot(*zip(*self.val_hist), label="validation", lw=1.5, marker="o", ms=3)
            if self.best_iter >= 0:
                ax.axvline(self.best_iter, ls="--", c="grey", lw=1)
                ax.annotate(f"best {self.best_val:.3f}", (self.best_iter, self.best_val),
                            textcoords="offset points", xytext=(6, 8), fontsize=8)
        ax.set_xlabel("iteration"); ax.set_ylabel("loss")
        ax.set_title(f"{self.run_dir.name} — SFT loss")
        ax.legend(); ax.grid(alpha=0.3)
        fig.tight_layout()
        out = self.run_dir / "loss_curve.png"
        fig.savefig(out, dpi=140); plt.close(fig)
        return out


class MLXCallback:
    """Adapter onto mlx_lm.tuner.callbacks.TrainingCallback's duck type."""

    def __init__(self, tracker: RunTracker):
        self.tracker = tracker

    def on_train_loss_report(self, train_info: dict) -> None:
        self.tracker.log_train(train_info)

    def on_val_loss_report(self, val_info: dict) -> None:
        self.tracker.log_val(val_info)
