"""Document packing: turning variable-length documents into fixed token blocks.

THE DIFFERENCE FROM SFT, AND WHY IT IS NOT COSMETIC
---------------------------------------------------
In the SFT project next door, each training example is one prompt+answer pair,
padded to the longest item in its batch. That is fine there: examples are
similar lengths and every one carries a label.

CPT documents vary wildly -- a 40-token Geolex fragment next to a 3,000-token
USGS abstract. Padding each to `block_size` would mean a 40-token document
occupies a 1,024-token slot and **wastes 96% of that compute on padding**.

So CPT *packs*: concatenate every document into one long token stream, then
slice it into equal blocks.

    doc1 <eos> doc2 <eos> doc3 <eos> doc4 ...
    |<-- block 0 -->|<-- block 1 -->|<-- block 2 -->|

Near-zero waste, every block full, constant memory per step.

THE COST: CROSS-DOCUMENT CONTAMINATION
--------------------------------------
A block can straddle a document boundary, so tokens at the start of doc3 can
attend to the tail of doc2 -- unrelated text. In practice this is tolerated
almost universally in pretraining, because the waste from avoiding it is worse
and the model learns that <eos> signals "what came before is irrelevant".

Serious implementations avoid it with block-diagonal attention masks; MLX's
trainer does not expose that, so this module offers the blunt alternative:
`no_cross_document=True` starts every document in a fresh block, trading
tokens for purity. Measure the waste before choosing -- `pack_stats` reports
it, and on this corpus it is large.

WHAT THE MODEL IS TRAINED ON
----------------------------
Every token. There is no prompt mask here: for block `x`, inputs are
`x[:-1]` and targets are `x[1:]`. That is the core difference from SFT, where
the entire prompt is excluded from the loss.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from rich.console import Console
from rich.progress import Progress
from rich.table import Table

console = Console()


def pack_documents(texts: list[str], tokenizer, block_size: int,
                   add_eos_between_docs: bool = True,
                   no_cross_document: bool = False) -> tuple[np.ndarray, dict]:
    """Tokenise and pack into an (n_blocks, block_size) int32 array."""
    eos = getattr(tokenizer, "eos_token_id", None)
    stream: list[int] = []
    blocks: list[list[int]] = []
    n_tokens = 0
    n_padding = 0

    with Progress(*Progress.get_default_columns(), console=console) as prog:
        task = prog.add_task("packing", total=len(texts))
        for text in texts:
            ids = tokenizer.encode(text, add_special_tokens=False)
            if add_eos_between_docs and eos is not None:
                ids = ids + [eos]
            n_tokens += len(ids)

            if no_cross_document:
                # every document starts a fresh block; tail is padded
                for i in range(0, len(ids), block_size):
                    chunk = ids[i:i + block_size]
                    if len(chunk) < block_size:
                        if eos is None:
                            continue
                        n_padding += block_size - len(chunk)
                        chunk = chunk + [eos] * (block_size - len(chunk))
                    blocks.append(chunk)
            else:
                stream.extend(ids)
                while len(stream) >= block_size:
                    blocks.append(stream[:block_size])
                    stream = stream[block_size:]
            prog.advance(task)

    # the trailing partial block is discarded rather than padded: it is a
    # fraction of one block and padding it would teach the model nothing
    dropped = len(stream)
    arr = np.asarray(blocks, dtype=np.int32) if blocks else np.zeros((0, block_size), np.int32)

    stats = {
        "documents": len(texts),
        "tokens": n_tokens,
        "blocks": int(arr.shape[0]),
        "block_size": block_size,
        "tokens_in_blocks": int(arr.size),
        "padding_tokens": n_padding,
        "dropped_tail_tokens": dropped,
        # TWO different questions, and conflating them hides the whole tradeoff:
        #   retention  = of the real tokens we had, how many survived into blocks?
        #                (packing drops a partial tail; no_cross_document drops none)
        #   efficiency = of the tokens we will actually COMPUTE ON, how many are
        #                real rather than padding? This is the compute question.
        # Packing:            ~97% retention, 100% efficiency.
        # no_cross_document: 100% retention,  ~62% efficiency on this corpus.
        "retention_pct": round(100 * (arr.size - n_padding) / max(n_tokens, 1), 2),
        "efficiency_pct": round(100 * (arr.size - n_padding) / max(arr.size, 1), 2),
        "no_cross_document": no_cross_document,
    }
    return arr, stats


def pack_stats_table(stats: dict) -> Table:
    t = Table(title="packing")
    t.add_column("metric"); t.add_column("value", justify="right")
    for k, v in stats.items():
        t.add_row(k, f"{v:,}" if isinstance(v, int) else str(v))
    return t


def save(arr: np.ndarray, stats: dict, out_dir: Path, name: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.npy"
    np.save(path, arr)
    (out_dir / f"{name}.stats.json").write_text(json.dumps(stats, indent=2))
    console.print(f"[green]packed[/green] {name}: {arr.shape} -> {path}")
    return path


def load(path: Path) -> np.ndarray:
    return np.load(path)
