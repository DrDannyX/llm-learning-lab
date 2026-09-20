"""Re-measure forgetting against an INDEPENDENT corpus.

The first TAPT run drew its "general" validation set from the replay corpus
(wikitext), so it measured whether the model learned wikitext rather than
whether it kept general ability. This rebuilds ONLY the forgetting probe --
from a corpus never used for replay and never trained on -- and re-scores the
existing checkpoint against it.

No retraining. Training data is left untouched, so the checkpoint remains
exactly the artifact it was.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rich.console import Console

from geocpt import paths
from geocpt.config import Config
from geocpt.corpus import sources
from geocpt.pack import pack_documents, save

console = Console()


def main(cfg_path: str, checkpoint: str) -> None:
    paths.ensure()
    cfg = Config.load(cfg_path)
    stage = cfg.corpus.stage

    console.rule("[bold]1/3 build an independent forgetting probe")
    docs = sources.load_general(cfg.corpus.forgetting_dataset,
                                cfg.corpus.forgetting_config,
                                cfg.corpus.forgetting_split,
                                cfg.corpus.val_docs, tag="forgetting")
    if not docs:
        console.print("[red]no forgetting corpus available[/red]")
        return
    out = paths.CORPUS / stage / "val_forgetting.jsonl"
    with out.open("w") as fh:
        for d in docs:
            fh.write(json.dumps(d, ensure_ascii=False) + "\n")
    console.print(f"-> {out}")

    console.rule("[bold]2/3 pack")
    from mlx_lm.utils import load as load_model
    _, tok = load_model(cfg.train.base_model)
    arr, stats = pack_documents([d["text"] for d in docs], tok,
                                cfg.pack.block_size, cfg.pack.add_eos_between_docs)
    # overwrite val_general so eval/perplexity picks it up unchanged
    save(arr, stats, paths.PACKED / stage, "val_general")

    console.rule("[bold]3/3 re-score base vs adapted")
    from geocpt.eval.perplexity import compare, save as save_report
    rep = compare(cfg.train.base_model, checkpoint, stage,
                  cfg.train.batch_size, cfg.eval.max_blocks)
    rep["forgetting_corpus"] = cfg.corpus.forgetting_dataset
    rep["replay_corpus"] = cfg.corpus.replay_dataset
    rep["note"] = ("forgetting measured on a corpus never used for replay and "
                   "never trained on; supersedes the first run's figure")
    save_report(rep, Path(checkpoint).parent / "ppl_report_independent.json")

    console.print(f"\n[bold]corrected forgetting[/bold]: "
                  f"{rep['forgetting_pct']:+.1f}% on {cfg.corpus.forgetting_dataset} "
                  f"(replay was {cfg.corpus.replay_dataset})")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
