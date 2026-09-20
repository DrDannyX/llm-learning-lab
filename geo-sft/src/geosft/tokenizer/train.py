"""Train a geoscience tokenizer from scratch.

READ THIS BEFORE YOU EXPECT A WIN
---------------------------------
A from-scratch tokenizer cannot be dropped into a pretrained LLM. Token ids
are row indices into the embedding matrix; swapping the tokenizer re-points
every index at an unrelated vector and destroys the model. Recovering from
that needs continued *pretraining* on billions of tokens, not SFT on a few
thousand examples.

So this tokenizer has two legitimate jobs, both of which it does well:

  1. **Measurement.** It quantifies how badly the base tokenizer fragments
     geoscience vocabulary (see analyze.py). That number is what justifies --
     or kills -- the vocabulary-extension step.
  2. **Candidate mining.** Its merges are a data-driven shortlist of domain
     terms worth grafting onto the base vocabulary (see extend.py).

Train it on the TRAIN SPLIT ONLY. A tokenizer fitted on test text has seen
test text, and your compression numbers become self-congratulatory.
"""
from __future__ import annotations

import json
from pathlib import Path

from rich.console import Console
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

from .. import paths
from ..config import TokenizerCfg

console = Console()

SPECIALS = ["<|pad|>", "<|endoftext|>", "<|unk|>"]


def corpus_lines(split: str = "train") -> list[str]:
    """Passage text from one split. Targets are excluded on purpose: JSON
    braces and schema keys are not geoscience language and would waste vocab."""
    meta = paths.PROCESSED / f"{split}.meta.jsonl"
    if not meta.exists():
        raise FileNotFoundError(f"{meta} missing -- run `geosft build` first")
    return [json.loads(line)["passage"] for line in meta.open()]


def train(cfg: TokenizerCfg, out_dir: Path | None = None) -> Path:
    out_dir = Path(out_dir or paths.TOKENIZERS / f"geo-{cfg.model_type}-{cfg.vocab_size}")
    out_dir.mkdir(parents=True, exist_ok=True)

    lines = corpus_lines("train")
    console.print(f"training {cfg.model_type} tokenizer on {len(lines)} passages")

    if cfg.model_type == "bpe":
        tok = Tokenizer(models.BPE(unk_token=None))
        trainer = trainers.BpeTrainer(
            vocab_size=cfg.vocab_size,
            min_frequency=cfg.min_frequency,
            special_tokens=SPECIALS,
            # ByteLevel alphabet keeps the tokenizer lossless on any input
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
            show_progress=True,
        )
    else:
        tok = Tokenizer(models.Unigram())
        trainer = trainers.UnigramTrainer(
            vocab_size=cfg.vocab_size,
            special_tokens=SPECIALS,
            unk_token="<|unk|>",
            show_progress=True,
        )

    # add_prefix_space=False matches how modern Llama/Qwen tokenizers behave
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    tok.train_from_iterator(lines, trainer=trainer, length=len(lines))

    path = out_dir / "tokenizer.json"
    tok.save(str(path))
    (out_dir / "config.json").write_text(json.dumps(cfg.model_dump(), indent=2))
    console.print(f"[green]tokenizer[/green] vocab={tok.get_vocab_size()} -> {path}")
    return out_dir


def load(out_dir: Path) -> Tokenizer:
    return Tokenizer.from_file(str(Path(out_dir) / "tokenizer.json"))
