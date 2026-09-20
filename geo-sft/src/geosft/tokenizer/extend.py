"""Vocabulary extension: graft geoscience tokens onto the base model.

This is the only way a domain tokenizer legitimately enters an SFT pipeline.
Instead of replacing the tokenizer, we ADD a few hundred domain tokens and
give each new token an embedding initialised from the pieces it replaces. The
model's existing 151k tokens keep their meaning, so nothing is destroyed.

    "orthoquartzite"  base: ['Ġorth','o','qu','artz','ite']  -> extended: 1 token
    "Pennsylvanian"   base: ['ĠPenn','s','ylv','anian']      -> extended: 1 token

THREE TRAPS THIS FILE HANDLES, ALL OF WHICH FAIL SILENTLY
---------------------------------------------------------
1. **Shrinking the embedding matrix.** Qwen3-4B declares vocab_size=151936 but
   its tokenizer only has 151669 entries -- 267 spare rows. The reflex call
   `model.resize_token_embeddings(len(tok))` would DELETE 267 rows, including
   live special tokens. We only ever grow, never shrink.
2. **Tied embeddings.** Qwen3 ties lm_head to embed_tokens. Writing to one
   writes to the other; treating them as separate matrices silently doubles
   your edits or leaves lm_head stale. We detect tying and act accordingly.
3. **Random init.** New rows default to random values drawn from the init
   distribution, which puts them nowhere near the manifold the model actually
   uses. Mean-of-subwords starts each new token at the centroid of the pieces
   it replaces, so the model begins from something close to correct.

WHETHER THIS HELPS IS AN EMPIRICAL QUESTION -- MEASURE IT
---------------------------------------------------------
Vocabulary extension is a *pretraining-scale* technique. On a few thousand SFT
examples the new embeddings get very little gradient signal, and extension
often makes no difference or slightly hurts. Build it, run the A/B against the
unextended base, and believe the eval. The measurement is the lesson; a null
result is a real result, not a failed experiment.

Note also that on the MLX path the new rows are frozen (LoRA touches attention
and MLP projections, not embeddings), so they keep their mean-init values. The
HF path can actually TRAIN them via PEFT `modules_to_save=["embed_tokens"]`.
That difference is exactly why this project keeps both backends.
"""
from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

import torch
from rich.console import Console
from tokenizers import AddedToken

from .. import paths
from ..config import TokenizerCfg
from ..data import vocab as vocab_mod

console = Console()

WORD_RE = re.compile(r"[A-Za-z][A-Za-z\-]{3,}")
#: Round the embedding matrix up to a multiple of this. Matrix dims that are
#: multiples of 64 keep the Metal/accelerate kernels on their fast paths.
PAD_MULTIPLE = 64


def mine_candidates(base_model: str, cfg: TokenizerCfg) -> list[dict]:
    """Rank domain terms by the tokens they would save across the corpus.

    savings = corpus_frequency * (base_token_count - 1)

    Frequency alone would add common short words that the base already encodes
    in one token; fragmentation alone would add rare 6-token minerals that
    appear twice. The product targets terms that are both common and badly
    handled, which is where the context budget actually goes.
    """
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(base_model)

    meta = paths.PROCESSED / "train.meta.jsonl"
    if not meta.exists():
        raise FileNotFoundError(f"{meta} missing -- run `geosft build` first")
    passages = [json.loads(l)["passage"] for l in meta.open()]

    # CASE MATTERS, AND GETTING IT WRONG SILENTLY WASTES THE WHOLE STEP.
    # Added tokens are matched with normalized=False, i.e. case-sensitively.
    # Grafting a lowercase "pennsylvanian" does nothing at all, because the
    # word is a proper noun that appears as "Pennsylvanian" every single time
    # -- and the proper nouns (Pennsylvanian, Cretaceous, Ordovician,
    # Mississippian) are exactly the high-value terms. So we count frequency
    # case-insensitively (to rank a term by all its occurrences) but graft the
    # DOMINANT SURFACE FORM actually observed in the corpus.
    surface_freq: Counter[str] = Counter()
    for p_ in passages:
        surface_freq.update(WORD_RE.findall(p_))

    freq: Counter[str] = Counter()
    variants: dict[str, Counter] = {}
    for surface, n in surface_freq.items():
        key = surface.lower()
        freq[key] += n
        variants.setdefault(key, Counter())[surface] += n

    def dominant(key: str, fallback: str) -> str:
        """The casing this term actually wears in the corpus."""
        v = variants.get(key)
        return v.most_common(1)[0][0] if v else fallback

    v = vocab_mod.load()
    gazetteer_surface = (
        list(v["lithologies"]) + [c["name"] for c in v["chronostrat"]] + list(v["minerals"])
    )
    domain_terms = {t.lower() for t in gazetteer_surface}
    # single-word only: multi-word added tokens interact badly with the
    # added-token trie and with whitespace handling
    domain_terms = {t for t in domain_terms if " " not in t and "-" not in t}

    # corpus-driven terms the gazetteers miss ("fossiliferous", "argillaceous")
    corpus_terms = {w for w, c in freq.items() if c >= 20}
    pool = domain_terms | corpus_terms

    # gazetteer casing is the fallback when a term never occurs in the corpus
    gazetteer_case = {t.lower(): t for t in gazetteer_surface}

    rows: list[dict] = []
    seen_surface: set[str] = set()
    for key in sorted(pool):
        f = freq.get(key, 0)
        if f < cfg.min_frequency:
            continue
        term = dominant(key, gazetteer_case.get(key, key))
        if term in seen_surface:
            continue
        n_base = len(tok.encode(" " + term, add_special_tokens=False))
        if n_base < cfg.min_base_tokens:
            continue
        seen_surface.add(term)
        rows.append({
            "term": term, "freq": f, "base_tokens": n_base,
            "savings": f * (n_base - 1),
            "in_gazetteer": key in domain_terms,
        })
    rows.sort(key=lambda r: -r["savings"])

    out = paths.ARTIFACTS / "vocab_candidates.json"
    out.write_text(json.dumps(rows[:2000], indent=2))
    console.print(
        f"[green]candidates[/green] {len(rows)} terms cost {cfg.min_base_tokens}+ base tokens; "
        f"top: " + ", ".join(f"{r['term']}({r['base_tokens']}t x{r['freq']})" for r in rows[:5])
    )
    console.print(f"-> {out}")
    return rows


def extend(base_model: str, cfg: TokenizerCfg, out_dir: Path | None = None) -> Path:
    """Write an extended copy of the base model to artifacts/models/."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    out_dir = Path(out_dir or paths.MODELS / f"{base_model.split('/')[-1]}-geo-ext")
    candidates = mine_candidates(base_model, cfg)[: cfg.extend_top_k]
    if not candidates:
        raise RuntimeError("no vocabulary candidates survived filtering")

    tok = AutoTokenizer.from_pretrained(base_model)
    # capture the ORIGINAL segmentation before the new tokens shadow it
    piece_ids = {
        c["term"]: tok.encode(" " + c["term"], add_special_tokens=False) for c in candidates
    }

    n_before = len(tok)
    added = tok.add_tokens([
        AddedToken(c["term"], lstrip=True, rstrip=False, single_word=True, normalized=False)
        for c in candidates
    ])
    console.print(f"tokenizer {n_before} -> {len(tok)} (+{added})")

    console.print(f"loading {base_model} (this pulls ~8GB on first run)")
    model = AutoModelForCausalLM.from_pretrained(base_model, dtype=torch.bfloat16)

    emb = model.get_input_embeddings()
    cur_rows = emb.weight.shape[0]
    needed = len(tok)
    tied = bool(getattr(model.config, "tie_word_embeddings", False))

    # TRAP 1: grow only. Never let a resize call shrink the matrix.
    if needed > cur_rows:
        target = -(-needed // PAD_MULTIPLE) * PAD_MULTIPLE
        console.print(f"resizing embeddings {cur_rows} -> {target}")
        model.resize_token_embeddings(target)
        emb = model.get_input_embeddings()
    else:
        console.print(
            f"[dim]no resize: {cur_rows} rows already cover {needed} tokens "
            f"({cur_rows - needed} spare)[/dim]"
        )

    # TRAP 3: mean-of-subwords init, in fp32 for numerical sanity
    W = emb.weight.data
    with torch.no_grad():
        for c in candidates:
            term = c["term"]
            new_id = tok.convert_tokens_to_ids(term)
            ids = [i for i in piece_ids[term] if i < cur_rows]
            if not ids or new_id is None or new_id < 0:
                continue
            W[new_id] = W[torch.tensor(ids)].to(torch.float32).mean(0).to(W.dtype)

    # TRAP 2: only touch lm_head when it is a genuinely separate matrix
    head = model.get_output_embeddings()
    if head is not None and not tied:
        Wh = head.weight.data
        with torch.no_grad():
            for c in candidates:
                new_id = tok.convert_tokens_to_ids(c["term"])
                ids = [i for i in piece_ids[c["term"]] if i < cur_rows]
                if ids and new_id is not None and new_id >= 0:
                    Wh[new_id] = Wh[torch.tensor(ids)].to(torch.float32).mean(0).to(Wh.dtype)
        console.print("lm_head initialised separately (untied)")
    elif tied:
        console.print("[dim]lm_head tied to embeddings -- already updated[/dim]")

    model.config.vocab_size = emb.weight.shape[0]
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_dir, safe_serialization=True)
    tok.save_pretrained(out_dir)
    patch_config_for_mlx(out_dir)
    (out_dir / "extension_report.json").write_text(json.dumps({
        "base_model": base_model,
        "tokens_added": added,
        "tokenizer_len": len(tok),
        "embedding_rows": int(emb.weight.shape[0]),
        "tied_embeddings": tied,
        "terms": candidates,
    }, indent=2))
    console.print(f"[green]extended model[/green] -> {out_dir}")
    return out_dir


def patch_config_for_mlx(out_dir: Path) -> dict:
    """Restore the legacy config keys mlx-lm still reads.

    TRAP 4 (a moving target, not a design flaw): transformers 5.x rewrites
    config.json on save -- `rope_theta` becomes a nested `rope_parameters`
    dict and `torch_dtype` becomes `dtype`. mlx-lm 0.31 reads the old flat
    keys, so converting a freshly re-saved model dies with

        TypeError: ModelArgs.__init__() missing 1 required positional
        argument: 'rope_theta'

    The weights are perfectly fine; only the metadata moved. We write the old
    keys back alongside the new ones, which keeps both libraries happy.
    """
    cfg_path = Path(out_dir) / "config.json"
    cfg = json.loads(cfg_path.read_text())
    changed = []

    rope = cfg.get("rope_parameters") or {}
    if "rope_theta" not in cfg and isinstance(rope, dict) and "rope_theta" in rope:
        cfg["rope_theta"] = rope["rope_theta"]
        changed.append("rope_theta")
        rope_type = rope.get("rope_type")
        if rope_type and rope_type != "default" and "rope_scaling" not in cfg:
            cfg["rope_scaling"] = {k: v for k, v in rope.items() if k != "rope_theta"}
            changed.append("rope_scaling")

    if "torch_dtype" not in cfg and "dtype" in cfg:
        cfg["torch_dtype"] = cfg["dtype"]
        changed.append("torch_dtype")

    if changed:
        cfg_path.write_text(json.dumps(cfg, indent=2))
        console.print(f"[dim]config: restored legacy keys for mlx-lm: {changed}[/dim]")
    return cfg


def verify(extended_dir: Path, base_model: str, samples: int = 200) -> dict:
    """Confirm the extension actually shortens real passages, and by how much."""
    from transformers import AutoTokenizer

    base = AutoTokenizer.from_pretrained(base_model)
    ext = AutoTokenizer.from_pretrained(extended_dir)
    meta = paths.PROCESSED / "test.meta.jsonl"
    texts = [json.loads(l)["passage"] for l in meta.open()][:samples]

    nb = sum(len(base.encode(t, add_special_tokens=False)) for t in texts)
    ne = sum(len(ext.encode(t, add_special_tokens=False)) for t in texts)
    res = {
        "passages": len(texts),
        "base_tokens": nb,
        "extended_tokens": ne,
        "reduction_pct": round(100 * (nb - ne) / max(nb, 1), 2),
    }
    console.print(
        f"[bold]context saving[/bold] {nb} -> {ne} tokens "
        f"([green]{res['reduction_pct']}%[/green] shorter) over {len(texts)} held-out passages"
    )
    return res
