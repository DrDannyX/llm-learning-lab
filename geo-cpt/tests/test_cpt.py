"""Tests for the parts of CPT that fail silently if they break."""
from __future__ import annotations

import numpy as np
import pytest

from geocpt.corpus.dedup import deduplicate, shingles, signature
from geocpt.corpus.quality import reject_reason
from geocpt.pack import pack_documents
from geocpt.train.cpt import build_schedule


class FakeTok:
    eos_token_id = 0

    def encode(self, text, add_special_tokens=False):  # noqa: ANN001
        return list(range(1, len(text.split()) + 1))


# ---------------- packing ----------------
def test_packing_produces_full_blocks():
    arr, stats = pack_documents(["w " * 100, "w " * 3000], FakeTok(), 256)
    assert arr.shape[1] == 256
    assert arr.shape[0] == stats["blocks"]
    assert stats["padding_tokens"] == 0, "packing must not pad"


def test_retention_and_efficiency_are_different_questions():
    """The bug this guards: reporting one number hides the whole tradeoff.
    Packing drops a partial tail (retention < 100) but wastes no compute
    (efficiency 100). no_cross_document is the exact opposite."""
    docs = ["w " * 40, "w " * 3000, "w " * 120]
    _, packed = pack_documents(docs, FakeTok(), 1024)
    _, nocross = pack_documents(docs, FakeTok(), 1024, no_cross_document=True)

    assert packed["retention_pct"] < 100 and packed["efficiency_pct"] == 100.0
    assert nocross["retention_pct"] == 100.0 and nocross["efficiency_pct"] < 70
    assert nocross["padding_tokens"] > 0


def test_eos_inserted_between_documents():
    arr, _ = pack_documents(["w " * 300, "w " * 300], FakeTok(), 64)
    assert (arr == FakeTok.eos_token_id).any(), "documents must be separated by EOS"


def test_no_masking_targets_are_shifted_inputs():
    """CPT's defining property: every token is a target. Unlike SFT, where
    the prompt was excluded, inputs/targets are just the block shifted by one."""
    arr, _ = pack_documents(["w " * 5000], FakeTok(), 128)
    block = arr[:1]
    inputs, targets = block[:, :-1], block[:, 1:]
    assert inputs.shape == targets.shape
    assert np.array_equal(inputs[0, 1:], targets[0, :-1])


# ---------------- dedup ----------------
def test_shingles_are_order_sensitive():
    assert shingles("a b c d", k=2) != shingles("d c b a", k=2)


def test_near_duplicate_detected_exact_not_required():
    """A realistic near-duplicate: a long document with one localised edit,
    which is what revised report editions actually look like."""
    base = ("The Austin Chalk is a Late Cretaceous carbonate unit exposed across "
            "central Texas. It overlies the Eagle Ford Group and underlies the "
            "Taylor Marl. Thickness ranges from 60 to 120 metres across the "
            "outcrop belt, thinning northward into Oklahoma where it grades into "
            "marl. The unit is an important reservoir target in the region. ") * 3
    near = base.replace("important reservoir", "significant reservoir", 1)
    far = ("Basaltic andesite flows record subduction related magmatism in the "
           "Cascade arc with plagioclase and clinopyroxene phenocrysts. ") * 6

    j_near = signature(base, 128, 5).jaccard(signature(near, 128, 5))
    j_far = signature(base, 128, 5).jaccard(signature(far, 128, 5))
    assert j_near > 0.8, f"one-word edit in a long doc should stay similar, got {j_near}"
    assert j_far < 0.2

    kept, stats = deduplicate(
        [{"id": "a", "text": base}, {"id": "b", "text": near}, {"id": "c", "text": far}],
        threshold=0.8)
    ids = {d["id"] for d in kept}
    assert "a" in ids and "c" in ids and "b" not in ids
    assert stats.near_removed == 1


def test_shingle_similarity_is_sensitive_to_word_order():
    """Worth knowing before you tune the threshold: swapping two adjacent
    words in a SHORT repeated text invalidates many 5-grams and drops Jaccard
    to ~0.49 -- below a 0.8 threshold. Shingle dedup is about shared phrasing,
    not shared meaning."""
    a = ("The Austin Chalk is a Late Cretaceous carbonate unit exposed across "
         "central Texas and studied widely by many workers. ") * 4
    b = a.replace("studied widely", "widely studied")
    j = signature(a, 128, 5).jaccard(signature(b, 128, 5))
    assert 0.3 < j < 0.7, f"expected a large drop from a word swap, got {j}"


def test_exact_duplicates_removed_first():
    d = {"id": "x", "text": "identical content here " * 20}
    kept, stats = deduplicate([d, dict(d, id="y")], threshold=0.9)
    assert len(kept) == 1 and stats.exact_removed == 1


# ---------------- quality ----------------
@pytest.mark.parametrize("text,expected", [
    ("short", "too_short"),
    ("1234567890 " * 200, "low_alpha_ratio"),
    ("word " * 200, "repetitive"),
])
def test_quality_rejections(text, expected):
    assert reject_reason(text, 300, 50_000) == expected


def test_good_prose_survives():
    good = ("The Austin Chalk is a Late Cretaceous carbonate unit. It overlies the "
            "Eagle Ford Group and underlies the Taylor Marl. Thickness ranges from "
            "60 to 120 metres across the outcrop belt. The unit thins northward. ") * 3
    assert reject_reason(good, 300, 50_000) is None


# ---------------- schedule ----------------
def test_schedule_is_in_optimizer_steps_not_iterations():
    """Same trap the SFT lab hit: MLX indexes schedules by optimizer steps."""
    import mlx.core as mx
    sched = build_schedule(1e-5, total_opt_steps=75, warmup_opt_steps=7,
                           kind="cosine", min_fraction=0.1)
    assert float(sched(mx.array(0))) == pytest.approx(0.0, abs=1e-12)
    assert float(sched(mx.array(7))) == pytest.approx(1e-5, rel=1e-3)
    assert float(sched(mx.array(74))) < 2e-6


def test_warmup_cannot_exceed_run_length():
    import mlx.core as mx
    sched = build_schedule(1e-5, total_opt_steps=3, warmup_opt_steps=500,
                           kind="cosine", min_fraction=0.1)
    assert float(sched(mx.array(1))) > 0.0


# ---------------- reuse contract ----------------
def test_task_text_comes_from_sft_train_split_only():
    """TAPT must never pretrain on the SFT test split -- invisible leakage."""
    import inspect

    from geocpt.corpus import sources
    src = inspect.getsource(sources.load_task_text)
    assert "train.meta.jsonl" in src, "TAPT must read the SFT TRAIN split"
    assert "test.meta.jsonl" not in src, "TAPT must never read the SFT test split"
    assert "valid.meta.jsonl" not in src


# ---------------- checkpointing ----------------
@pytest.mark.slow
def test_checkpoint_saves_and_reloads():
    """The test that would have saved a 48-minute run.

    The first TAPT run died at iteration 300 with
    `ImportError: cannot import name 'save_weights'` -- a function name that
    was guessed, never exercised, and only reached after the first
    `save_every` interval. A checkpoint path must be tested BEFORE a long run,
    on the smallest model available, and must prove the result is reloadable.
    """
    import tempfile
    from pathlib import Path

    from mlx_lm.utils import load

    from geocpt.train.cpt import save_checkpoint

    repo = "Qwen/Qwen3-0.6B"
    model, tok, config = load(repo, return_config=True)
    with tempfile.TemporaryDirectory() as td:
        run_dir = Path(td)
        out = save_checkpoint(model, tok, run_dir, 42, keep_last_n=2,
                              src_repo=repo, config=config)
        assert out.exists()
        assert (out / "config.json").exists(), "config must be written"
        assert list(out.glob("*.safetensors")), "weights must be written"

        # donate_model=False must leave the original model usable
        import mlx.core as mx
        _ = model(mx.array([[1, 2, 3]]))

        # and the checkpoint must load back as a real model
        reloaded, _ = load(str(out))
        out2 = reloaded(mx.array([[1, 2, 3]]))
        assert out2.shape[-1] == config["vocab_size"]


def test_prune_keeps_only_last_n():
    import tempfile
    from pathlib import Path

    from geocpt.train.cpt import _prune_checkpoints

    with tempfile.TemporaryDirectory() as td:
        run = Path(td)
        for step in (100, 200, 300, 400):
            (run / f"checkpoint-{step:06d}").mkdir()
        _prune_checkpoints(run, keep=2)
        remaining = sorted(p.name for p in run.glob("checkpoint-*"))
        assert remaining == ["checkpoint-000300", "checkpoint-000400"]


def test_forgetting_probe_is_not_the_replay_corpus():
    """The bug this guards: the first TAPT run replayed wikitext into training
    and measured 'forgetting' on held-out wikitext. Different documents, same
    distribution -- so general perplexity IMPROVED 40% and the run reported
    -40% forgetting. It was measuring 'did we also learn wikitext'."""
    from geocpt.config import CorpusCfg

    cfg = CorpusCfg()
    assert cfg.forgetting_dataset != cfg.replay_dataset, (
        "the forgetting probe must come from a corpus never used for replay")

    import inspect

    from geocpt.corpus import build
    src = inspect.getsource(build.run)
    assert "load_general(cfg.forgetting_dataset" in src
    assert "general_val = pool[" not in src, "general_val must not be sliced from the replay pool"
