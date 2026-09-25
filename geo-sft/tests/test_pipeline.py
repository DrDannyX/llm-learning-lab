"""Unit tests for the parts that fail silently if they break."""
from __future__ import annotations

import json

import pytest

from geosft.data import asud
from geosft.data.fetch import clean_passage
from geosft.data.label import (Labeller, extract_relations, infer_rank,
                               parse_thickness)
from geosft.data.match import Gazetteer
from geosft.eval.metrics import Scorer, extract_json
from geosft.schema import GeoExtraction, Relation, build_messages


# ---------------- schema ----------------
def test_lists_are_canonical():
    a = GeoExtraction(lithologies=["shale", "chalk", "shale"], chronostrat=["Permian"])
    b = GeoExtraction(lithologies=["chalk", "shale"], chronostrat=["Permian"])
    assert a.to_json() == b.to_json(), "identical facts must serialise identically"


def test_relations_dedupe_and_sort():
    g = GeoExtraction(relations=[
        Relation(kind="underlies", unit="Cygnet Coal Measures"),
        Relation(kind="overlies", unit="Minnie Point Formation"),
        Relation(kind="underlies", unit="cygnet coal measures"),
    ])
    assert len(g.relations) == 2


def test_prompt_symmetry():
    """Train and inference prompts must be identical up to the answer turn."""
    t = GeoExtraction(unit_name="Abels Bay Formation")
    train = build_messages("some passage", t)
    infer = build_messages("some passage")
    assert train[:-1] == infer
    assert train[-1]["role"] == "assistant"


# ---------------- matching ----------------
def test_longest_match_wins():
    g = Gazetteer({"cretaceous": "Cretaceous", "late cretaceous": "Late Cretaceous"})
    assert g.values("of Late Cretaceous age") == ["Late Cretaceous"]


def test_plural_handling():
    g = Gazetteer({"shale": "shale"}, plurals=True)
    assert g.values("interbedded shales") == ["shale"]


# ---------------- labeller ----------------
def test_relation_names_are_not_greedy():
    """The bug this guards: re.IGNORECASE makes [A-Z] match anything, so the
    name group swallows 'member of Howard limestone'."""
    rels = extract_relations(
        "Below the Barnetts member of Abels Bay formation.",
        strat_names={"Barnetts"}, self_name=None,
    )
    assert rels and rels[0].unit == "Barnetts Member"
    assert "of" not in rels[0].unit


def test_australian_relation_wording():
    rels = extract_relations(
        "Abels Bay Formation. Conformably overlain by the Cygnet Coal Measures. "
        "Intruded by the Ben Lomond Granite.",
        strat_names={"Cygnet", "Ben"}, self_name="Abels Bay Formation",
    )
    assert {(r.kind, r.unit) for r in rels} == {
        ("underlies", "Cygnet Coal Measures"), ("intruded_by", "Ben Lomond Granite")}


def test_relation_requires_known_strat_name():
    assert extract_relations("Overlies Something Unknown.", strat_names=set(), self_name=None) == []


def test_explicit_rank_beats_lithology_fallback():
    assert infer_rank("Aarde shale member of Howard limestone.", "Aarde") == "Member"
    assert infer_rank("Austin chalk is exposed nearby.", "Austin") == "Formation"
    assert infer_rank("Austin Group comprises several units.", "Austin") == "Group"


@pytest.mark.parametrize("name,rank", [
    ("Tumblagooda Sandstone", "Formation"),   # rank from the name's own last word
    ("Bulgonunna Volcanics", "Formation"),    # not de-pluralised to 'volcanic'
    ("Moolayember Beds", "Formation"),        # ASUD 'Beds' are Formation rank, not Bed
    ("Babel Island Suite", "Suite"),
    ("Warakurna Supersuite", "Supersuite"),
    ("Glyde Hill Volcanic Complex", "Unknown"),  # ASUD ranks Complexes both ways
])
def test_rank_from_australian_names(name, rank):
    assert infer_rank(f"{name}. Conformably overlies basement.", name) == rank


@pytest.mark.parametrize("text,lo,hi", [
    ("Thickness 100 feet.", 30.48, 30.48),
    ("is 2 to 7 feet thick", 0.61, 2.13),
    ("ranges from 10 to 30 m in thickness", 10.0, 30.0),
    ("Thickness range: up to 1.5 km", 1500.0, 1500.0),
])
def test_thickness_parsing(text, lo, hi):
    t = parse_thickness(text)
    assert t and t.min_m == pytest.approx(lo, abs=0.02) and t.max_m == pytest.approx(hi, abs=0.02)


def test_thickness_ignores_unrelated_numbers():
    assert parse_thickness("Located 5 miles north; exposed along 200 feet of road.") is None


def _au_labeller() -> Labeller:
    return Labeller.from_vocab({
        "lithologies": ["sandstone", "shale", "limestone"], "chronostrat": [
            {"name": "Early Devonian"}, {"name": "Paleozoic"}],
        "chronostrat_aliases": {"lower devonian": "Early Devonian", "palaeozoic": "Paleozoic"},
        "minerals": [], "strat_names": ["Tumblagooda Sandstone", "Dirk Hartog Group"]})


def test_unit_names_do_not_leak_into_lithology():
    """The most common rule error in the Geolex gold review: rock words inside
    proper names. Every ASUD passage leads with one, so it must be masked."""
    out = _au_labeller().label({
        "unit_name": "Tumblagooda Sandstone",
        "passage": "Tumblagooda Sandstone. Reservoir sealed by Dirk Hartog Group shales."})
    assert out.lithologies == ["shale"]
    assert out.rank == "Formation"


def test_capitalised_minerals_are_names():
    lab = Labeller.from_vocab({"lithologies": ["granite"], "chronostrat": [],
                               "minerals": ["silver", "biotite"], "strat_names": []})
    out = lab.label({"unit_name": None, "passage":
                     "Mount You You Granite. Silver Spur Subprovince. Biotite granite: I-type."})
    assert out.minerals == ["biotite"] and out.lithologies == ["granite"]


def test_superseded_names_are_masked_too():
    """'Carcoar Granite' is no longer an ASUD name, but it is still a name."""
    out = _au_labeller().label({"unit_name": None, "passage":
                                "Originally included in the Carcoar Granite. Grey shale."})
    assert out.lithologies == ["shale"]


def test_no_self_relations_and_no_age_prefix():
    rels = extract_relations(
        "Breakfast Sandstone. Unconformably overlies Palaeoproterozoic Murphy Metamorphics. "
        "Buddycurrawa Volcanics overlies Breakfast Sandstone. Overlying unit: Bowgan Sandstone.",
        strat_names={"Murphy", "Breakfast", "Bowgan"}, self_name="Breakfast Sandstone")
    assert {(r.kind, r.unit) for r in rels} == {
        ("unconformable_on", "Murphy Metamorphics"), ("underlies", "Bowgan Sandstone")}


@pytest.mark.parametrize("text,expected", [
    # the article decides the direction
    ("Gogo Formation. Shown overlying the Sadler Limestone.", ("overlies", "Sadler Limestone")),
    ("Alsace Quartzite. The overlying Bortala Formation.", ("underlies", "Bortala Formation")),
    ("Gerowie Tuff. Overlies: Koolpin Formation.", ("overlies", "Koolpin Formation")),
    ("Tanwarra Shale. Underlain by Pipers Flat Formation.", ("overlies", "Pipers Flat Formation")),
    ("Goyder Formation. Is overlain unconformably by Pacoota Sandstone.", ("underlies", "Pacoota Sandstone")),
    ("Apex Basalt. Over Marble Bar Chert Member; under Panorama Formation.", ("overlies", "Marble Bar Chert Member")),
    # a shared core is not the same unit
    ("Murchison Volcanics. Intruded by Murchison Granite.", ("intruded_by", "Murchison Granite")),
    ("Symons Granite. Intrudes the Archaean Mulgathing Complex.", ("intrudes", "Mulgathing Complex")),
])
def test_relation_templates(text, expected):
    heads = {"Sadler", "Bortala", "Koolpin", "Pipers", "Pacoota", "Marble", "Murchison", "Mulgathing"}
    subject = text.split(".")[0]
    rels = {(r.kind, r.unit) for r in extract_relations(text, heads, subject)}
    assert expected in rels
    flip = {"overlies": "underlies", "underlies": "overlies"}.get(expected[0])
    assert (flip, expected[1]) not in rels, "a relation must never come out in both directions"


def test_hyphenated_terms_are_split():
    lab = Labeller.from_vocab({"lithologies": ["granite"], "chronostrat": [
        {"name": "Carnian"}, {"name": "Norian"}], "minerals": ["Biotite", "Muscovite"],
        "strat_names": []})
    out = lab.label({"unit_name": None, "passage": "Muscovite-biotite granite of Carnian-Norian age."})
    assert out.minerals == ["Biotite", "Muscovite"] and out.chronostrat == ["Carnian", "Norian"]


def test_state_names_inside_place_names():
    assert _au_labeller().label({"unit_name": None,
                                 "passage": "Used in the second Victoria Bridge."}).states == []


def test_australian_spellings_canonicalise():
    out = _au_labeller().label({"unit_name": None,
                                "passage": "Of Lower Devonian age, within the Palaeozoic."})
    assert out.chronostrat == ["Early Devonian", "Paleozoic"]


@pytest.mark.parametrize("text,states", [
    ("Mapped across New South Wales and Qld.", ["NSW", "QLD"]),
    ("Exposed along the Victoria River.", []),         # a river, not the state
    ("Outcrops in Victoria near Omeo.", ["VIC"]),
    ("An act of deposition in a sa basin.", []),        # lowercase abbreviations are words
])
def test_states(text, states):
    assert _au_labeller().label({"unit_name": None, "passage": text}).states == states


def test_labeller_only_claims_visible_unit_names():
    lab = Labeller.from_vocab({"lithologies": ["shale"], "chronostrat": [],
                               "minerals": [], "strat_names": []})
    out = lab.label({"unit_name": "Nowhere", "passage": "A grey shale is exposed."})
    assert out.unit_name is None, "must not assert a name absent from the passage"


# ---------------- ASUD parsing ----------------
def test_report_table_stitches_split_records_and_folds_pipes():
    text = ("Stratno | Name | Comments\n"
            "1|Abels Bay Formation|Overlain by\ncoal measures.\n"
            "2|Amber Formation|a | b\n")
    rows, bad = asud.parse_table(text)
    assert bad == 0
    assert rows[0]["Comments"] == "Overlain by coal measures."
    assert rows[1]["Comments"] == "a | b"


def test_related_table_direction():
    """The related table reads `<related> <relation> <subject>`; two columns share
    a header name, so the parser must use positional names."""
    text = ("Related Stratno | Related unit name | Relation Type | Related Stratno | "
            "Stratigraphic Name |Contact Type | Comments\n"
            "5105|Cygnet Coal Measures|overlies|23322|Abels Bay Formation||\n")
    (row,), _ = asud.parse_table(text, asud.RELATED_COLS)
    assert (row["related_stratno"], row["relation"], row["stratno"]) == ("5105", "overlies", "23322")


def test_boilerplate_stripped_from_comments():
    got = clean_passage("Sharp boundary with Hutton Formation. Location in text includes "
                        "p325 Fig.3, p328. See also p66 Fig.1. See 100k_geologyp_lut.csv.")
    assert got == "Sharp boundary with Hutton Formation."


# ---------------- metrics ----------------
def test_extract_json_from_fenced_and_chatty_output():
    assert extract_json('```json\n{"a": 1}\n```') == {"a": 1}
    assert extract_json('Sure! {"a": 1} Hope that helps.') == {"a": 1}
    assert extract_json("no json here") is None


def test_unparseable_counts_as_wrong_not_skipped():
    gold = {"unit_name": "X", "rank": "Group", "lithologies": ["shale"],
            "chronostrat": [], "minerals": [], "states": [],
            "thickness": None, "relations": []}
    s = Scorer()
    s.add("I cannot answer.", gold)
    r = s.report()
    assert r["gates"]["parse_rate"] == 0.0
    assert r["set_fields"]["lithologies"]["fn"] == 1, "gold items must become false negatives"


def test_perfect_prediction_scores_one():
    gold = {"unit_name": "Abels Bay Formation", "rank": "Formation", "lithologies": ["siltstone"],
            "chronostrat": ["Late Permian"], "minerals": [], "states": ["TAS"],
            "thickness": {"min_m": 10.0, "max_m": 20.0},
            "relations": [{"kind": "underlies", "unit": "Cygnet Coal Measures"}]}
    s = Scorer()
    s.add(json.dumps(gold), gold)
    r = s.report()
    assert r["macro_f1"] == 1.0
    assert r["scalar_fields"]["rank_accuracy"] == 1.0
    assert r["gates"]["schema_valid_rate"] == 1.0


def test_empty_field_excluded_from_macro():
    """minerals is empty in both gold and prediction: undefined, not zero."""
    gold = {"unit_name": "A", "rank": "Group", "lithologies": ["shale"],
            "chronostrat": [], "minerals": [], "states": [],
            "thickness": None, "relations": []}
    s = Scorer()
    s.add(json.dumps(gold), gold)
    r = s.report()
    assert "minerals" not in r["macro_f1_fields"]
    assert r["macro_f1"] == 1.0


# ---------------- LR schedule ----------------
def test_schedule_accounts_for_gradient_accumulation():
    """The bug this guards: MLX indexes schedules by optimizer steps, so
    passing iteration counts straight through stretches warmup by the
    accumulation factor and the run crawls at ~0 LR."""
    import mlx.core as mx

    from geosft.config import Config
    from geosft.train.mlx_train import _schedule

    c = Config()
    c.train.iters = 900
    c.train.grad_accumulation_steps = 4
    c.train.warmup_steps = 60      # iterations -> 15 optimizer steps
    c.train.learning_rate = 1e-4
    sched = _schedule(c.train, c.train.iters)

    assert float(sched(mx.array(0))) == pytest.approx(0.0, abs=1e-9)
    # peak must be reached at the optimizer step warmup maps to, not 60
    assert float(sched(mx.array(15))) == pytest.approx(1e-4, rel=1e-3)
    # and must decay, not sit at peak
    assert float(sched(mx.array(224))) < 2e-5


def test_warmup_never_exceeds_run_length():
    import mlx.core as mx

    from geosft.config import Config
    from geosft.train.mlx_train import _schedule

    c = Config()
    c.train.iters = 20          # shorter than warmup_steps
    c.train.grad_accumulation_steps = 8
    c.train.warmup_steps = 300
    sched = _schedule(c.train, c.train.iters)
    assert float(sched(mx.array(1))) > 0.0, "a short run must still train"


def test_added_tokens_must_match_real_surface_casing():
    """The bug this guards: added tokens are matched with normalized=False,
    so grafting lowercase 'pennsylvanian' never fires -- the word is a proper
    noun and always appears capitalised. Silently wastes the vocab slot."""
    from tokenizers import AddedToken
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B-Instruct-2507")
    before = len(tok.tokenize(" Pennsylvanian"))
    tok.add_tokens([AddedToken("pennsylvanian", lstrip=True, single_word=True,
                               normalized=False)])
    assert len(tok.tokenize(" Pennsylvanian")) == before, "lowercase graft must NOT fire"
    tok.add_tokens([AddedToken("Pennsylvanian", lstrip=True, single_word=True,
                               normalized=False)])
    assert len(tok.tokenize(" Pennsylvanian")) == 1, "correct casing must collapse to 1 token"


def test_training_sequence_matches_inference_prompt_exactly():
    """The bug this guards: Qwen3's template injects <think></think> before
    assistant content in a FULL conversation but not in the generation
    prompt, so the scaffolding lands in the trained region and the model
    learns to emit it. Observed for real: strict JSON rate 1.000 (base) ->
    0.000 (tuned) while macro F1 rose. Nothing errors."""
    from mlx_lm import load

    from geosft.train.mlx_train import PromptCompletionDataset

    _, tok = load("mlx-community/Qwen3-4B-Instruct-2507-4bit")
    target = GeoExtraction(unit_name="X")
    msgs = build_messages("X chalk.", target)
    ds = PromptCompletionDataset([{"messages": msgs}], tok)
    tokens, offset = ds.process({"messages": msgs})

    # what the model is graded on must be exactly the answer
    trained = tok.decode(tokens[offset:])
    assert trained.startswith("{"), f"loss must start at the JSON, got {trained[:40]!r}"
    assert "<think>" not in trained

    # and the context it is graded in must be what inference supplies
    eval_prompt = tok.apply_chat_template(
        build_messages("X chalk."), tokenize=False, add_generation_prompt=True
    )
    assert tok.decode(tokens[:offset]) == eval_prompt
    assert tokens[-1] == tok.eos_token_id, "must learn to stop"


def test_gold_and_test_evals_write_to_different_files(tmp_path, monkeypatch):
    """Two bugs guarded here, and the second is why this test EXECUTES the
    function instead of grepping its source:

    1. gold and test runs both wrote eval_report.json, so scoring against the
       hand-reviewed gold set silently destroyed the test-set results.
    2. the fix defined `tag` AFTER its first use, raising UnboundLocalError on
       every eval. The original version of this test only checked that the
       source contained the right strings, so it passed while the code was
       completely broken. A test that greps source is not a test.
    """
    from geosft.config import Config
    from geosft.eval import evaluate

    rows = [{"passage": "X chalk.", "unit_name": "X",
             "target": GeoExtraction(unit_name="X").model_dump(mode="json")}]
    monkeypatch.setattr(evaluate, "load_eval_rows", lambda *a, **k: rows)
    monkeypatch.setattr(evaluate, "generate_mlx",
                        lambda *a, **k: ['{"unit_name": "X"}'])

    cfg = Config()
    cfg.eval.compare_base = True
    for use_gold, expected in ((False, "eval_report.json"),
                               (True, "eval_report_gold.json")):
        out = tmp_path / ("gold" if use_gold else "test")
        evaluate.run(cfg, adapter_path=None, model_path="m",
                     use_gold=use_gold, out_dir=out)
        assert (out / expected).exists(), f"{expected} not written"
    assert not (tmp_path / "gold" / "eval_report.json").exists(), \
        "a gold run must never write the test report file"
