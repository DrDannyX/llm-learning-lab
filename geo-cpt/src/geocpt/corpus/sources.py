"""Where the raw text comes from.

CPT needs *volume* of raw text, not labelled pairs. That changes the sourcing
problem completely from the SFT project next door:

  SFT  ~18k passages, each needing a high-quality label   -> label quality is the work
  CPT  ~10^5 documents, no labels at all                  -> corpus hygiene is the work

Three open sources, all from Geoscience Australia (CC BY 4.0, attribution
required -- "(c) Commonwealth of Australia (Geoscience Australia)"):

  ecat   abstracts of GA's publications, from its eCat catalogue (~21k records)
  asud   the WHOLE Australian stratigraphic lexicon: every definition card and
         reference note, including superseded units that geo-sft leaves out
  task   geo-sft's own passages with the JSON labels thrown away (this is TAPT)

The third is the important trick: TAPT is not a different technique from DAPT,
it is the same technique pointed at the task's own unlabelled text.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import httpx
from rich.console import Console
from rich.progress import Progress, TimeElapsedColumn
from selectolax.parser import HTMLParser
from tenacity import retry, stop_after_attempt, wait_exponential

from geosft.data.fetch import ADMIN_SECTIONS as _ADMIN_SECTIONS

from .. import paths

console = Console()
ECAT_API = "https://ecat.ga.gov.au/geonetwork/srv/api/search/records/_search"
UA = "geo-cpt/0.1 (educational CPT project)"
ECAT_PAGE = 500

_WS = re.compile(r"\s+")


def strip_html(text: str) -> str:
    """Catalogue abstracts sometimes arrive as HTML fragments (<p>, &nbsp;)."""
    if not text:
        return ""
    if "<" in text:
        text = HTMLParser(text).text(separator=" ")
    return _WS.sub(" ", text.replace("\xa0", " ")).strip()


@retry(wait=wait_exponential(min=2, max=30), stop=stop_after_attempt(4), reraise=True)
def _search(client: httpx.Client, body: dict) -> dict:
    r = client.post(ECAT_API, json=body, headers={"Accept": "application/json"})
    r.raise_for_status()
    return r.json()


def fetch_ecat(resource_types: list[str], max_docs: int | None,
               cache: Path | None = None) -> list[dict]:
    """Abstracts from Geoscience Australia's eCat catalogue (GeoNetwork).

    Real technical prose written by GA scientists about Australian geology,
    ~600 characters each. Paged with `search_after` on the record uuid:
    Elasticsearch refuses plain from/size paging past 10,000 hits, and eCat
    holds ~21,000 documents.
    """
    cache = cache or (paths.RAW / "ecat.jsonl")
    if cache.exists():
        docs = [json.loads(l) for l in cache.open()]
        console.print(f"[dim]ecat cached: {len(docs)} docs[/dim]")
        return docs

    query = {"bool": {"must": [{"terms": {"resourceType": resource_types}},
                               {"exists": {"field": "resourceAbstractObject.default"}}]}}
    docs: list[dict] = []
    after = None
    with httpx.Client(timeout=90.0, headers={"User-Agent": UA}, follow_redirects=True) as client:
        with Progress(*Progress.get_default_columns(), TimeElapsedColumn(),
                      console=console) as prog:
            task = prog.add_task("ecat abstracts", total=max_docs)
            while max_docs is None or len(docs) < max_docs:
                body = {"size": ECAT_PAGE, "query": query, "sort": [{"uuid": "asc"}],
                        "_source": ["uuid", "resourceTitleObject.default",
                                    "resourceAbstractObject.default",
                                    "publicationYearForResource"]}
                if after:
                    body["search_after"] = after
                hits = _search(client, body)["hits"]["hits"]
                if not hits:
                    break
                for h in hits:
                    src = h["_source"]
                    text = strip_html((src.get("resourceAbstractObject") or {}).get("default", ""))
                    if not text:
                        continue
                    year = src.get("publicationYearForResource")
                    docs.append({
                        "id": f"ecat:{src.get('uuid')}",
                        "source": "ecat",
                        "title": (src.get("resourceTitleObject") or {}).get("default", ""),
                        "year": int(year[0]) if isinstance(year, list) and year else None,
                        "text": text,
                    })
                after = hits[-1]["sort"]
                prog.update(task, completed=len(docs))
    docs = docs[:max_docs] if max_docs else docs

    cache.parent.mkdir(parents=True, exist_ok=True)
    with cache.open("w") as fh:
        for d in docs:
            fh.write(json.dumps(d, ensure_ascii=False) + "\n")
    console.print(f"[green]ecat[/green] {len(docs)} abstracts -> {cache}")
    return docs


def fetch_asud_full(not_current: bool = True) -> list[dict]:
    """The whole ASUD lexicon as raw text: one document per unit.

    geo-sft already snapshotted ASUD (data/raw/asud), so nothing is
    re-downloaded. Unlike geo-sft this keeps EVERY unit's text -- no
    per-unit cap, no label filter, superseded names too -- because CPT
    wants volume and needs no labels.

    A unit's definition card and all its reference notes become ONE document,
    the way the lexicon itself presents a unit. Most notes are one or two
    sentences; as separate documents 94% of them fall below any sensible
    minimum length and the quality filter discards them. Repeated notes
    within a unit ("Geological Province: Sydney Basin.") are kept once.
    Leakage into geo-sft's test split is handled by `heldout_units`.
    """
    from geosft.data import asud
    from geosft.data.fetch import clean_passage

    asud.fetch(not_current=not_current)
    docs: list[dict] = []
    for status in (["Current", "Notcurrent"] if not_current else ["Current"]):
        parts: dict[str, list[str]] = {}
        names: dict[str, str] = {}
        for r in asud.read_table("definition", status):
            body = clean_passage(r.get("Contents", ""))
            if body and r.get("Category") not in _ADMIN_SECTIONS:
                parts.setdefault(r["Stratno"], []).append(f"{r['Category']}: {body}")
                names.setdefault(r["Stratno"], r.get("Stratigraphic Name", ""))
        for r in asud.read_table("articles", status):
            body = clean_passage(r.get("Reference Comments", ""))
            if body:
                parts.setdefault(r["Stratno"], []).append(body)
                names.setdefault(r["Stratno"], r.get("Stratigraphic Name", ""))
        for sn, texts in parts.items():
            uniq = list(dict.fromkeys(texts))
            docs.append({"id": f"asud:{status}:{sn}", "source": "asud", "unit_id": asud._int(sn),
                         "title": names[sn], "year": None,
                         "text": f"{names[sn]}. " + " ".join(uniq)})
    console.print(f"[green]asud[/green] {len(docs)} unit entries "
                  f"({sum(len(d['text']) for d in docs):,} chars)")
    return docs


def heldout_units() -> set[int]:
    """ASUD units in geo-sft's valid, test or gold sets.

    The DAPT corpus is the whole lexicon, which CONTAINS geo-sft's test
    passages. Pretraining on them and then scoring SFT on them is leakage
    that no loss curve will show you, so every held-out unit's text is
    dropped from DAPT. TAPT does not need this: it reads the train split only.
    """
    from geosft import paths as sft_paths

    out: set[int] = set()
    for p in (sft_paths.PROCESSED / "valid.meta.jsonl", sft_paths.PROCESSED / "test.meta.jsonl",
              sft_paths.GOLD / "gold.jsonl"):
        if not p.exists():
            raise FileNotFoundError(f"{p} missing. Run `geosft build` in ../geo-sft first -- "
                                    "DAPT must know which units to hold out.")
        out |= {json.loads(line)["unit_id"] for line in p.open()}
    return out


def load_task_text() -> list[dict]:
    """TAPT corpus: geo-sft's TRAIN passages, labels discarded.

    Two rules, both easy to get wrong and both fatal to your evaluation:

    1. **Train split only.** Pretraining on the text of your test set is
       leakage, and it is invisible -- your downstream numbers just quietly
       improve for no honest reason.
    2. **Text only.** Drop the JSON targets. If you pretrain on the answers
       you are doing a sloppy, unmasked form of SFT and calling it CPT.
    """
    from geosft import paths as sft_paths

    meta = sft_paths.PROCESSED / "train.meta.jsonl"
    if not meta.exists():
        raise FileNotFoundError(
            f"{meta} missing. Run `geosft build` in ../geo-sft first -- TAPT "
            f"trains on that project's task text."
        )
    docs = []
    for i, line in enumerate(meta.open()):
        row = json.loads(line)
        docs.append({
            "id": f"task:{row.get('unit_id')}:{i}",
            "source": "task",
            "title": row.get("unit_name") or "",
            "year": row.get("ref_year"),
            "text": row["passage"],
        })
    console.print(f"[green]task text[/green] {len(docs)} passages (train split, labels dropped)")
    return docs


def load_general(dataset: str, config: str | None, split: str, n_docs: int,
                 tag: str = "general") -> list[dict]:
    """Load an arbitrary general-text corpus, for the forgetting probe.

    Kept separate from `load_replay` on purpose: the corpus you REPLAY and the
    corpus you MEASURE FORGETTING ON must be different, or you are measuring
    whether you learned the replay distribution. See CorpusCfg.forgetting_dataset.
    """
    try:
        from datasets import load_dataset
    except ImportError:
        console.print("[yellow]datasets not installed — forgetting probe disabled[/yellow]")
        return []
    try:
        ds = load_dataset(dataset, config, split=split) if config else \
            load_dataset(dataset, split=split)
    except Exception as exc:
        console.print(f"[yellow]forgetting corpus unavailable ({exc})[/yellow]")
        return []

    docs, buf = [], []
    for row in ds:
        t = (row.get("text") or "").strip()
        if not t:
            continue
        buf.append(t)
        if sum(len(x) for x in buf) > 1500:
            docs.append({"id": f"{tag}:{len(docs)}", "source": tag,
                         "title": "", "year": None, "text": " ".join(buf)})
            buf = []
            if len(docs) >= n_docs:
                break
    console.print(f"[green]{tag}[/green] {len(docs)} held-out documents from {dataset}")
    return docs


def load_replay(dataset: str, config: str, n_docs: int) -> list[dict]:
    """General English, mixed back in to fight catastrophic forgetting.

    Any sufficiently general corpus works; wikitext is small and quick.
    """
    try:
        from datasets import load_dataset
    except ImportError:
        console.print("[yellow]datasets not installed -- replay disabled[/yellow]")
        return []
    try:
        ds = load_dataset(dataset, config, split="train")
    except Exception as exc:
        console.print(f"[yellow]replay corpus unavailable ({exc}) -- continuing without[/yellow]")
        return []

    docs, buf = [], []
    for row in ds:
        t = (row.get("text") or "").strip()
        if not t:
            continue
        buf.append(t)
        if sum(len(x) for x in buf) > 1500:
            docs.append({"id": f"replay:{len(docs)}", "source": "replay",
                         "title": "", "year": None, "text": " ".join(buf)})
            buf = []
            if len(docs) >= n_docs:
                break
    console.print(f"[green]replay[/green] {len(docs)} general-English documents")
    return docs
