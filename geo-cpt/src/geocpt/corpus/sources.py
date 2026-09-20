"""Where the raw text comes from.

CPT needs *volume* of raw text, not labelled pairs. That changes the sourcing
problem completely from the SFT project next door:

  SFT  8,757 passages, each needing a high-quality label   -> label quality is the work
  CPT  ~10^5 documents, no labels at all                   -> corpus hygiene is the work

Three public-domain / open sources, all reusable without redistribution:

  usgs    USGS Publications Warehouse abstracts (public domain, US Gov work)
  geolex  the FULL USGS lexicon -- 16,684 units, vs the 3,302 geo-sft sampled
  task    geo-sft's own passages with the JSON labels thrown away (this is TAPT)

The third is the important trick: TAPT is not a different technique from DAPT,
it is the same technique pointed at the task's own unlabelled text.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Iterator

import httpx
from rich.console import Console
from rich.progress import Progress, TimeElapsedColumn
from selectolax.parser import HTMLParser
from tenacity import retry, stop_after_attempt, wait_exponential

from .. import paths

console = Console()
USGS_API = "https://pubs.usgs.gov/pubs-services/publication/"
UA = "geo-cpt/0.1 (educational CPT project)"

_WS = re.compile(r"\s+")


def strip_html(text: str) -> str:
    """USGS abstracts arrive as HTML fragments (<h1>, <p>, &nbsp;)."""
    if not text:
        return ""
    if "<" in text:
        text = HTMLParser(text).text(separator=" ")
    return _WS.sub(" ", text.replace("\xa0", " ")).strip()


@retry(wait=wait_exponential(min=2, max=30), stop=stop_after_attempt(4), reraise=True)
def _get(client: httpx.Client, params: dict) -> dict:
    r = client.get(USGS_API, params=params)
    r.raise_for_status()
    return r.json()


def fetch_usgs(queries: list[str], max_per_query: int,
               cache: Path | None = None) -> list[dict]:
    """Abstracts from the USGS Publications Warehouse.

    These are real technical geoscience prose written by USGS scientists, and
    as a US Government work they are public domain. Roughly 300 tokens each.
    """
    cache = cache or (paths.RAW / "usgs.jsonl")
    if cache.exists():
        docs = [json.loads(l) for l in cache.open()]
        console.print(f"[dim]usgs cached: {len(docs)} docs[/dim]")
        return docs

    page_size = 100
    docs: list[dict] = []
    seen_ids: set[int] = set()
    with httpx.Client(timeout=45.0, headers={"User-Agent": UA},
                      follow_redirects=True) as client:
        with Progress(*Progress.get_default_columns(), TimeElapsedColumn(),
                      console=console) as prog:
            task = prog.add_task("usgs abstracts", total=len(queries))
            for q in queries:
                got = 0
                page = 1
                while got < max_per_query:
                    try:
                        payload = _get(client, {"q": q, "page_size": page_size,
                                                "page_number": page, "format": "json"})
                    except Exception as exc:
                        console.print(f"[yellow]usgs '{q}' p{page}: {exc}[/yellow]")
                        break
                    records = payload.get("records") or []
                    if not records:
                        break
                    for rec in records:
                        rid = rec.get("id")
                        if rid in seen_ids:
                            continue
                        body = strip_html(rec.get("docAbstract") or "")
                        if not body:
                            continue
                        seen_ids.add(rid)
                        docs.append({
                            "id": f"usgs:{rid}",
                            "source": "usgs",
                            "title": rec.get("title") or "",
                            "year": rec.get("publicationYear"),
                            "text": body,
                        })
                        got += 1
                    page += 1
                    if len(records) < page_size:
                        break
                prog.advance(task)

    cache.parent.mkdir(parents=True, exist_ok=True)
    with cache.open("w") as fh:
        for d in docs:
            fh.write(json.dumps(d, ensure_ascii=False) + "\n")
    console.print(f"[green]usgs[/green] {len(docs)} abstracts -> {cache}")
    return docs


def fetch_geolex_full(max_units: int) -> list[dict]:
    """The whole USGS lexicon, reusing geo-sft's cached fetcher.

    geo-sft already downloaded ~3,300 units and its HTTP cache is on disk, so
    those come back instantly; only the remainder hits the network.
    """
    from geosft import paths as sft_paths
    from geosft.data.fetch import fetch_details, fetch_index
    from geosft.data.http import CachedClient

    client = CachedClient(sft_paths.RAW / "http-cache")
    try:
        index = fetch_index(client, max_units)
        docs: list[dict] = []
        for rec in fetch_details(client, index):
            uid = rec.get("id")
            name = (rec.get("name") or "").strip()
            for i, ref in enumerate(rec.get("unit_reference_summaries") or []):
                body = " ".join(s for s in (ref.get("summary") or []) if s)
                body = _WS.sub(" ", body).strip()
                if not body:
                    continue
                docs.append({
                    "id": f"geolex:{uid}:{i}",
                    "source": "geolex",
                    "title": name,
                    "year": ref.get("reference_year"),
                    "text": body,
                })
    finally:
        client.close()
    console.print(f"[green]geolex[/green] {len(docs)} summaries")
    return docs


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
