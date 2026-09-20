"""Corpus acquisition: the USGS Geologic Names Lexicon (Geolex).

Why this source, for a fine-tuning lab:

* It is genuinely **public domain** (a US Government work), so nothing here is
  licence-encumbered.
* The passages are **real geologist prose**, much of it written between 1890
  and 1980. It is abbreviated, inconsistent and full of archaic usage -- which
  is exactly the kind of text a domain model has to survive, and exactly what a
  general-purpose instruct model handles badly out of the box.
* Each passage arrives with **structured metadata** (unit name, age, rank,
  states, province) that we can use to anchor weak supervision instead of
  hallucinating labels with a teacher model.
"""
from __future__ import annotations

import json
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Iterator

from rich.console import Console
from rich.progress import Progress, TimeElapsedColumn

from .. import paths
from .http import CachedClient

console = Console()
GEOLEX = "https://ngmdb.usgs.gov/connect/apiv1/geolex/units/"
PAGE_SIZE = 100  # server-fixed


def fetch_index(client: CachedClient, max_units: int) -> list[dict]:
    """Page the Geolex unit index until we have at least `max_units` entries."""
    n_pages = -(-max_units // PAGE_SIZE)
    rows: list[dict] = []
    with Progress(*Progress.get_default_columns(), TimeElapsedColumn(),
                  console=console) as prog:
        task = prog.add_task("geolex index", total=n_pages)
        for page in range(1, n_pages + 1):
            payload = client.get_json(GEOLEX, {"page": page})
            results = payload.get("results") or []
            rows.extend(results)
            prog.advance(task)
            if not payload.get("next"):
                break
    return rows[:max_units]


def fetch_details(client: CachedClient, index_rows: list[dict],
                  workers: int = 6) -> Iterator[dict]:
    """Fetch the full record (including reference summaries) for each unit.

    Serially this is ~1.7s per unit -- nearly two hours for 4,000 units, almost
    all of it spent waiting on the network. A small thread pool fixes that.
    `workers` is deliberately modest: this is a free public API run by a
    government agency, and the per-request delay in CachedClient still applies
    inside each worker, so the aggregate rate stays civil.
    """
    uids = [row["id"] for row in index_rows if row.get("id") is not None]

    def one(uid: int):
        try:
            return client.get_json(f"{GEOLEX}{uid}/")
        except Exception as exc:  # a single 500 must not kill a 4,000-unit run
            console.print(f"[yellow]skip unit {uid}: {exc}[/yellow]")
            return None

    with Progress(*Progress.get_default_columns(), TimeElapsedColumn(),
                  console=console) as prog:
        task = prog.add_task("geolex details", total=len(uids))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for rec in pool.map(one, uids):
                prog.advance(task)
                if rec is not None:
                    yield rec


#: Lexicon prose is littered with bracketed editorial asides and page refs.
#: Stripping them raises the signal-to-noise ratio of the input without
#: removing any fact we ask the model to extract.
_PGREF = re.compile(r"^\s*Pg\.\s*[\d\-–,\s]+\.?\s*", re.I)
_WS = re.compile(r"\s+")


def clean_passage(text: str) -> str:
    text = _PGREF.sub("", text or "")
    return _WS.sub(" ", text).strip()


def extract_passages(details: Iterator[dict], cfg_min: int, cfg_max: int) -> list[dict]:
    """Flatten unit records into one passage per reference summary.

    `unit_id` is carried through so the splitter can keep every passage about a
    given unit inside a single split (see data/build.py -- leakage control).
    """
    out: list[dict] = []
    for rec in details:
        uid = rec.get("id")
        unit_name = (rec.get("name") or "").strip()
        ages = rec.get("age_description") or []
        usages = rec.get("usages") or []
        states = sorted({
            s for u in usages for s in (u.get("states") or []) if isinstance(s, str)
        })
        usage_strings = [u.get("usage", "") for u in usages if u.get("usage")]

        for ref in rec.get("unit_reference_summaries") or []:
            body = " ".join(s for s in (ref.get("summary") or []) if s)
            passage = clean_passage(body)
            if not (cfg_min <= len(passage) <= cfg_max):
                continue
            out.append({
                "unit_id": uid,
                "unit_name": unit_name,
                "passage": passage,
                # --- anchors for the labeller, never shown to the model ---
                "age_description": ages,
                "usages": usage_strings,
                "states": states,
                "ref_lithology": ref.get("lithology") or [],
                "ref_province": ref.get("province") or [],
                "ref_year": ref.get("reference_year"),
                "ref_publication": ref.get("publication"),
                "source_url": rec.get("url") or f"https://ngmdb.usgs.gov/Geolex/Units/{uid}",
            })
    return out


def run(max_units: int, min_chars: int, max_chars: int, force: bool = False) -> Path:
    paths.ensure()
    out = paths.INTERIM / "passages.jsonl"
    if out.exists() and not force:
        n = sum(1 for _ in out.open())
        console.print(f"[dim]passages cached ({n}) -> {out}[/dim]")
        return out

    client = CachedClient(paths.RAW / "http-cache")
    try:
        index = fetch_index(client, max_units)
        console.print(f"indexed [cyan]{len(index)}[/cyan] Geolex units")
        passages = extract_passages(fetch_details(client, index), min_chars, max_chars)
    finally:
        console.print(f"[dim]http cache: {client.hits} hits / {client.misses} misses[/dim]")
        client.close()

    with out.open("w") as fh:
        for p in passages:
            fh.write(json.dumps(p, ensure_ascii=False) + "\n")
    console.print(
        f"[green]passages[/green] {len(passages)} from "
        f"{len({p['unit_id'] for p in passages})} units -> {out}"
    )
    return out
