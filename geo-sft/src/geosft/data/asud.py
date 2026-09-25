"""Geoscience Australia's Australian Stratigraphic Units Database (ASUD).

ASUD is the national authority on Australian stratigraphic names, maintained by
Geoscience Australia for the Australian Stratigraphy Commission. It is released
under **CC BY 4.0** -- free to use, including for training, but attribution is
required: "(c) Commonwealth of Australia (Geoscience Australia)".

It is reached two ways, and this lab needs both:

  WFS      services.ga.gov.au/gis/stratunits   the documented OGC web service.
           18k units with name, rank, lithology class, age class, hierarchy
           and state. Stable and paged -- but DESCRIPTION is truncated to 255
           characters and ages are era-level only, so it holds almost no prose.
  reports  per-jurisdiction ZIP downloads     the "download state reports"
           linked from asud.ga.gov.au. Six pipe-delimited tables per
           jurisdiction, rebuilt weekly: the long prose (definition cards,
           per-reference comments), CURATED relations (overlies, intrudes,
           ...), thickness in metres and fine-grained ages.

The WFS is the unit index; the reports are the text and the curated facts.
Both are joined on STRATNO, ASUD's permanent unit number.

REPRODUCIBILITY
---------------
The reports are rebuilt every week, so "re-download" is not "reproduce". The
ZIPs are snapshotted once into data/raw/asud/ with a manifest of SHA-256 hashes
and server Last-Modified dates. Everything downstream reads the snapshot; only
`--force` refetches. Record the manifest next to any number you publish.
"""
from __future__ import annotations

import hashlib
import json
import re
import zipfile
from collections import defaultdict
from dataclasses import dataclass, field
from email.utils import parsedate_to_datetime
from functools import lru_cache
from pathlib import Path

import httpx
from rich.console import Console
from tenacity import retry, stop_after_attempt, wait_exponential

from .. import paths
from .http import UA

console = Console()

WFS = "https://services.ga.gov.au/gis/stratunits/wfs"
WFS_TYPE = "stratunit:StratigraphicUnit"
#: The report ZIPs linked from the ASUD site's download page.
REPORTS = ("https://objectstorage.ap-sydney-1.oraclecloud.com/n/sddylhldbjue"
           "/b/stratigraphic-reports/o")
#: ASUD's jurisdiction codes. OFF = offshore (Commonwealth waters), ATA =
#: Australian Antarctic Territory.
JURISDICTIONS = ["ACT", "ATA", "NSW", "NT", "OFF", "QLD", "SA", "TAS", "VIC", "WA"]
UNIT_URL = "https://asud.ga.gov.au/search-stratigraphic-units/results/{}"
ATTRIBUTION = ("Australian Stratigraphic Units Database, (c) Commonwealth of Australia "
               "(Geoscience Australia), CC BY 4.0")

#: WFS attributes this lab reads. GA warns that attribute names in this service
#: have changed before, so they are checked against DescribeFeatureType at run
#: time and a rename fails loudly instead of silently producing empty fields.
WFS_FIELDS = ["STRATNO", "NAME", "RANK", "LITHOLOGY", "STATE", "OLDERAGE", "YOUNGERAGE",
              "NUMERICOLDERAGE", "NUMERICYOUNGERAGE", "DESCRIPTION"]
WFS_PAGE = 2000  # the server default is 1,000,000 -- never page without a count

ROOT = paths.RAW / "asud"


# --------------------------------------------------------------------------
# download
# --------------------------------------------------------------------------

@retry(wait=wait_exponential(min=2, max=30), stop=stop_after_attempt(4), reraise=True)
def _get(client: httpx.Client, url: str, params: dict | None = None) -> httpx.Response:
    r = client.get(url, params=params)
    r.raise_for_status()
    return r


def _check_wfs_schema(client: httpx.Client) -> None:
    xsd = _get(client, WFS, {"service": "WFS", "version": "2.0.0",
                             "request": "DescribeFeatureType", "typeNames": WFS_TYPE}).text
    have = set(re.findall(r'name="([A-Z_]+)"', xsd))
    missing = [f for f in WFS_FIELDS if f not in have]
    if missing:
        raise RuntimeError(f"ASUD WFS no longer exposes {missing}; GA has renamed attributes. "
                           f"Available: {sorted(have)}")


def fetch_wfs_units(client: httpx.Client) -> list[dict]:
    """Every unit from the WFS, paged and sorted.

    `sortBy` matters: GA's service reports that paging is not
    transaction-safe, and unsorted paging can skip or repeat rows.
    """
    _check_wfs_schema(client)
    rows: list[dict] = []
    start = 0
    while True:
        payload = _get(client, WFS, {
            "service": "WFS", "version": "2.0.0", "request": "GetFeature",
            "typeNames": WFS_TYPE, "outputFormat": "application/json",
            "sortBy": "STRATNO", "count": WFS_PAGE, "startIndex": start,
        }).json()
        feats = payload.get("features") or []
        rows += [{k: f["properties"].get(k) for k in WFS_FIELDS} for f in feats]
        if len(feats) < WFS_PAGE:
            break
        start += WFS_PAGE
    return rows


def fetch(force: bool = False, not_current: bool = True) -> Path:
    """Snapshot the WFS unit table and the report ZIPs into data/raw/asud/.

    `not_current` also downloads reports for superseded names. Their prose is
    real geology and useful as pretraining text; the SFT and graph labs use
    current units only.
    """
    ROOT.mkdir(parents=True, exist_ok=True)
    manifest_p = ROOT / "manifest.json"
    if manifest_p.exists() and not force:
        console.print(f"[dim]ASUD snapshot cached -> {ROOT}[/dim]")
        return ROOT

    manifest: dict = {"attribution": ATTRIBUTION, "files": {}}
    with httpx.Client(timeout=180.0, headers={"User-Agent": UA}, follow_redirects=True) as c:
        units = fetch_wfs_units(c)
        (ROOT / "units_wfs.json").write_text(json.dumps(units))
        console.print(f"[green]ASUD WFS[/green] {len(units)} units")
        statuses = ["Current", "Notcurrent"] if not_current else ["Current"]
        for status in statuses:
            for j in JURISDICTIONS:
                name = f"{j}{status}.zip"
                r = _get(c, f"{REPORTS}/{name}")
                (ROOT / name).write_bytes(r.content)
                manifest["files"][name] = {
                    "sha256": hashlib.sha256(r.content).hexdigest(),
                    "bytes": len(r.content),
                    "last_modified": r.headers.get("last-modified"),
                }
        console.print(f"[green]ASUD reports[/green] {len(manifest['files'])} ZIPs")
    manifest_p.write_text(json.dumps(manifest, indent=2))
    return ROOT


# --------------------------------------------------------------------------
# parsing the report tables
# --------------------------------------------------------------------------

#: Report table -> the substring that identifies its file inside each ZIP.
#: NSW splits its articles over articles1..3, hence a prefix match.
TABLES = {
    "names": "Stratigraphic names",
    "definition": "Stratigraphic definition",
    "articles": "Stratigraphic articles",
    "references": "Stratigraphic references",
    "related": "Related Stratames",
    "provinces": "Preferred Provinces",
}

#: The related table names two columns "Related Stratno". Positional names
#: make the direction explicit: `<related> <relation> <subject>`, e.g.
#: "Cygnet Coal Measures | overlies | Abels Bay Formation".
RELATED_COLS = ["related_stratno", "related_name", "relation", "stratno", "name",
                "contact", "comments"]


def _decode(raw: bytes) -> str:
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        return raw.decode("cp1252", errors="replace")


def parse_table(text: str, columns: list[str] | None = None) -> tuple[list[dict], int]:
    """Parse one pipe-delimited report table.

    Free-text fields occasionally contain a newline, which splits a record over
    two physical lines. A line is a continuation when either the record being
    built is still short of fields, or the line itself is too short to be a
    record -- then it belongs to the previous record's final free-text field.
    Gluing it to the NEXT line instead would corrupt that record's STRATNO.
    A field containing a pipe gives a record too many fields; the surplus is
    folded into the last column, which is always free text.
    Returns (rows, unrecoverable_records).
    """
    lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    header = [h.strip() for h in lines[0].split("|")]
    cols = columns or header
    n = len(header)
    records: list[str] = []
    for line in lines[1:]:
        if not line.strip():
            continue
        if records and (records[-1].count("|") < n - 1 or line.count("|") < n - 1):
            records[-1] += " " + line
        else:
            records.append(line)

    rows: list[dict] = []
    bad = 0
    for rec in records:
        parts = rec.split("|")
        if len(parts) < n:
            bad += 1
            continue
        if len(parts) > n:
            parts = parts[:n - 1] + ["|".join(parts[n - 1:])]
        rows.append({c: re.sub(r"\s+", " ", p).strip() for c, p in zip(cols, parts)})
    return rows, bad


def read_table(table: str, status: str = "Current") -> list[dict]:
    """One table across every jurisdiction. Adds `_jurisdiction` to each row."""
    key = TABLES[table]
    cols = RELATED_COLS if table == "related" else None
    out: list[dict] = []
    for j in JURISDICTIONS:
        zp = ROOT / f"{j}{status}.zip"
        if not zp.exists():
            continue
        with zipfile.ZipFile(zp) as z:
            for member in sorted(z.namelist()):
                if key not in member:
                    continue
                rows, _ = parse_table(_decode(z.read(member)), cols)
                for r in rows:
                    r["_jurisdiction"] = j
                out += rows
    return out


# --------------------------------------------------------------------------
# the joined unit record
# --------------------------------------------------------------------------

def _int(s) -> int | None:
    try:
        return int(str(s).strip())
    except (TypeError, ValueError):
        return None


def _float(s) -> float | None:
    try:
        return float(str(s).replace(",", "").strip())
    except (TypeError, ValueError):
        return None


@dataclass
class Unit:
    """One ASUD unit: identity from the WFS, curated facts from the reports."""
    stratno: int
    name: str
    rank: str                        # ASUD rank, e.g. "Formation, beds", "Group, Suite"
    states: list[str]                # jurisdiction codes: NSW, QLD, ..., OFF, ATA
    lithology: str | None            # CGI lithology class, e.g. "acid volcanic rock"
    lithology_desc: str | None       # curated one-line description
    top_age: str | None              # youngest age name, e.g. "Calymmian"
    base_age: str | None             # oldest age name
    top_ma: float | None
    base_ma: float | None
    thickness_min_m: float | None
    thickness_max_m: float | None
    parent_stratno: int | None
    parent_name: str | None
    status: str | None               # Formal, Informal, Unnamed, Reserved, ...
    provinces: list[str] = field(default_factory=list)

    @property
    def url(self) -> str:
        return UNIT_URL.format(self.stratno)


@lru_cache(maxsize=1)
def load_units() -> dict[int, Unit]:
    """Current ASUD units keyed by STRATNO, WFS joined to the names report."""
    wfs = {r["STRATNO"]: r for r in json.loads((ROOT / "units_wfs.json").read_text())}
    names: dict[int, list[dict]] = defaultdict(list)
    for r in read_table("names"):
        if (sn := _int(r.get("Stratno"))) is not None:
            names[sn].append(r)
    provinces: dict[int, set[str]] = defaultdict(set)
    for r in read_table("provinces"):
        if (sn := _int(r.get("Stratno"))) is not None and r.get("Province Name"):
            provinces[sn].add(r["Province Name"])

    units: dict[int, Unit] = {}
    for sn in sorted(set(wfs) | set(names)):
        w = wfs.get(sn) or {}
        rs = names.get(sn) or [{}]
        r0 = rs[0]
        states = {s.strip() for s in (w.get("STATE") or "").split(",") if s.strip()}
        states |= {r["_jurisdiction"] for r in rs if r.get("_jurisdiction")}
        name = (r0.get("Stratigraphic Name") or w.get("NAME") or "").strip()
        if not name:
            continue
        units[int(sn)] = Unit(
            stratno=int(sn),
            name=name,
            rank=r0.get("Rank") or w.get("RANK") or "unknown",
            states=sorted(s for s in states if s in JURISDICTIONS),
            lithology=w.get("LITHOLOGY") or r0.get("Primary Lithology Group") or None,
            lithology_desc=r0.get("Lithology Description") or w.get("DESCRIPTION") or None,
            top_age=r0.get("Top Minimum Age Name") or w.get("YOUNGERAGE") or None,
            base_age=r0.get("Base Maximum Age Name") or w.get("OLDERAGE") or None,
            top_ma=_float(r0.get("Top Minimum Age (Ma)")) or _float(w.get("NUMERICYOUNGERAGE")),
            base_ma=_float(r0.get("Base Maximum Age (Ma)")) or _float(w.get("NUMERICOLDERAGE")),
            thickness_min_m=_float(r0.get("Minimum Thickness (m)")),
            thickness_max_m=_float(r0.get("Maximum Thickness (m)")),
            parent_stratno=_int(r0.get("Parent Stratno")),
            parent_name=r0.get("Parent Name") or None,
            status=r0.get("Status") or None,
            provinces=sorted(provinces.get(sn, ())),
        )
    return units


@dataclass(frozen=True)
class Related:
    """A curated relation, stored in reading order: `src <relation> dst`."""
    src: int
    relation: str       # overlies, underlies, is equivalent to, intrudes, ...
    dst: int
    contact: str        # conformity, unconformity, ... or ""


@lru_cache(maxsize=1)
def load_related() -> list[Related]:
    out: set[Related] = set()
    for r in read_table("related"):
        a, b = _int(r["related_stratno"]), _int(r["stratno"])
        rel = r["relation"].strip().lower()
        if a is None or b is None or a == b or rel in {"", "unknown"}:
            continue
        out.add(Related(a, rel, b, r["contact"].strip().lower()))
    return sorted(out, key=lambda x: (x.src, x.dst, x.relation))


@lru_cache(maxsize=1)
def load_references() -> dict[str, dict]:
    return {r["Reference Id"]: r for r in read_table("references") if r.get("Reference Id")}


def summary() -> dict:
    """What is in the snapshot -- printed by `geosft fetch` and recorded in the dataset card."""
    manifest = json.loads((ROOT / "manifest.json").read_text())
    units = load_units()
    return {
        "units": len(units),
        "related": len(load_related()),
        "snapshot_last_modified": max(
            (f["last_modified"] for f in manifest["files"].values() if f.get("last_modified")),
            key=parsedate_to_datetime, default=None),
        "attribution": ATTRIBUTION,
    }

