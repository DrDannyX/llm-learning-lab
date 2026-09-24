"""Knowledge-graph QA: the model writes Cypher, Neo4j answers it.

This system sees NO passage text -- only the rows its query returns. When the
query is right it is the only system that can count, filter across thousands
of units, or walk a hierarchy exactly. When the query is wrong it fails
silently: an empty result looks exactly like "there are none".

Three things make text-to-Cypher workable with a small local model:
  1. a compact, exact schema in the prompt (not an auto-dump of every property),
  2. entity linking done beforehand, so the model is given keys, not names,
  3. one repair round: on an error or an empty result, the model sees what
     happened and tries again.
"""
from __future__ import annotations

import json
import re
import time

from .. import db
from ..config import Config
from ..llm import complete
from .base import Retrieval
from .link import linker

SCHEMA = """\
Nodes
  (:Unit {key, name, full_name, rank, geolex, age_text, thickness_min_m, thickness_max_m})
      key is the unique id, e.g. 'geolex:6304'. rank is one of Supergroup, Group,
      Subgroup, Formation, Member, Bed, Tongue, Lentil, Unknown.
      geolex=false marks units only mentioned in text (no curated metadata).
  (:Interval {name, rank, top_ma, base_ma})   geologic time; rank is age/epoch/period/era/eon
  (:Lithology {name})    lowercase rock type, e.g. 'shale', 'limestone'
  (:Mineral {name})      e.g. 'Pyrite'
  (:State {code, name})  e.g. {code:'TX', name:'Texas'}
  (:Province {name})     geologic province/basin, e.g. 'Permian basin'

Relationships
  (:Unit)-[:PART_OF]->(:Unit)              Member -> Formation -> Group hierarchy
  (:Unit)-[:OVERLIES {unconformable}]->(:Unit)   stratigraphically above (there is no UNDERLIES)
  (:Unit)-[:EQUIVALENT_TO|GRADES_INTO|INTERTONGUES_WITH]->(:Unit)   lateral relations; match undirected
  (:Unit)-[:HAS_AGE]->(:Interval)
  (:Interval)-[:WITHIN]->(:Interval)       Virgilian -> Pennsylvanian -> Carboniferous ...
  (:Unit)-[:HAS_LITHOLOGY {sources}]->(:Lithology)
  (:Unit)-[:HAS_MINERAL]->(:Mineral)
  (:Unit)-[:OCCURS_IN]->(:State)
  (:Unit)-[:IN_PROVINCE]->(:Province)"""

EXAMPLES = """\
Q: What is the age of the Aarde Shale Member?   (linked key 'geolex:6304')
MATCH (u:Unit {key: 'geolex:6304'})-[:HAS_AGE]->(i:Interval)
RETURN u.full_name AS unit, collect(i.name) AS ages, u.age_text AS age_text

Q: What unit underlies the Aarde Shale Member?
MATCH (u:Unit {key: 'geolex:6304'})-[:OVERLIES]->(below:Unit)
RETURN below.full_name AS underlying_unit

Q: Which units make up the Chickamauga Group?   (linked key 'geolex:1036')
MATCH (m:Unit)-[:PART_OF]->(g:Unit {key: 'geolex:1036'})
RETURN m.full_name AS unit, m.rank AS rank

Q: Which Cretaceous units in Texas contain chalk?
MATCH (u:Unit)-[:HAS_AGE]->(:Interval)-[:WITHIN*0..6]->(:Interval {name: 'Cretaceous'})
MATCH (u)-[:OCCURS_IN]->(:State {code: 'TX'})
MATCH (u)-[:HAS_LITHOLOGY]->(:Lithology {name: 'chalk'})
RETURN DISTINCT u.full_name AS unit

Q: How many Pennsylvanian units occur in Kansas?
MATCH (u:Unit)-[:HAS_AGE]->(:Interval)-[:WITHIN*0..6]->(:Interval {name: 'Pennsylvanian'})
MATCH (u)-[:OCCURS_IN]->(:State {code: 'KS'})
RETURN count(DISTINCT u) AS n"""

SYSTEM = f"""You translate questions about US geologic units into Neo4j Cypher.

Graph schema:
{SCHEMA}

Rules:
- Use ONLY the labels, relationship types and properties in the schema.
- When the question names a unit, match it by the exact key given under
  "Linked entities". Never match units by name when a key is given.
- For "units of <interval> age", traverse HAS_AGE then WITHIN*0..6 to the interval,
  so that sub-intervals count.
- Use DISTINCT for lists; use count(DISTINCT ...) for "how many".
- Return readable properties (full_name, name), never whole nodes.
- Read-only: MATCH / OPTIONAL MATCH / WITH / RETURN only.
- Output ONE query in a ```cypher code block and nothing else.

Examples:
{EXAMPLES}"""

_BLOCK = re.compile(r"```(?:cypher)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)


def extract_cypher(text: str) -> str:
    m = _BLOCK.search(text)
    q = (m.group(1) if m else text).strip().rstrip(";")
    return q


def format_rows(cypher: str, rows: list[dict], cap: int, linked: str) -> str:
    """Rows alone are unmoored -- `{"overlying_unit": "Church Member"}` does not
    say *what* it overlies. Showing the query and the linked entities is what
    lets the answer model read the rows correctly."""
    head = f"Entities in the question:\n{linked}\n\nGraph query:\n{cypher}\n\n"
    if not rows:
        return head + "The graph query returned no rows."
    lines = [json.dumps(r, ensure_ascii=False, default=str) for r in rows[:cap]]
    more = f"\n... and {len(rows) - cap} more rows" if len(rows) > cap else ""
    return head + f"Result ({len(rows)} rows):\n" + "\n".join(lines) + more


def retrieve(cfg: Config, question: str) -> Retrieval:
    t = time.perf_counter()
    linked = linker(cfg.neo4j).link(question)
    prompt = f"Linked entities:\n{linked.describe()}\n\nQuestion: {question}"
    attempts: list[dict] = []
    rows: list[dict] = []
    error = None
    usage: dict = {}
    for _ in range(1 + cfg.retrieval.kg_retries):
        text, u = complete(cfg.llm, SYSTEM, prompt)
        usage = {k: usage.get(k, 0) + v for k, v in u.items()}
        cypher = extract_cypher(text)
        try:
            rows = db.read_untrusted(cfg.neo4j, cypher, limit=500)
            error = None
        except Exception as e:  # noqa: BLE001 -- any Neo4j/validation error is feedback
            rows, error = [], f"{type(e).__name__}: {str(e)[:400]}"
        attempts.append({"cypher": cypher, "rows": len(rows), "error": error})
        if rows:
            break
        feedback = (f"That query failed with:\n{error}" if error
                    else "That query returned no rows. Check the relationship directions, "
                         "the linked keys, and whether a filter is too strict.")
        prompt += f"\n\nYour previous query:\n```cypher\n{cypher}\n```\n{feedback}\nWrite a corrected query."

    return Retrieval(
        system="kg",
        context=format_rows(attempts[-1]["cypher"], rows, cfg.retrieval.kg_max_rows,
                            linked.describe()),
        sources=[u["key"] for u in linked.units],
        debug={"linked": linked.describe(), "attempts": attempts, "cypher_usage": usage},
        seconds=time.perf_counter() - t,
        error=error,
    )
