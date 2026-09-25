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
  (:Unit {key, name, full_name, rank, asud, has_text, age_text, thickness_min_m, thickness_max_m})
      key is the unique id, e.g. 'asud:332'. rank is one of Supergroup, Supersuite,
      Group, Suite, Subgroup, Formation, Member, Bed, Unknown.
      asud=false marks units only mentioned in text (no curated metadata).
  (:Interval {name, rank, top_ma, base_ma})   geologic time; rank is age/epoch/period/era/eon/supereon
  (:Lithology {name})    lowercase rock type, e.g. 'sandstone', 'granite'
  (:Mineral {name})      e.g. 'Biotite'
  (:State {code, name})  Australian state/territory, e.g. {code:'QLD', name:'Queensland'}
  (:Province {name})     geological province/basin, e.g. 'Sydney Basin'

Relationships
  (:Unit)-[:PART_OF]->(:Unit)              Member -> Formation -> Group hierarchy
  (:Unit)-[:OVERLIES {unconformable, contact, sources}]->(:Unit)   stratigraphically above (there is no UNDERLIES)
  (:Unit)-[:INTRUDES]->(:Unit)             an intrusion and the unit it intrudes (there is no INTRUDED_BY)
  (:Unit)-[:EQUIVALENT_TO|GRADES_INTO|INTERTONGUES_WITH]->(:Unit)   lateral relations; match undirected
  (:Unit)-[:HAS_AGE]->(:Interval)          a unit's oldest and youngest ages
  (:Interval)-[:WITHIN]->(:Interval)       Statherian -> Paleoproterozoic -> Proterozoic ...
  (:Unit)-[:HAS_LITHOLOGY {sources}]->(:Lithology)
  (:Unit)-[:HAS_MINERAL]->(:Mineral)
  (:Unit)-[:OCCURS_IN]->(:State)
  (:Unit)-[:IN_PROVINCE]->(:Province)"""

EXAMPLES = """\
Q: What is the age of the Alsace Quartzite?   (linked key 'asud:332')
MATCH (u:Unit {key: 'asud:332'})-[:HAS_AGE]->(i:Interval)
RETURN u.full_name AS unit, collect(i.name) AS ages, u.age_text AS age_text

Q: What unit underlies the Alsace Quartzite?
MATCH (u:Unit {key: 'asud:332'})-[:OVERLIES]->(below:Unit)
RETURN below.full_name AS underlying_unit

Q: Which units make up the Mount Isa Group?   (linked key 'asud:12822')
MATCH (m:Unit)-[:PART_OF]->(g:Unit {key: 'asud:12822'})
RETURN m.full_name AS unit, m.rank AS rank

Q: Which Cambrian units in Tasmania contain limestone?
MATCH (u:Unit)-[:HAS_AGE]->(:Interval)-[:WITHIN*0..6]->(:Interval {name: 'Cambrian'})
MATCH (u)-[:OCCURS_IN]->(:State {code: 'TAS'})
MATCH (u)-[:HAS_LITHOLOGY]->(:Lithology {name: 'limestone'})
RETURN DISTINCT u.full_name AS unit

Q: How many Permian units occur in New South Wales?
MATCH (u:Unit)-[:HAS_AGE]->(:Interval)-[:WITHIN*0..6]->(:Interval {name: 'Permian'})
MATCH (u)-[:OCCURS_IN]->(:State {code: 'NSW'})
RETURN count(DISTINCT u) AS n"""

SYSTEM = f"""You translate questions about Australian geologic units into Neo4j Cypher.

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
