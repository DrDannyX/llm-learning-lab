"""The extraction target.

Everything in this project points at one contract: messy geological prose in,
this object out, as JSON. Keeping the schema in one place means the labeller,
the prompt builder, the trainer and the evaluator can never drift apart.

Design notes that matter for SFT quality:

* Every list field is **canonically sorted and de-duplicated**. If the same
  input could map to two different orderings of the same answer, you are
  teaching the model to predict a coin flip. Determinism in the target is
  free accuracy.
* Vocabularies are **closed** where possible (lithology, chronostrat, rank).
  A closed vocabulary makes the task learnable from a few thousand examples
  and makes the metric meaningful.
* `None`/`[]` are legitimate answers. The model must learn to abstain, so the
  training set must contain examples where a field is genuinely absent.
"""

from __future__ import annotations

import json
from typing import Literal

from pydantic import BaseModel, Field, field_validator

#: ASUD's ranks, with Suite/Supersuite kept distinct from Group/Supergroup:
#: they are the same rank, but the NAME says Suite, and the label reads the name.
RankT = Literal[
    "Supergroup", "Supersuite", "Group", "Suite", "Subgroup",
    "Formation", "Member", "Bed", "Unknown",
]

RelationT = Literal[
    "overlies", "underlies", "unconformable_on", "grades_into",
    "intertongues_with", "equivalent_to", "intrudes", "intruded_by",
]


class Thickness(BaseModel):
    """Reported thickness, always normalised to metres."""
    min_m: float | None = None
    max_m: float | None = None


class Relation(BaseModel):
    """A stratigraphic relationship to another named unit."""
    kind: RelationT
    unit: str


class GeoExtraction(BaseModel):
    """Structured facts extracted from a paragraph of geological description."""

    unit_name: str | None = Field(
        default=None, description="Principal stratigraphic unit the passage is about."
    )
    rank: RankT = Field(default="Unknown", description="Lithostratigraphic rank.")
    lithologies: list[str] = Field(
        default_factory=list, description="Rock types, Macrostrat lithology vocabulary."
    )
    chronostrat: list[str] = Field(
        default_factory=list, description="Chronostratigraphic intervals, ICS names."
    )
    minerals: list[str] = Field(default_factory=list, description="Named minerals.")
    thickness: Thickness | None = Field(default=None, description="Reported thickness in metres.")
    relations: list[Relation] = Field(
        default_factory=list, description="Stratigraphic relations to other units."
    )
    states: list[str] = Field(
        default_factory=list,
        description="Australian state/territory codes (NSW, QLD, ...) NAMED IN THE PASSAGE.",
    )

    # --- canonicalisation: identical facts must always serialise identically ---
    @field_validator("lithologies", "chronostrat", "minerals", "states")
    @classmethod
    def _sorted_unique(cls, v: list[str]) -> list[str]:
        return sorted({s.strip() for s in v if s and s.strip()})

    @field_validator("relations")
    @classmethod
    def _sorted_relations(cls, v: list[Relation]) -> list[Relation]:
        seen: dict[tuple[str, str], Relation] = {}
        for r in v:
            seen[(r.kind, r.unit.lower())] = r
        return [seen[k] for k in sorted(seen)]

    def to_json(self, indent: int | None = None) -> str:
        """Compact, key-stable JSON. This exact string is the training target."""
        return json.dumps(
            self.model_dump(mode="json", exclude_none=False),
            indent=indent,
            ensure_ascii=False,
            sort_keys=False,
        )

    def is_empty(self) -> bool:
        """True when nothing beyond the unit name was found (used to filter trivia)."""
        return not (
            self.lithologies or self.chronostrat or self.minerals
            or self.relations or self.thickness
        )


#: Shown to the model so it knows the output contract. Kept short on purpose:
#: a long schema dump eats context that the passage needs, and the model learns
#: the shape from examples far faster than from prose.
SCHEMA_HINT = """{
  "unit_name": string|null, "rank": one of [Supergroup,Supersuite,Group,Suite,Subgroup,Formation,Member,Bed,Unknown],
  "lithologies": [string], "chronostrat": [string], "minerals": [string],
  "thickness": {"min_m": number|null, "max_m": number|null}|null,
  "relations": [{"kind": one of [overlies,underlies,unconformable_on,grades_into,intertongues_with,equivalent_to,intrudes,intruded_by], "unit": string}],
  "states": [one of NSW,QLD,VIC,TAS,SA,WA,NT,ACT,ATA]
}"""

SYSTEM_PROMPT = (
    "You are a geoscience information-extraction engine. Read the passage from "
    "Australia's stratigraphic lexicon and return ONLY a JSON object matching this schema:\n"
    f"{SCHEMA_HINT}\n"
    "Use the exact wording of the source for names. Omit nothing that is stated; "
    "invent nothing that is not. Sort every list alphabetically."
)


def build_messages(passage: str, target: GeoExtraction | None = None) -> list[dict[str, str]]:
    """Chat-format one example. Used identically at train and inference time.

    Train/inference prompt symmetry is the single most common source of silent
    fine-tuning failure, so both paths call this one function.
    """
    msgs = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Passage:\n{passage.strip()}"},
    ]
    if target is not None:
        msgs.append({"role": "assistant", "content": target.to_json()})
    return msgs
