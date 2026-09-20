"""Longest-match gazetteer tagging over word n-grams.

A 6,000-entry regex alternation is both slow and fragile. Tokenising once and
looking up n-grams in a dict is O(n * k) with a tiny constant, handles
multi-word terms ("lime mudstone", "New Mexico") and gives longest-match
semantics for free -- so "Late Cretaceous" wins over "Cretaceous".
"""
from __future__ import annotations

import re
from dataclasses import dataclass

WORD = re.compile(r"[A-Za-z][A-Za-z'\-]*")

#: Lexicon prose pluralises rock names freely ("shales", "marls").
_PLURAL_SUFFIXES = ("s", "es")


@dataclass(frozen=True)
class Hit:
    canonical: str
    start: int
    end: int
    surface: str


class Gazetteer:
    """Case-insensitive, longest-match phrase tagger."""

    def __init__(self, entries: dict[str, str], max_ngram: int = 4, plurals: bool = False):
        """entries maps a lowercase surface phrase -> canonical output form."""
        self.table: dict[str, str] = {}
        for surface, canonical in entries.items():
            key = surface.lower().strip()
            if not key:
                continue
            self.table.setdefault(key, canonical)
            if plurals:
                for suf in _PLURAL_SUFFIXES:
                    self.table.setdefault(key + suf, canonical)
        self.max_ngram = max_ngram

    def find(self, text: str) -> list[Hit]:
        toks = [(m.group(0), m.start(), m.end()) for m in WORD.finditer(text)]
        lowered = [t[0].lower() for t in toks]
        hits: list[Hit] = []
        i = 0
        n = len(toks)
        while i < n:
            matched = False
            # longest match first so "Late Cretaceous" beats "Cretaceous"
            for span in range(min(self.max_ngram, n - i), 0, -1):
                phrase = " ".join(lowered[i:i + span])
                canonical = self.table.get(phrase)
                if canonical is not None:
                    hits.append(Hit(canonical, toks[i][1], toks[i + span - 1][2],
                                    text[toks[i][1]:toks[i + span - 1][2]]))
                    i += span
                    matched = True
                    break
            if not matched:
                i += 1
        return hits

    def values(self, text: str) -> list[str]:
        return sorted({h.canonical for h in self.find(text)})
