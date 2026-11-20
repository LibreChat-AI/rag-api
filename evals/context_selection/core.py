"""Experimental selection contracts. No application imports or provider calls."""

from collections import Counter
from dataclasses import dataclass
import math
import re
import unicodedata


@dataclass(frozen=True)
class Passage:
    id: str
    source: str
    text: str
    supports: tuple[str, ...] = ()


@dataclass(frozen=True)
class Case:
    id: str
    query: str
    required: tuple[str, ...]
    passages: tuple[Passage, ...]

    def __post_init__(self):
        if not self.id or not self.query.strip():
            raise ValueError("case id and query must be nonempty")
        ids = [p.id for p in self.passages]
        if len(ids) != len(set(ids)) or any(not value for value in ids):
            raise ValueError("passage ids must be nonempty and unique within a case")
        if len(self.required) != len(set(self.required)):
            raise ValueError("required evidence labels must be unique")
        labels = set(self.required)
        for passage in self.passages:
            if not passage.source or not passage.text.strip():
                raise ValueError("passages require a source and nonempty text")
            if not set(passage.supports) <= labels:
                raise ValueError("passage references an unknown evidence label")
        available = {label for p in self.passages for label in p.supports}
        if available != labels:
            raise ValueError(
                "every required label needs evidence in the candidate pool"
            )


@dataclass(frozen=True)
class Budget:
    max_results: int = 10
    max_chars: int = 12000

    def __post_init__(self):
        if self.max_results < 0 or self.max_chars < 0:
            raise ValueError("budgets must be nonnegative")


def tokens(text: str) -> list[str]:
    """Preserve identifiers, negations and Unicode; no English-only stemming."""
    return re.findall(r"\w+", unicodedata.normalize("NFKC", text).casefold())


def bm25(query: str, passages: tuple[Passage, ...]) -> tuple[float, ...]:
    """Standard BM25, k1=1.5 and b=0.75, with pool-local document frequency."""
    frequencies = [Counter(tokens(p.text)) for p in passages]
    if not frequencies:
        return ()
    lengths = [sum(frequency.values()) for frequency in frequencies]
    average = sum(lengths) / len(lengths) or 1.0
    terms = set(tokens(query))
    document_frequency = Counter(
        term for frequency in frequencies for term in terms if term in frequency
    )
    scores = []
    for frequency, length in zip(frequencies, lengths):
        score = 0.0
        for term in sorted(terms):
            count = frequency[term]
            if not count:
                continue
            df = document_frequency[term]
            idf = math.log1p((len(passages) - df + 0.5) / (df + 0.5))
            score += (
                idf * count * 2.5 / (count + 1.5 * (0.25 + 0.75 * length / average))
            )
        scores.append(score)
    return tuple(scores)


def number(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("expected a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError("expected a finite number")
    return result


def cosine(left: list[float], right: list[float]) -> float:
    if not left or len(left) != len(right):
        raise ValueError("embedding dimensions must match and be nonempty")
    left = [number(value) for value in left]
    right = [number(value) for value in right]
    a = math.sqrt(math.fsum(value * value for value in left))
    b = math.sqrt(math.fsum(value * value for value in right))
    if not math.isfinite(a * b) or a == 0 or b == 0:
        raise ValueError("embedding norms must be finite and nonzero")
    return max(-1.0, min(1.0, math.fsum(x * y for x, y in zip(left, right)) / (a * b)))


def select(
    passages: tuple[Passage, ...],
    budget: Budget,
    scores: tuple[float, ...] | None = None,
    *,
    minimum: float | None = None,
    relative: float | None = None,
    deduplicate: bool = False,
) -> tuple[Passage, ...]:
    """Select whole original passages; never pad an honestly empty selection.

    Stable ties use input order. The character budget includes passage bodies,
    not citation wrappers, and is not a tokenizer-derived prompt-token budget.
    """
    if scores is not None:
        if len(scores) != len(passages):
            raise ValueError("one score is required per passage")
        scores = tuple(number(score) for score in scores)
    if minimum is not None:
        minimum = number(minimum)
    if relative is not None:
        relative = number(relative)
        if not 0 <= relative <= 1:
            raise ValueError("relative threshold must be in [0, 1]")
    if (minimum is not None or relative is not None) and scores is None:
        raise ValueError("thresholds require scores")
    order = (
        sorted(range(len(passages)), key=lambda i: (-scores[i], i))
        if scores is not None
        else range(len(passages))
    )
    best = max(scores, default=0.0) if scores is not None else 0.0
    chosen = []
    used = 0
    seen = set()
    for index in order:
        if len(chosen) >= budget.max_results:
            break
        if scores is not None:
            if minimum is not None and scores[index] < minimum:
                continue
            if relative is not None and (best <= 0 or scores[index] < relative * best):
                continue
        passage = passages[index]
        # Exact duplicates only. Paraphrase/overlap merging needs separate evidence.
        if deduplicate and passage.text in seen:
            continue
        if used + len(passage.text) > budget.max_chars:
            continue
        chosen.append(passage)
        seen.add(passage.text)
        used += len(passage.text)
    return tuple(chosen)


def measure(case: Case, selected: tuple[Passage, ...]) -> dict:
    originals = {p.id: p for p in case.passages}
    if len({p.id for p in selected}) != len(selected) or any(
        originals.get(p.id) != p for p in selected
    ):
        raise ValueError(
            "selection must preserve original passage identity and content"
        )
    covered = {label for p in selected for label in p.supports}
    missing = sorted(set(case.required) - covered)
    return {
        "selected_ids": [p.id for p in selected],
        "sources": [p.source for p in selected],
        "selected_count": len(selected),
        "input_chars": sum(len(p.text) for p in case.passages),
        "selected_chars": sum(len(p.text) for p in selected),
        "evidence_precision": (
            sum(bool(p.supports) for p in selected) / len(selected)
            if selected
            else None
        ),
        "evidence_recall": (
            len(covered) / len(case.required) if case.required else None
        ),
        "missing_evidence": missing,
        "complete_evidence": not missing,
        "correct_abstention": not selected if not case.required else None,
    }
