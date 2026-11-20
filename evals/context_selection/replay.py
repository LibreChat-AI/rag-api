"""Replay fixed, already-authorized passages. Run with python -m ...replay."""

import argparse
import asyncio
from dataclasses import asdict
import hashlib
import json
import os
import platform
from pathlib import Path
import time

from .core import Budget, Case, Passage, bm25, measure, number, select


def load_cases(path: Path) -> tuple[tuple[Case, ...], str]:
    raw = path.read_bytes()
    try:
        data = json.loads(raw)
        cases = []
        for item in data:
            if not isinstance(item["id"], str) or not isinstance(item["query"], str):
                raise ValueError
            required = item["required"]
            if not isinstance(required, list) or any(
                not isinstance(x, str) for x in required
            ):
                raise ValueError
            passages = []
            for p in item["passages"]:
                if any(not isinstance(p[key], str) for key in ("id", "source", "text")):
                    raise ValueError
                supports = p.get("supports", [])
                if not isinstance(supports, list) or any(
                    not isinstance(x, str) for x in supports
                ):
                    raise ValueError
                passages.append(
                    Passage(p["id"], p["source"], p["text"], tuple(supports))
                )
            cases.append(
                Case(item["id"], item["query"], tuple(required), tuple(passages))
            )
        if not cases or len({case.id for case in cases}) != len(cases):
            raise ValueError
        return tuple(cases), hashlib.sha256(raw).hexdigest()
    except (KeyError, TypeError, ValueError, AttributeError):
        raise ValueError("invalid corpus schema or evidence labels") from None


async def remote_scores(case: Case, name: str, args) -> dict:
    if not args.allow_network:
        return {"status": "skipped", "reason": "network_disabled"}
    # Dedicated experiment credentials, never app .env files or application globals.
    if name == "embeddings":
        endpoint = os.getenv("CONTEXT_EMBEDDING_URL")
        key = os.getenv("CONTEXT_EMBEDDING_KEY")
        model = os.getenv("CONTEXT_EMBEDDING_MODEL")
    else:
        endpoint = os.getenv("CONTEXT_JEV_URL", "https://api.typesafe.ai/v1/systemone")
        key = os.getenv("TYPESAFE_API_KEY")
        model = os.getenv("CONTEXT_JEV_MODEL", "jev-latest")
    if not endpoint or not key or not model:
        return {"status": "skipped", "reason": "missing_provider_configuration"}
    import httpx
    from .providers import HTTPScorer, ProviderError

    started = time.perf_counter()
    scorer = None
    try:
        async with httpx.AsyncClient() as client:
            scorer = HTTPScorer(
                client,
                endpoint,
                model,
                key,
                concurrency=args.concurrency,
                timeout=args.timeout,
            )
            result = await getattr(scorer, name)(case.query, case.passages)
        return {
            "status": "ok",
            **asdict(result),
            "elapsed_ms": (time.perf_counter() - started) * 1000,
        }
    except ProviderError as error:
        return {
            "status": "error",
            "reason": error.code,
            "requests": scorer.requests if scorer else 0,
            "reported_input_tokens": scorer.reported_input_tokens if scorer else 0,
            "input_tokens": None,
        }
    except TimeoutError:
        return {
            "status": "error",
            "reason": "deadline_exceeded",
            "requests": scorer.requests if scorer else 0,
            "reported_input_tokens": scorer.reported_input_tokens if scorer else 0,
            "input_tokens": None,
        }
    except (ValueError, httpx.HTTPError):
        return {"status": "error", "reason": "invalid_provider_configuration"}


def summaries(rows: list[dict]) -> list[dict]:
    result = []
    for strategy in dict.fromkeys(row["strategy"] for row in rows):
        group = [row for row in rows if row["strategy"] == strategy]
        ok = [row for row in group if row["status"] == "ok"]

        def average(field):
            values = [row[field] for row in ok if row[field] is not None]
            return sum(values) / len(values) if values else None

        result.append(
            {
                "strategy": strategy,
                "completed": len(ok),
                "precision_cases": sum(
                    row["evidence_precision"] is not None for row in ok
                ),
                "recall_cases": sum(row["evidence_recall"] is not None for row in ok),
                "skipped": sum(row["status"] == "skipped" for row in group),
                "errors": sum(row["status"] == "error" for row in group),
                "macro_precision": average("evidence_precision"),
                "macro_evidence_recall": average("evidence_recall"),
                "mean_selected_chars": average("selected_chars"),
                "answerable_cases_missing_evidence": sum(
                    bool(row["missing_evidence"]) for row in ok
                ),
                "correct_abstentions": sum(
                    row["correct_abstention"] is True for row in ok
                ),
            }
        )
    return result


async def replay(args) -> dict:
    cases, fingerprint = load_cases(args.corpus)
    budget = Budget(args.max_results, args.max_chars)
    if len(args.scorers) != len(set(args.scorers)):
        raise ValueError("scorers must be unique")
    if not 0 <= number(args.relative) <= 1 or not 0 <= number(args.jev_min) <= 3:
        raise ValueError("invalid selection threshold")
    if not -1 <= number(args.cosine_min) <= 1:
        raise ValueError("cosine threshold must be in [-1, 1]")
    if args.concurrency < 1 or number(args.timeout) <= 0:
        raise ValueError("concurrency and timeout must be positive")
    if args.max_candidates < 1 or args.max_input_chars < 1:
        raise ValueError("input limits must be positive")
    if any(
        len(case.passages) > args.max_candidates
        or len(case.query) + sum(len(p.text) for p in case.passages)
        > args.max_input_chars
        for case in cases
    ):
        raise ValueError("corpus exceeds per-case input limits; no provider was called")
    cached = {}
    if args.reuse_scores:
        saved = json.loads(args.reuse_scores.read_text())
        if (
            saved.get("corpus_sha256") != fingerprint
            or saved.get("schema_version") != 1
        ):
            raise ValueError("saved scores belong to a different corpus/schema")
        if not isinstance(saved.get("score_runs"), list):
            raise ValueError("invalid saved score runs")
        pool_sizes = {case.id: len(case.passages) for case in cases}
        for run in saved["score_runs"]:
            if run.get("status") not in ("ok", "skipped", "error"):
                raise ValueError("invalid saved status")
            if run["case"] not in pool_sizes or run["scorer"] not in (
                "bm25",
                "embeddings",
                "jev",
            ):
                raise ValueError("invalid saved scoring identity")
            if run["status"] == "ok":
                values = run.get("values")
                if (
                    not isinstance(values, list)
                    or len(values) != pool_sizes[run["case"]]
                ):
                    raise ValueError("invalid saved score count")
                values = [number(value) for value in values]
                if run["scorer"] == "jev" and any(
                    not 0 <= value <= 3 for value in values
                ):
                    raise ValueError("invalid saved usefulness")
                if run["scorer"] == "embeddings" and any(
                    not -1 <= value <= 1 for value in values
                ):
                    raise ValueError("invalid saved cosine")
            key = (run["case"], run["scorer"])
            if key in cached:
                raise ValueError("duplicate saved scoring run")
            cached[key] = run
    rows, score_runs = [], []
    for case in cases:
        for strategy, cap in [
            ("all", len(case.passages)),
            ("input-topk", budget.max_results),
        ]:
            rows.append(
                {
                    "case": case.id,
                    "strategy": strategy,
                    "status": "ok",
                    **measure(
                        case, select(case.passages, Budget(cap, budget.max_chars))
                    ),
                }
            )
        for name in args.scorers:
            if args.reuse_scores and name != "bm25":
                run = dict(
                    cached.get(
                        (case.id, name),
                        {"status": "skipped", "reason": "no_saved_scores"},
                    )
                )
                run.update({"requests": 0, "input_tokens": 0, "origin": "saved_scores"})
            elif name == "bm25":
                started = time.perf_counter()
                run = {
                    "status": "ok",
                    "values": bm25(case.query, case.passages),
                    "model": "bm25-k1=1.5-b=0.75-nfkc",
                    "requests": 0,
                    "input_tokens": 0,
                    "elapsed_ms": (time.perf_counter() - started) * 1000,
                }
            else:
                run = await remote_scores(case, name, args)
            run.update({"case": case.id, "scorer": name})
            score_runs.append(run)
            policies = [(f"{name}-topk", {})]
            policies.append(
                (
                    f"{name}-filter",
                    (
                        {"relative": args.relative}
                        if name == "bm25"
                        else {
                            "minimum": (
                                args.jev_min if name == "jev" else args.cosine_min
                            )
                        }
                    ),
                )
            )
            for strategy, threshold in policies:
                row = {"case": case.id, "strategy": strategy, "status": run["status"]}
                if run["status"] == "ok":
                    selected = select(
                        case.passages,
                        budget,
                        tuple(run["values"]),
                        deduplicate=args.deduplicate,
                        **threshold,
                    )
                    row.update(measure(case, selected))
                else:
                    row["reason"] = run["reason"]
                rows.append(row)
    return {
        "schema_version": 1,
        "implementation_sha256": hashlib.sha256(
            b"".join(
                Path(__file__).with_name(name).read_bytes()
                for name in ("core.py", "providers.py", "replay.py")
            )
        ).hexdigest(),
        "python_version": platform.python_version(),
        "corpus_sha256": fingerprint,
        "config": {
            "max_results": budget.max_results,
            "max_chars": budget.max_chars,
            "deduplicate": args.deduplicate,
            "relative": args.relative,
            "jev_min": args.jev_min,
            "cosine_min": args.cosine_min,
            "concurrency": args.concurrency,
            "timeout": args.timeout,
            "max_candidates": args.max_candidates,
            "max_input_chars": args.max_input_chars,
        },
        "score_runs": score_runs,
        "rows": rows,
        "summary": summaries(rows),
    }


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument(
        "--corpus", type=Path, default=Path(__file__).with_name("cases.json")
    )
    result.add_argument("--output", type=Path)
    result.add_argument("--reuse-scores", type=Path)
    result.add_argument(
        "--scorers",
        nargs="+",
        choices=["bm25", "embeddings", "jev"],
        default=["bm25", "embeddings", "jev"],
    )
    result.add_argument("--allow-network", action="store_true")
    result.add_argument("--deduplicate", action="store_true")
    result.add_argument("--max-results", type=int, default=10)
    result.add_argument("--max-chars", type=int, default=12000)
    result.add_argument("--relative", type=float, default=0.5)
    result.add_argument("--jev-min", type=float, default=1.5)
    result.add_argument("--cosine-min", type=float, default=0.35)
    result.add_argument("--concurrency", type=int, default=4)
    result.add_argument("--timeout", type=float, default=30)
    result.add_argument("--max-candidates", type=int, default=256)
    result.add_argument("--max-input-chars", type=int, default=256000)
    return result


def main():
    arguments = parser()
    args = arguments.parse_args()
    try:
        report = asyncio.run(replay(args))
    except (ValueError, KeyError, TypeError, OSError):
        arguments.error("invalid corpus, saved scores, configuration or output path")
    rendered = json.dumps(report, indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.write_text(rendered)
        print(json.dumps(report["summary"], indent=2))
    else:
        print(rendered, end="")
    # A failed requested scorer must not be presented as a successful experiment.
    if any(row["status"] == "error" for row in report["rows"]):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
