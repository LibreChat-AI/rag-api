"""Synthetic/HTTP-emulated contract checks, not evidence of model quality."""

import asyncio
from dataclasses import replace
import json
from pathlib import Path
from unittest.mock import patch

import httpx
import pytest

from evals.context_selection.core import (
    Budget,
    Case,
    Passage,
    bm25,
    cosine,
    measure,
    select,
)
from evals.context_selection.providers import HTTPScorer, ProviderError
from evals.context_selection.replay import load_cases, parser, replay


P = (
    Passage("a", "source-a", "first fact", ("fact",)),
    Passage("b", "source-b", "second text"),
)


def test_empty_selection_is_success_not_fallback():
    case = Case("case", "question", ("fact",), P)
    result = measure(case, select(P, Budget(), (0.2, 0.3), minimum=1.5))
    assert result["selected_ids"] == []
    assert result["evidence_recall"] == 0
    assert result["evidence_precision"] is None
    assert result["missing_evidence"] == ["fact"]


def test_stable_ties_preserve_original_provenance():
    chosen = select(P, Budget(2), (2, 2))
    assert chosen == P
    assert chosen[0] is P[0]
    assert measure(Case("case", "query", ("fact",), P), chosen)["sources"] == [
        "source-a",
        "source-b",
    ]


def test_character_limit_keeps_whole_passages_not_truncated_evidence():
    chosen = select(P, Budget(10, len(P[1].text)), (1, 2))
    assert chosen == (P[1],)
    assert select(P, Budget(0), (1, 2)) == ()
    assert select(P, Budget(10, 0), (1, 2)) == ()


def test_exact_duplicate_suppression_is_explicit():
    duplicate = replace(P[0], id="c", source="mirror")
    assert len(select((P[0], duplicate), Budget(), (1, 1))) == 2
    assert select((P[0], duplicate), Budget(), (1, 1), deduplicate=True) == (P[0],)


@pytest.mark.parametrize(
    "scores", [(1,), (True, 1), (float("nan"), 1), (float("inf"), 1)]
)
def test_malformed_scores_rejected(scores):
    with pytest.raises(ValueError):
        select(P, Budget(), scores)


@pytest.mark.parametrize("value", [-1, 2, float("nan")])
def test_invalid_relative_threshold_rejected(value):
    with pytest.raises(ValueError):
        select(P, Budget(), (1, 2), relative=value)


def test_relative_filter_does_not_pad_zero_match():
    scores = bm25("unmatched-term", P)
    assert scores == (0, 0)
    assert select(P, Budget(), scores, relative=0.5) == ()
    assert len(select(P, Budget(), scores)) == 2


def test_bm25_preserves_negations_identifiers_and_unicode():
    passages = (
        Passage("a", "s", "BUG_1842 not deleted 保管"),
        Passage("b", "s", "unrelated"),
    )
    assert bm25("BUG_1842", passages)[0] > 0
    assert bm25("not", passages)[0] > 0
    assert bm25("保管", passages)[0] > 0


@pytest.mark.parametrize(
    "left,right",
    [([0, 0], [1, 1]), ([1], [1, 1]), ([float("nan")], [1]), ([True], [1])],
)
def test_invalid_vectors_rejected(left, right):
    with pytest.raises(ValueError):
        cosine(left, right)


def test_cosine_handles_non_unit_vectors():
    assert cosine([3, 4], [6, 8]) == pytest.approx(1)
    assert cosine([3, 4], [-6, -8]) == pytest.approx(-1)


def test_metric_detects_exception_loss_and_rewritten_citation():
    exception = replace(P[1], supports=("exception",))
    case = Case("case", "query", ("fact", "exception"), (P[0], exception))
    result = measure(case, (P[0],))
    assert result["evidence_precision"] == 1
    assert result["evidence_recall"] == 0.5
    assert result["missing_evidence"] == ["exception"]
    with pytest.raises(ValueError):
        measure(case, (replace(P[0], text="invented"),))


@pytest.mark.parametrize(
    "passages,required",
    [((P[0], P[0]), ("fact",)), (P, ("absent",)), (P, ("fact", "fact"))],
)
def test_invalid_evidence_corpus_rejected(passages, required):
    with pytest.raises(ValueError):
        Case("case", "query", required, passages)


def scorer(client, **kwargs):
    return HTTPScorer(
        client, "https://provider.invalid/v1", "model-test", "secret-test", **kwargs
    )


def jev_body(score):
    return {"answers": {"usefulness": {"score": score}}, "usage": {"input_tokens": 4}}


async def test_jev_uses_text_only_not_labels_and_preserves_request_order():
    payloads = []

    async def handler(request):
        payload = json.loads(request.content)
        payloads.append(payload)
        if payload["state"] == P[0].text:
            await asyncio.sleep(0.01)
        return httpx.Response(
            200, json=jev_body(2 if payload["state"] == P[0].text else 0)
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        result = await scorer(client).jev("query", P)
    assert result.values == (2, 0)
    assert result.input_tokens == 8
    assert result.requests == 2
    assert all("supports" not in p and "source" not in p for p in payloads)


async def test_concurrency_limit_shared_across_overlapping_calls():
    active = maximum = 0

    async def handler(request):
        nonlocal active, maximum
        active += 1
        maximum = max(maximum, active)
        try:
            await asyncio.sleep(0.01)
            return httpx.Response(200, json=jev_body(2))
        finally:
            active -= 1

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        adapter = scorer(client, concurrency=2)
        await asyncio.gather(adapter.jev("q1", P * 3), adapter.jev("q2", P * 3))
    assert maximum == 2
    assert active == 0


@pytest.mark.parametrize("score", [-1, 4, float("inf"), "2", True])
async def test_jev_rejects_malformed_scores(score):
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(
            lambda req: httpx.Response(200, json=jev_body(score))
        )
    ) as client:
        with pytest.raises(ProviderError, match="invalid_usefulness"):
            await scorer(client).jev("query", P)


async def test_failure_cancels_and_drains_siblings_then_retry_works():
    running = asyncio.Event()
    stopped = asyncio.Event()
    fail = True

    async def handler(request):
        if not fail:
            return httpx.Response(200, json=jev_body(2))
        payload = json.loads(request.content)
        if payload["state"] == P[1].text:
            running.set()
            try:
                await asyncio.Event().wait()
            finally:
                stopped.set()
        await running.wait()
        return httpx.Response(401, text="secret-test and private passage")

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        adapter = scorer(client)
        with pytest.raises(ProviderError) as caught:
            await adapter.jev("query", P)
        assert str(caught.value) == "http_401"
        assert stopped.is_set()
        fail = False
        assert (await adapter.jev("query", P)).values == (2, 2)


@pytest.mark.parametrize("cancel", [False, True])
async def test_deadline_and_cancellation_release_active_calls(cancel):
    active = 0
    started = asyncio.Event()

    async def handler(request):
        nonlocal active
        active += 1
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            active -= 1

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        task = asyncio.create_task(scorer(client, timeout=0.03).jev("query", P))
        await started.wait()
        if cancel:
            task.cancel()
        with pytest.raises(asyncio.CancelledError if cancel else TimeoutError):
            await task
    assert active == 0


async def test_embedding_response_indices_not_response_order():
    body = {
        "data": [
            {"index": 2, "embedding": [0, 1]},
            {"index": 0, "embedding": [2, 0]},
            {"index": 1, "embedding": [3, 0]},
        ],
        "usage": {"prompt_tokens": 9},
    }
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda req: httpx.Response(200, json=body))
    ) as client:
        result = await scorer(client).embeddings("query", P)
    assert result.values == (1, 0)
    assert result.input_tokens == 9


@pytest.mark.parametrize(
    "rows",
    [
        [],
        [{"index": 0, "embedding": [1, 0]}] * 3,
        [{"index": i, "embedding": [0, 0]} for i in range(3)],
    ],
)
async def test_bad_embedding_batches_rejected(rows):
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(
            lambda req: httpx.Response(200, json={"data": rows})
        )
    ) as client:
        with pytest.raises(ProviderError, match="invalid_embeddings"):
            await scorer(client).embeddings("query", P)


async def test_default_replay_never_constructs_network_client_even_with_credentials():
    args = parser().parse_args([])
    with patch.dict(
        "os.environ",
        {"TYPESAFE_API_KEY": "secret-test", "CONTEXT_EMBEDDING_KEY": "secret-test"},
    ), patch("httpx.AsyncClient", side_effect=AssertionError("network forbidden")):
        report = await replay(args)
    assert all(
        run["status"] == "skipped"
        for run in report["score_runs"]
        if run["scorer"] != "bm25"
    )
    assert all(
        row["status"] == "ok"
        for row in report["rows"]
        if row["strategy"].startswith("bm25")
    )
    assert "secret-test" not in json.dumps(report)


async def test_replay_reuses_saved_scores_for_ablation_without_network(tmp_path):
    args = parser().parse_args([])
    report = await replay(args)
    cases, _ = load_cases(args.corpus)
    for case in cases:
        report["score_runs"].append(
            {
                "case": case.id,
                "scorer": "jev",
                "status": "ok",
                "values": [2 if p.supports else 0 for p in case.passages],
                "model": "FAKE-NOT-JEV",
                "requests": 99,
            }
        )
    report["score_runs"] = [
        r
        for r in report["score_runs"]
        if not (r["scorer"] == "jev" and r["status"] == "skipped")
    ]
    saved = tmp_path / "scores.json"
    saved.write_text(json.dumps(report))
    args.reuse_scores = saved
    with patch("httpx.AsyncClient", side_effect=AssertionError("network forbidden")):
        reused = await replay(args)
    filtered = [r for r in reused["rows"] if r["strategy"] == "jev-filter"]
    assert all(row["complete_evidence"] for row in filtered)
    assert all(
        run["requests"] == 0 for run in reused["score_runs"] if run["scorer"] == "jev"
    )
    report["corpus_sha256"] = "different"
    saved.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="different corpus"):
        await replay(args)


async def test_input_limits_reject_before_provider_use():
    args = parser().parse_args(["--max-candidates", "1", "--allow-network"])
    with patch(
        "httpx.AsyncClient", side_effect=AssertionError("network forbidden")
    ), pytest.raises(ValueError, match="input limits"):
        await replay(args)


async def test_empty_provider_pools_make_no_http_requests():
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda req: pytest.fail("unexpected inference"))
    ) as client:
        adapter = scorer(client)
        assert (await adapter.embeddings("query", ())).requests == 0
        assert (await adapter.jev("query", ())).requests == 0
        assert adapter.requests == 0


async def test_jev_records_resolved_model_and_unknown_usage():
    body = {"model": "resolved-model", "answers": {"usefulness": {"score": 2}}}
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda req: httpx.Response(200, json=body))
    ) as client:
        result = await scorer(client).jev("query", P)
    assert result.model == "resolved-model"
    assert result.input_tokens is None


async def test_jev_rejects_mixed_resolved_models():
    async def handler(request):
        body = jev_body(2)
        body["model"] = json.loads(request.content)["state"]
        return httpx.Response(200, json=body)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(ProviderError, match="inconsistent_model"):
            await scorer(client).jev("query", P)


@pytest.mark.parametrize("status", [302, 429, 529])
async def test_provider_errors_are_not_retried_or_redirected(status):
    calls = 0

    def handler(request):
        nonlocal calls
        calls += 1
        return httpx.Response(
            status, headers={"location": "https://other.invalid"}, text="secret-test"
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        adapter = scorer(client, concurrency=1)
        with pytest.raises(ProviderError, match=f"http_{status}"):
            await adapter.jev("query", P)
        assert adapter.requests == calls == 1


async def test_missing_live_credentials_are_skipped_before_client_creation():
    args = parser().parse_args(["--allow-network"])
    with patch.dict("os.environ", {}, clear=True), patch(
        "httpx.AsyncClient", side_effect=AssertionError("network forbidden")
    ):
        report = await replay(args)
    assert all(
        run["reason"] == "missing_provider_configuration"
        for run in report["score_runs"]
        if run["scorer"] != "bm25"
    )


@pytest.mark.parametrize("change", ["status", "count", "range", "duplicate"])
async def test_saved_score_validation_rejects_corruption(tmp_path, change):
    args = parser().parse_args([])
    report = await replay(args)
    cases, _ = load_cases(args.corpus)
    run = {
        "case": cases[0].id,
        "scorer": "jev",
        "status": "ok",
        "values": [2] * len(cases[0].passages),
    }
    report["score_runs"] = [run]
    if change == "status":
        run["status"] = "unknown"
    elif change == "count":
        run["values"] = []
    elif change == "range":
        run["values"][0] = 4
    else:
        report["score_runs"].append(run)
    saved = tmp_path / "corrupt.json"
    saved.write_text(json.dumps(report))
    args.reuse_scores = saved
    with pytest.raises(ValueError):
        await replay(args)


def test_cli_reports_failed_provider_without_claiming_success(
    monkeypatch, tmp_path, capsys
):
    from evals.context_selection import replay as module

    async def failed(*args):
        return {
            "status": "error",
            "reason": "http_429",
            "requests": 1,
            "input_tokens": None,
        }

    monkeypatch.setattr(module, "remote_scores", failed)
    output = tmp_path / "report.json"
    monkeypatch.setattr(
        "sys.argv", ["replay", "--scorers", "jev", "--output", str(output)]
    )
    with pytest.raises(SystemExit) as caught:
        module.main()
    assert caught.value.code == 1
    report = json.loads(output.read_text())
    assert all(
        row["status"] == "error"
        for row in report["rows"]
        if row["strategy"].startswith("jev")
    )
    assert "secret-test" not in capsys.readouterr().out


async def test_failed_batch_exposes_attempt_counts_not_provider_body(monkeypatch):
    from evals.context_selection import replay as module

    class Client(httpx.AsyncClient):
        def __init__(self):
            super().__init__(
                transport=httpx.MockTransport(
                    lambda req: httpx.Response(429, text="secret-test")
                )
            )

    args = parser().parse_args(["--allow-network", "--concurrency", "1"])
    monkeypatch.setenv("TYPESAFE_API_KEY", "secret-test")
    monkeypatch.setattr("httpx.AsyncClient", Client)
    result = await module.remote_scores(
        Case("case", "query", ("fact",), P), "jev", args
    )
    assert result["status"] == "error"
    assert result["requests"] == 1
    assert result["input_tokens"] is None
    assert "secret-test" not in json.dumps(result)
