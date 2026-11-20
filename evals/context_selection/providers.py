"""Provider contracts for the experiment, never imported by the production API."""

import asyncio
from dataclasses import dataclass

import httpx

from .core import Passage, cosine, number


class ProviderError(Exception):
    """Only this stable code is reported; provider bodies may contain secrets/text."""

    def __init__(self, code: str):
        self.code = code
        super().__init__(code)


@dataclass(frozen=True)
class Scored:
    values: tuple[float, ...]
    model: str
    requests: int
    input_tokens: int | None


class HTTPScorer:
    def __init__(
        self,
        client: httpx.AsyncClient,
        endpoint: str,
        model: str,
        key: str,
        *,
        concurrency: int = 4,
        timeout: float = 30.0,
    ):
        if concurrency < 1 or number(timeout) <= 0 or not model or not key:
            raise ValueError("provider configuration is incomplete or invalid")
        url = httpx.URL(endpoint)
        if (
            url.scheme not in ("http", "https")
            or not url.host
            or url.username
            or url.password
        ):
            raise ValueError("endpoint must be an HTTP(S) URL without credentials")
        self.client = client
        self.endpoint = endpoint
        self.model = model
        self.key = key
        self.concurrency = concurrency
        self.slots = asyncio.Semaphore(concurrency)
        self.timeout = timeout
        self.requests = 0
        self.reported_input_tokens = 0

    async def _post(self, payload: dict, token_field: str) -> tuple[dict, int | None]:
        async with self.slots:
            self.requests += 1
            try:
                response = await self.client.post(
                    self.endpoint,
                    headers={"Authorization": f"Bearer {self.key}"},
                    json=payload,
                    timeout=self.timeout,
                    follow_redirects=False,
                )
            except httpx.HTTPError:
                raise ProviderError("transport_error") from None
        if response.status_code != 200:
            raise ProviderError(f"http_{response.status_code}")
        try:
            body = response.json()
            if not isinstance(body, dict):
                raise ValueError
            usage = body.get("usage", {})
            if not isinstance(usage, dict):
                raise ValueError
            count = usage.get(token_field)
            if count is not None and (
                isinstance(count, bool) or not isinstance(count, int) or count < 0
            ):
                raise ValueError
            if count is not None:
                self.reported_input_tokens += count
            return body, count
        except (ValueError, TypeError):
            raise ProviderError("invalid_response") from None

    async def embeddings(self, query: str, passages: tuple[Passage, ...]) -> Scored:
        """One embedding batch, with explicit response-index validation."""
        if not passages:
            return Scored((), self.model, 0, 0)
        async with asyncio.timeout(self.timeout):
            body, usage = await self._post(
                {"model": self.model, "input": [query, *[p.text for p in passages]]},
                "prompt_tokens",
            )
        try:
            count = len(passages) + 1
            rows = body["data"]
            if not isinstance(rows, list) or len(rows) != count:
                raise ValueError
            vectors = [None] * count
            for row in rows:
                index = row["index"]
                if (
                    isinstance(index, bool)
                    or not isinstance(index, int)
                    or not 0 <= index < count
                    or vectors[index] is not None
                    or not isinstance(row["embedding"], list)
                ):
                    raise ValueError
                vectors[index] = row["embedding"]
            values = tuple(cosine(vectors[0], vector) for vector in vectors[1:])
            # An explicit model mismatch invalidates a supposedly fixed-space replay.
            actual_model = body.get("model", self.model)
            if actual_model != self.model:
                raise ValueError
            return Scored(values, self.model, 1, usage)
        except (KeyError, TypeError, ValueError, OverflowError):
            raise ProviderError("invalid_embeddings") from None

    async def jev(self, query: str, passages: tuple[Passage, ...]) -> Scored:
        """Use fixed worker tasks, not one queued task per passage.

        Any failure cancels and drains siblings. Cancellation is not converted to
        fallback. Successful threshold rejection remains an honestly empty result.
        The shared semaphore also bounds overlapping score() calls on this adapter.
        """
        values = [0.0] * len(passages)
        usages = [None] * len(passages)
        models = [self.model] * len(passages)
        remaining = iter(enumerate(passages))

        async def worker():
            for index, passage in remaining:
                body, usages[index] = await self._post(
                    {
                        "model": self.model,
                        "state": passage.text,
                        "questions": {
                            "usefulness": {
                                "type": "score",
                                "instructions": (
                                    "Rate evidence useful for answering this question, "
                                    "including necessary conditions, exceptions, or "
                                    f"contradicting facts: {query}"
                                ),
                                "criteria": [
                                    "No information bearing on the question",
                                    "Topic overlap without answer evidence",
                                    "Evidence for part of the question or a necessary caveat",
                                    "Specific evidence for the main requested facts",
                                ],
                            }
                        },
                    },
                    "input_tokens",
                )
                try:
                    score = number(body["answers"]["usefulness"]["score"])
                    models[index] = body.get("model", self.model)
                    if not 0 <= score <= 3 or not isinstance(models[index], str):
                        raise ValueError
                    values[index] = score
                except (KeyError, TypeError, ValueError):
                    raise ProviderError("invalid_usefulness") from None

        try:
            async with asyncio.timeout(self.timeout):
                async with asyncio.TaskGroup() as group:
                    for _ in range(min(self.concurrency, len(passages))):
                        group.create_task(worker())
        except ExceptionGroup as errors:
            # Suppress the exception group: nested messages are provider-controlled.
            for error in errors.exceptions:
                if isinstance(error, ProviderError):
                    raise error from None
            raise ProviderError("scoring_failed") from None
        if len(set(models)) > 1:
            raise ProviderError("inconsistent_model")
        total = sum(usages) if all(value is not None for value in usages) else None
        model = models[0] if models else self.model
        return Scored(tuple(values), model, len(passages), total)
