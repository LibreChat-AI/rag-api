# Context selection experiment

This is an independently authored, opt-in replay harness, not a production
search change. It tests whether selecting **answer-bearing evidence**, rather
than always filling a top-k window, can reduce context without losing facts,
exceptions, or opposing evidence.

The implementation and synthetic fixtures are independently authored. BM25
uses its standard formula with a simple NFKC/case-folded word tokenizer
that retains negation and identifiers. Provider adapters implement only
the HTTP contracts needed for the experiment.

## Boundary and ownership

```text
fixed, already-authorized passage pool + query
  -> score once (BM25 / embeddings / Jev)
  -> selection ablations using those same scores
  -> original passage IDs and sources + evidence-retention metrics
```

The harness imports neither `app.config` nor application storage. Production
routes, schemas, chunking, caches, authorization and ingestion are untouched.
There is no server endpoint, database probe, scrape, writer LLM or automatic OCR.
The operator owns corpus authorization and permission to send its content to a
provider. Never use private production documents without that authorization.

A candidate is one complete existing chunk. This experiment does not re-chunk
or summarize it. Keep candidate generation constant across strategies; later
chunk-size experiments need separately fingerprinted corpora. A selector cannot
recover evidence omitted by the retriever.

Selection owns stable tie ordering, thresholds and output budgets. The HTTP
adapter owns response validation, shared concurrency and cancellation. Evaluation
labels are used only after selection and are never included in provider payloads.

## Strategies

| Strategy | Behavior |
| --- | --- |
| `all` | Original pool order, subject to the character budget, not top-k |
| `input-topk` | Original retrieval order, top-k and character budget |
| `bm25-topk` | Lexical scoring without a relevance threshold |
| `bm25-filter` | Lexical scoring, relative cutoff against the best score |
| `embeddings-topk` | Cosine ranking without a threshold |
| `embeddings-filter` | Cosine ranking with an absolute threshold |
| `jev-topk` | Usefulness ranking without a threshold |
| `jev-filter` | Usefulness ranking with an absolute threshold |

`max_results` is a ceiling, not a requirement to fill slots. Relative BM25
selection returns empty when no term matches. Successful Jev rejection returns
empty, never the first passages as a disguised fallback. A provider failure is
an **error**, not a successful lexical run attributed to the provider. Provider
errors make the command exit 1; skipped runs are explicitly counted.

Thresholds are experimental settings, not demonstrated calibration. Jev scores
are expected rubric levels in `[0, 3]`, not answer-correctness probabilities.
BM25 and cosine scores are not on that scale. There is no evidence that one
threshold fits every query type, language or model.

`max_chars` counts Python Unicode characters in complete selected passage
bodies. It is **not** a model token budget, UTF-8 byte budget or final prompt
budget. Citation wrappers also cost tokens. Oversized passages are skipped,
not truncated into apparently complete evidence. Optional exact-text
deduplication applies to scored strategies; `all` and `input-topk` remain
unmodified controls. It does not merge overlapping or paraphrased passages.

## Run offline

Python 3.12 is the repository's CI version. The local CLI uses only the standard
library. HTTP checks need `httpx` from `test_requirements.txt`.

```sh
mkdir -p .venv
python3 -m evals.context_selection.replay \
  --max-results 2 --output .venv/context-offline.json
```

The default includes BM25 and explicitly **skips** embeddings and Jev. Even if
keys exist, no client is constructed without `--allow-network`. There is no
`.env` loading. Missing live configuration is also recorded as skipped, never
silently replaced with simulated model scores.

The committed 10 cases and 82 passages are synthetic boundary fixtures:
refund exceptions, exact issue identifiers, negation, multi-facet answers,
conflicting dated sources, Unicode, duplicate passages, evidence after the
first 50 candidates, irrelevant pools, and empty pools. They establish behavior
and expose lexical blind spots. They are **not** a representative benchmark or
an independent estimate of answer quality.

## Live replay, once credentials are provided

Set these through the environment or secret tooling, not tracked files or PR
comments:

- `CONTEXT_EMBEDDING_URL`: full OpenAI-compatible embeddings endpoint URL.
- `CONTEXT_EMBEDDING_MODEL`: explicit model name.
- `CONTEXT_EMBEDDING_KEY`: provider key.
- `TYPESAFE_API_KEY`: Jev provider key.
- `CONTEXT_JEV_MODEL`: defaults to `jev-latest`; pin a model when possible.
- `CONTEXT_JEV_URL`: optional, defaults to the System One endpoint.

The URLs are operator-controlled endpoints, not untrusted user URL inputs.
TLS verification remains enabled and redirects are not followed. Secrets and
provider response bodies are not written to reports or printed on failures.
Queries and passages go to the configured provider only on explicitly enabled
runs. Score outputs include models, indices, call counts, reported input tokens
(or null if absent), and measured scoring time. Failed batches retain attempted
request counts and the subtotal of input tokens reported before failure, but
mark total usage unknown. No pricing is invented.

```sh
python3 -m evals.context_selection.replay --allow-network \
  --max-results 2 --output .venv/context-live.json

# Sweep selection settings without paying for the same scores again.
python3 -m evals.context_selection.replay \
  --reuse-scores .venv/context-live.json --jev-min 2.0 \
  --max-results 4 --output .venv/context-sweep.json
```

Corpus bytes and schema version must match for reuse. Scoring happens once per
case/provider; top-k and filtering reuse that result. Costs/calls must be read
from `score_runs`, not summed across selection rows. A saved-score replay
records zero new provider calls. Do not compare its inherited scoring latency
to a new network run.

Jev uses a fixed number of worker tasks and a semaphore shared across calls on
that adapter. A batch deadline includes queue wait. Failure or cancellation
cancels and drains sibling tasks. We intentionally make no automatic retries;
rerun a failed experiment explicitly. The per-case candidate/character bounds
are enforced before any provider call. The default embedding batch contains
query plus candidate text; provider-specific batch limits may be lower and
should be reflected in the flags. Do not silently drop excess candidates.

## Corpus and metrics

`--corpus` accepts a JSON array of cases. Passage order is the original
retrieval order. IDs must be unique within a case; case IDs are globally unique.
Each required fact has an evidence label; any passage may support several.
Every required label must be present somewhere in the candidate pool, so
selection recall is not confused with retrieval recall. A `required: []` case
means no candidate contains useful answer evidence, not that the real-world
question has no answer.

```json
[
  {
    "id": "retention-example",
    "query": "When can a record be deleted?",
    "required": ["duration", "legal-hold"],
    "passages": [
      {"id": "p1", "source": "policy:page1", "text": "Keep records for seven years.", "supports": ["duration"]},
      {"id": "p2", "source": "policy:page2", "text": "A legal hold prevents deletion.", "supports": ["legal-hold"]}
    ]
  }
]
```

Reports retain IDs and source handles, not query or passage bodies. They contain:

- evidence precision: fraction of selected passages carrying annotated evidence;
- evidence recall: fraction of required labels retained;
- missing-evidence labels, including exceptions and contradicting sources;
- correct abstentions for pools with no annotated evidence;
- selected characters and passage counts;
- separate successful, skipped and error counts.

Precision is null for empty selections, recall is null for no-evidence cases.
Macro summaries report their denominators and exclude null values, so inspect
missing-evidence and abstention counts alongside them. High precision alone can mean useful facts were lost.
This is label-based selection evaluation, not generated-answer faithfulness.

## Verify this slice

```sh
# The harness needs no application fixtures, database, or embedding initialization.
python -m pytest --noconftest tests/test_context_selection.py
python -m black --check evals/context_selection tests/test_context_selection.py
python -m compileall -q evals/context_selection tests/test_context_selection.py
```

The test file is also discovered by the repository's normal CI unit-test run.
MockTransport emulates external APIs; failure/cancellation tests exercise the
actual adapter tasks and cleanup, not a fake implementation of selection.

## Acceptance before production integration

1. Replay representative, authorized LibreChat web/file candidate pools with
   human-reviewed evidence labels. Keep a held-out set for threshold selection.
2. Compare identical candidates, budgets, and writer models if answer generation
   is added. Track identifier queries, broad questions, caveats, conflicts,
   multilingual content and answer evidence late in the pool separately.
3. Require retained evidence and citation provenance, not token reduction alone.
   Record provider latency, inference usage and downstream prompt-token counts
   with the actual tokenizer before claiming speed or cost wins.
4. Exercise the real adapters with provider credentials. MockTransport tests
   prove our expected HTTP contract, not current provider behavior or calibration.
5. Only then add an opt-in production selector. Existing search is the rollback
   path. Auth/tenant/entity scope must be checked before inference egress, and
   filtering must never substitute for whole-file content inspection.
6. Update the consuming agents contract to distinguish an honest empty evidence
   set from service failure. Its current RAG reranker treats empty results as a
   bad response and falls back; that must not be reused unchanged for filtering.

No model quality, calibration, answer-quality, real-provider latency, downstream
cost reduction, or production no-regression claim follows from the offline run.
