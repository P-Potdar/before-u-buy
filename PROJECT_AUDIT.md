# PROJECT_AUDIT

## Scope and authoritative version

The inspected workspace is a clean rebuild, not the historical experiment repository. The authoritative app is the React + TypeScript frontend backed by FastAPI. The original assessment implementation was backed up before substantial work; the current authoritative files have since been updated with the requested features.

- Frontend entry: `frontend/src/main.tsx` explicitly imports `frontend/src/App.tsx`.
- Backend entry: `backend/main.py`.
- Assessment rules: `backend/assessment.py`.
- Knowledge retrieval: `backend/knowledge.py` and `data/common/*.md`.

## Current functionality

- Structured purchase assessment with validation, missing-input indicators, optional affordability comparisons, and cost-per-use arithmetic.
- Explainable 0–100 regret-risk reflection heuristic with displayed factors and limitations.
- Lexicon-based sentiment/emotional cue signals with evidence and confidence caveats.
- Weighted local Markdown knowledge retrieval with source titles, URLs, stable tie ordering, and empty-base handling.
- `/api/assess`, `/api/chat`, `/api/report/pdf`, and `/health` HTTP endpoints.
- Optional local Ollama chat; deterministic fallback is the default and remains available if Ollama is unreachable.
- React form, assessment display, source links, tool/timing labels, follow-up chat, and PDF download.
- Automated tests, 30 labeled sentiment examples, evaluation scripts, response scoring template, and technical report.

## Repository observations

- Runtime: Python 3.11, FastAPI, Pydantic, ReportLab; Node 20, React, TypeScript, Vite.
- Inspected host: macOS 26.5.2, Apple Silicon `arm64`, 8 GiB RAM, Python 3.11.6, Node 20.11.1, npm 10.8.2.
- Local inference discovery: Ollama is installed and exposes `qwen3:4b` (4.0B, Q4_K_M, about 2.5 GB). No other runtime/model was established as available; no weights were downloaded.
- `frontend/src/App.js` and `frontend/src/main.js` are older duplicate JavaScript files. They are intentionally preserved but are not runtime entry points; the TypeScript entry imports `App.tsx` explicitly to avoid extension-resolution ambiguity.
- The clean rebuild did not contain `study_guide.md`, historical model result CSVs, the former experiment scripts, or prior run logs. No such historical data or metrics were created or relabeled as original.
- The historical owner-reported numbers in the specification remain unverified and are not represented as measurements from this workspace.

## Knowledge references checked

The note source pages were checked over HTTPS on 2026-10-25 and returned HTTP 200:

- MyMoney.gov: `https://www.mymoney.gov/`
- FDIC Consumer Resource Center: `https://www.fdic.gov/consumer-resource-center`
- CFPB Credit Cards: `https://www.consumerfinance.gov/consumer-tools/credit-cards/`
- Federal Reserve Consumer and Community Affairs: `https://www.federalreserve.gov/consumerscommunities.htm`

One earlier CFPB budgeting URL returned 404 and was replaced. Notes are educational context; the 30-day technique is described as optional and not guaranteed.

## Confirmed validation (2026-10-25)

- `pytest -q`: **62 passed**.
- `cd frontend && npm run build`: TypeScript check and Vite production build succeeded.
- `cd frontend && npm audit --audit-level=moderate`: **0 vulnerabilities**.
- `python scripts/evaluate_sentiment.py`: 30 examples; accuracy 0.767, with per-class metrics and confusion matrix in `TEST_RESULTS.md`.
- `python scripts/summarize_evaluations.py`: 5 manually reviewed fallback replies passed; this is not a live LLM evaluation.
- `python scripts/check_secrets.py`: passed.
- Live `/health`: HTTP 200, status `ok`; frontend dev server returned HTTP 200.
- The exact five-message demo and two follow-ups were exercised through the browser chat without completing an assessment first. Follow-up recall, neutral decision support, and session continuity were observed. Five uses/week for two years yielded 520 uses and INR 15.3846/use; the INR 6,000 monthly surplus remained user-stated and was not treated as verified.
- Browser checks covered blank/oversized chat input, an intentionally failed request and retry, new-conversation clearing, and 390 × 844 / 1440 × 900 layouts. The mobile document width matched the viewport.
- Browser model inference showed the thinking indicator and disabled inputs during an 18,780.9 ms request. Local `qwen3:4b` output failed safety checks and was replaced with an explicitly labeled deterministic fallback; no clean model-generated response is claimed.
- Live `POST /api/report/pdf`: HTTP 200, `application/pdf`, attachment; a direct HTTP download was verified by `pdfinfo` and `pdftotext` as a readable two-page PDF 1.4 file with expected report content. The browser Download PDF action triggered the attachment response, though the harness did not expose its native download event/body.

See `TEST_RESULTS.md` for exact commands, output, and limits.

## Risks and limitations

1. Sentiment is a simple lexicon signal, not a trained model or psychological assessment; its small sample metrics do not establish generalization.
2. Assessment points and regret-risk weights are transparent design heuristics, not validated science or a regret probability.
3. Ollama and `qwen3:4b` are installed, but tested live outputs were rejected as reasoning-style rather than accepted as safe, useful final replies. Deterministic fallback is the currently verified user-facing path; no fine-tuning or model download was performed.
4. The chat session is kept in process memory, limited to the latest 12 turns, and is not designed as a multi-user production service.
5. The UI download action and attachment response work; the available browser harness does not expose a native download event or PDF response body.
6. No browser screen recording or historical experiment reproduction was performed.

## Launch path

Follow the tested setup commands in `README.md`. In brief: install backend requirements, run `uvicorn backend.main:app --reload --port 8000`, then from `frontend/` run `npm install && npm run dev`.

## Follow-up implementation audit (2026-10-26)

- The current source and fresh browser load show the Overview dashboard already exists at the initial route. No historical removal commit or prior dashboard implementation could be established because Git metadata traversal failed outside this project. Do not interpret the current dashboard as a verified restoration from history.
- Fixed deterministic purchase-intent fallback: when a product is recognized and budget/use information is missing, ask a concise, product-specific follow-up rather than echoing the message. Actual browser/API verification for the headphone example is in `TALK_IT_THROUGH_TEST_LOG.md`.
- Corrected journal navigation and added visible persisted decision history, status updates, per-entry deletion, and loading/error/retry states for workspace data.
- Fixed CORS support for existing `PATCH` and `DELETE` controls and surfaced errors from watchlist mutation/export/delete operations.
- Analytics now groups amounts and trends by currency. Cross-currency aggregate totals are nullable/empty, and potential savings are not estimated without a counterfactual.
- Pytest uses a disposable SQLite database provisioned by root `conftest.py`; tests do not use the user's application database.
- Latest verification: `pytest -q` **105 passed in 1.05s**; frontend production build passed; `npm audit --audit-level=moderate` reported 0 vulnerabilities; `python scripts/check_secrets.py` passed.
- Live browser verified initial dashboard empty state, reload, all navigation targets, and responsive widths at 375/768/1440 px with no page or console errors. Populated-dashboard browser verification was not performed to avoid writing test data to the user's persistent database; populated journal/analytics are covered by isolated API tests.
- `GET /api/providers/status` returned `configured: false`, `mode: manual_only`. Mocked provider tests passed; no authenticated SerpApi call was performed.
- Product refresh now refuses to overwrite a tracked variant with a different returned variant. Offer normalization rejects non-finite prices and URLs containing credentials or unsupported schemes; observation inputs are validated, and stale offers receive a ranking penalty.
- Local Qwen model output remains rejected on the observed real inference; the purchase prompt is identified as deterministic fallback, not model-generated. No fine-tuning or weight download occurred.

## Progressive chat analysis update (2026-10-26)

- Chat remains in the existing FastAPI/React architecture; session facts are still temporary in-memory state and are isolated by session ID.
- The deterministic extractor now tracks product/price/currency, motivation, use case, ownership, current-item condition, explicit budget/surplus, use frequency, and lifespan. Clear correction language replaces an earlier value; unresolved conflicts are surfaced and prevent a misleading complete assessment.
- `/api/chat` returns `decision_state` after every message. It includes provisional status, purchase-pressure and regret-risk factors, missing inputs, evidence for/against, an explicitly limited decision signal, and any supported cost-per-use estimate. The UI renders these values as conversation context changes.
- Calculations use only extracted or form-supplied values; bounds prevent huge/non-finite parsed amounts and nonsensical usage rates from being treated as facts. A supported correction updates the next calculation; conflicting values suppress the assessment.
- Local chat now requests a validated JSON schema and uses only its user-facing `assistant_message`; model-interpreted facts are never used to mutate deterministic session facts. A separate Ollama `thinking` field is not used. Malformed/unsafe output retains the deterministic fallback.
- Actual browser exercised no-assessment start, the headphones/trending motivation, working current pair/use case, and five-times-weekly/two-year estimate. It observed fallback labeling, scores, the one-question behavior, 520 uses, INR 15.38/use, and responsive layouts at 390 and 1440 px. No page errors occurred.
- The live Ollama structured-output path and successful clean model response remain unverified. Prior real Qwen output was rejected; fine-tuning and model downloads were not performed.
- Latest automated outcomes are recorded in `TEST_RESULTS.md`; do not treat prior 105/107-test entries as the result of the newest run.
