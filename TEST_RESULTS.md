# Test Results

Validation date: 2026-10-26. Results below were run against this clean rebuild; they do not represent historical experiments or owner-reported results. Earlier figures are retained as historical validation entries; the latest run is in the addendum below.

## Automated checks

| Command | Result |
|---|---|
| `pytest -q` (project root) | **105 passed** |
| `cd frontend && npm run build` | Passed: `tsc -b` and Vite production build (Vite 6.4.5) |
| `cd frontend && npm audit --audit-level=moderate` | **0 vulnerabilities** |
| `python scripts/evaluate_sentiment.py` | 30 labeled examples; accuracy 0.767 |
| `python scripts/summarize_evaluations.py` | 5 reviewed; 5 pass; 0 fail; 0 pending |
| `python scripts/check_secrets.py` | Passed |

Sentiment per-class precision / recall / F1:

| Class | Precision | Recall | F1 |
|---|---:|---:|---:|
| Positive | 1.000 | 0.700 | 0.824 |
| Negative | 1.000 | 0.600 | 0.750 |
| Neutral | 0.588 | 1.000 | 0.741 |

Confusion matrix (rows = actual, columns = predicted):

| Actual \ Predicted | Positive | Negative | Neutral |
|---|---:|---:|---:|
| Positive | 7 | 0 | 3 |
| Negative | 0 | 6 | 4 |
| Neutral | 0 | 0 | 10 |

This is a small manually labeled sample evaluated against a deterministic keyword heuristic. These results do not establish generalization or measure an LLM.

## HTTP, UI, and PDF checks

- Live `GET http://127.0.0.1:8000/health`: HTTP 200, `status: ok`, deterministic mode.
- Live frontend dev server at `http://127.0.0.1:5173/`: HTTP 200, `text/html`.
- The five exact demonstration messages and two follow-ups were sent through the browser chat without first completing an assessment. All seven requests used one session; the follow-up recalled supplied facts, and the assistant explicitly left the decision to the user.
- Message 3's explicit 5 uses/week × 52 weeks/year × 2 years gave **520 estimated uses**; ₹8,000 / 520 gave **₹15.3846 per use** (approximately ₹15.38). The 52-week assumption is stated in the UI and report.
- Message 4's stated ₹6,000 monthly surplus was stored only as a user-supplied value; no independent verification or other financial facts were inferred.
- Browser checks confirmed the empty Send button is disabled, the chat input enforces its 2,000-character limit, Shift+Enter inserts a newline, two simultaneous submissions produce only one chat POST, retries work after a deliberately aborted request, and New conversation clears chat turns. Separate-session isolation and malformed/oversized API requests are covered by backend tests.
- The browser was checked at 390 × 844 and 1440 × 900; the mobile document width matched the viewport (no horizontal overflow).
- The five actual deterministic fallback replies were manually checked and scored 5/5 in `response_evaluation.csv`. They did not direct a purchase, exceed two questions, invent finances, or expose internal tool syntax.
- A separate ten-scenario chat regression run was completed before and after targeted fixes. Exact inputs, responses, scores, and pre/post comparison are recorded in [TALK_IT_THROUGH_TEST_LOG.md](./TALK_IT_THROUGH_TEST_LOG.md).
- An earlier follow-up browser check caught and fixed an unstated-currency display issue and irrelevant references on conversational-recall turns. A chat-only price of `8000` now remains currency-unspecified, does not trigger a currency-denominated assessment, and is recalled as `8000 (currency not specified)`; the recall turn does not run knowledge retrieval. Later test counts and checks are in the dated follow-up section.
- Browser assessment displayed the calculations and sources. Clicking Download PDF review triggered HTTP 200 with `Content-Disposition: attachment` and `Content-Type: application/pdf`. The browser harness did not expose a native download event or response body; a direct HTTP download was therefore used for document inspection.
- Live `POST /api/report/pdf` returned a valid, non-empty PDF 1.4 document. `pdfinfo` confirmed a readable two-page Letter document; `pdftotext` confirmed the supplied item and INR 15.3846 cost-per-use. The integration test checks the PDF signature and non-empty size.
- `GET /api/models` discovered installed `qwen3:4b` on Ollama 0.40.3. A live browser inference took **18,780.9 ms**; the model output failed application safety checks. The API labeled the result `deterministic_fallback`, and the UI showed the fallback reason, response, and latency. The animated thinking state was visible and the input/send controls were disabled during inference. The user-facing fallback—not model-generated text—was verified.
- A later browser inference during the ten-scenario regression work took **28,936.25 ms**. Its reasoning-style response was also rejected by the safety checks, and the UI displayed the deterministic fallback without exposing the reasoning trace. This is not a verified successful model-powered response.

## LLM tests and limitations

Live Ollama inference probes were performed, but the generated reasoning-style replies were rejected and are not counted as successful user-facing model responses. Tests use deterministic mocks to cover unavailable service, timeout, empty/malformed output, and unsafe response fallback. No weights were downloaded. Default operation requires no model or paid API.

Other limitations: the sentiment lexicon and reflective score are heuristic; no production user research or broad financial validation was conducted. The browser harness could not confirm a browser-native save dialog/download event or expose the PDF response body; the endpoint was downloaded and inspected directly instead.

## Follow-up verification — 2026-10-26

- **PASS** `pytest -q`: **105 passed in 1.05s**. A root `conftest.py` provisions and cleans an isolated temporary SQLite database for the test session; no application database was deleted or used by the tests.
- **PASS** `cd frontend && npm run build`: `tsc -b` and Vite 6.4.5 production build passed (latest run 0.663s).
- **PASS** `cd frontend && npm audit --audit-level=moderate`: 0 vulnerabilities.
- **PASS** `python scripts/check_secrets.py`: secret scan passed.
- **PASS** Actual browser/API request for `I want to buy headphones`: HTTP 200, `deterministic_fallback`, no model, 7.49 ms. Reply: “I can help you compare headphones around what matters to you. What budget range are you considering, and will you mainly use them for music, calls, commuting, gaming, or work?” Recognized only the product name; did not claim the answer was generated by Ollama.
- **PASS** Reproduction baseline: with `qwen3:4b` selected, browser/API request took 35,828.83 ms; response was labeled `deterministic_fallback` after the local output failed safety checks. Actual reply was: “You have described some interest in this item. You mentioned ‘I want to buy headphones’. I can help examine the reasons, timing, costs, or alternatives without deciding for you. Related notes: Need vs. want.” This was fallback text, not model-authored text.
- **PASS** Browser initial route after workspace loading shows the Overview dashboard and truthful empty state. Refresh preserves it. Navigation targets were inspected for Overview, assessment, chat, watchlist, and journal; the journal empty state appeared at its section. Viewports 375, 768, and 1440 px had no document horizontal overflow; no browser page or console errors were observed.
- **PASS** Isolated backend tests cover saved journal listing, status update, individual deletion, variant separation, target-price editing, currency-grouped analytics, explicit CORS PATCH support, product-provider mock success/failure, and deletion confirmation.
- **PASS** Product audit regressions reject non-finite prices, malformed/unsafe credential-bearing/javascript URLs, and invalid stored observations; stale offers rank below otherwise matching live offers. A watchlist refresh with only a different variant leaves the saved price and history unchanged.
- **PASS** `GET /api/providers/status` returned HTTP 200 with `configured: false`, `mode: manual_only`, country `IN`. No authenticated/live SerpApi request was made.
- **BLOCKED** A populated-data dashboard was not demonstrated in the browser because the running application uses the user's persistent database and must not be seeded or altered without permission. Populated analytics and journal persistence were exercised in the isolated pytest database.
- **PASS** Browser-only analytics HTTP 503 interception showed visible error and Retry controls; removing the interception and reloading recovered the empty dashboard.
- **NOT RUN** Historical Git comparison. `git status`/`git log` could not traverse unrelated parent-repository metadata (`Desktop/klang` authentication/read errors), so no historical dashboard-removal commit is claimed.

## Conversational analysis upgrade — 2026-10-26

- **PASS** `pytest -q`: **109 passed in 2.21s** (latest run after adding progressive facts, price correction/conflict, numeric bounds, decision-signal, and Ollama schema regressions).
- **PASS** `cd frontend && npm run build`: TypeScript and Vite 6.4.5 production build passed (1.31s).
- **PASS** `python scripts/check_secrets.py`: passed.
- **PASS — live browser, no prior assessment:** started a new conversation with an ₹8,000 headphones/trending message. The HTTP response was 200 and labeled `deterministic_fallback`; the reply asked one product-relevant question, and the interface showed provisional `WAIT` scores (pressure 70, regret-risk 45).
- **PASS — live browser follow-ups:** explicitly stated that the current headphones work and are mainly used for music/calls. The UI updated to `DON'T BUY YET` based on stated pressure plus the working alternative, and the reply remained nonjudgmental and limited to one question. No budget was invented.
- **PASS — arithmetic in browser/API:** five uses/week for two years returned **520 estimated uses** and **15.3846 INR/use** (shown as **INR 15.38**). The answer and UI state both stated the 52-week/year assumption.
- **PASS — responsive/error observations:** at 390 × 844 and 1440 × 900, document width equaled viewport width; the browser captured no page errors. New UI displayed the updated scores, status, facts, unknown inputs, and limitations.
- **PASS — automated edge cases:** out-of-range price/usage text is not accepted as a usable fact; explicit corrections replace prior facts; unresolved conflicting prices suppress assessment until correction; distinct sessions remain isolated.
- **PASS — structured model output:** mocked Ollama responses validate the required JSON schema and expose only `assistant_message`; separate `thinking` is ignored and malformed/incomplete output is rejected. No successful live model response was measured in this run; no fine-tuning or model download occurred.
- **Command note:** `npm run build` at the project root is not configured; the correct command is `cd frontend && npm run build` and passed.

These browser requests used temporary in-memory chat sessions and did not create journal/watchlist records or alter the user's SQLite data. The local model was not selected for this browser run; it exercised deterministic fallback, so the structured Ollama path remains verified only by tests.
