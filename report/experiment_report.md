# BeforeYouBuy AI — implementation and experiment report

## 1. Overview

BeforeYouBuy AI is a neutral purchase-reflection application. It combines user-entered item details with visible arithmetic, a small transparent sentiment lexicon, local educational-note retrieval, an explainable reflection heuristic, optional local Ollama conversation, and PDF export. It does not make a buy-or-don't-buy decision for a person.

The workspace is a clean rebuild. It contained no preserved historical model logs, study guide, or result CSVs. Historical owner-reported metrics in the original specification are therefore unverified and are not presented as measurements from this application.

## 2. Architecture and data flow

1. React/TypeScript form sends only entered fields to the FastAPI backend.
2. Pydantic validates all supplied numeric values and enumerated answers.
3. `backend.assessment.evaluate` calculates the assessment, explicit sentiment cues, risk heuristic, optional affordability comparisons, and cost per use.
4. `backend.knowledge` reads Markdown reference notes and ranks them by normalized title/point matches.
5. `/api/chat` retains up to 12 turns and explicitly extracted facts per session in server memory (30-minute inactivity expiry, maximum 1,000 live sessions). It uses already-calculated assessment/retrieval outputs for optional Ollama prompting, or a deterministic fallback.
6. `/api/report/pdf` recomputes from the submitted assessment fields and returns a ReportLab PDF. Requests are not written to disk by the service.

Main files: `frontend/src/App.tsx`, `backend/main.py`, `backend/assessment.py`, `backend/knowledge.py`, and `data/common/`.

## 3. Tool descriptions

- **Assessment:** deterministic scoring, bounds-checked arithmetic, and missing-input indicators.
- **Sentiment/tone:** keyword classification with matched-word evidence; not a trained language model or psychological assessment.
- **Cost per use:** entered uses per week × 52 weeks × entered lifespan; recurring cost is separate.
- **Affordability comparison:** compares the entered budget against the price and any entered 12-month recurring amount. A separately entered monthly surplus is displayed as the number of stated-surplus months; it is not verified or a complete affordability judgment.
- **Regret-risk reflection:** supplied answers and explicit text cues add source-configured design weights to a 0–100 score. The score is not validated science, a probability, or a prediction.
- **Knowledge retrieval:** normalized keyword matching, title weighting, deterministic tie order, and source title/URL output.
- **LLM:** local Ollama model discovery/selection, bounded timeout/concurrency, response safety checks, and deterministic fallback. No cloud API is configured.
- **PDF:** ReportLab endpoint returns a valid PDF attachment.

## 4. Knowledge sources

Seven Markdown notes are present in `data/common/`. Their source links were checked on 2026-10-25 and returned HTTP 200:

- MyMoney.gov — <https://www.mymoney.gov/>
- FDIC Consumer Resource Center — <https://www.fdic.gov/consumer-resource-center>
- CFPB Credit Cards — <https://www.consumerfinance.gov/consumer-tools/credit-cards/>
- Federal Reserve Consumer and Community Affairs — <https://www.federalreserve.gov/consumerscommunities.htm>

One earlier CFPB budgeting URL was found to return 404 and was replaced. Note content is educational; it does not represent product research or individualized financial advice. The 30-day rule is explicitly optional and not guaranteed.

## 5. Sentiment labeling and evaluation

`sentiment_labels.csv` contains 30 manually assigned examples: 10 positive, 10 negative, and 10 neutral. `scripts/evaluate_sentiment.py` compares the deterministic keyword classifier with those labels.

Measured on this sample:

- Accuracy: **23/30 = 0.767**
- Positive precision/recall/F1: **1.000 / 0.700 / 0.824**
- Negative precision/recall/F1: **1.000 / 0.600 / 0.750**
- Neutral precision/recall/F1: **0.588 / 1.000 / 0.741**
- Confusion matrix (actual rows, predicted columns):

  | Actual \ Predicted | Positive | Negative | Neutral |
  |---|---:|---:|---:|
  | Positive | 7 | 0 | 3 |
  | Negative | 0 | 6 | 4 |
  | Neutral | 0 | 0 | 10 |

This small manually written sample is not representative and does not establish generalization. Lexicon signals can miss nuance and context.

## 6. HTTP and UI evaluation

The following five exact messages were exercised sequentially against the HTTP chat API in deterministic fallback mode:

1. `I want a Rs 8000 pair of headphones because they are trending`
2. `My current ones work fine, I just like the design`
3. `I would use them about 5 times a week for about 2 years`
4. `I have around Rs 6000 spare each month and an emotional week behind me`
5. `I think I would still want them in a month`

Observed:

- The explicit Rs 8,000 item was retained as the price; no income, savings, debt, or other financial values were inferred.
- The third message yielded 520 estimated uses (`5 × 52 × 2`) and a cost per use of 15.3846 INR (about INR 15.38).
- The fourth message was recorded as an explicitly stated INR 6,000 monthly surplus. The tool showed a comparison of about 1.33 months of that stated value, without treating it as independently verified.
- Chat replies remained neutral, respectful, contained no purchase directives or raw tool syntax, and asked zero questions.
- Two browser follow-ups recalled explicitly stated product, usage, and surplus details from the same session; a separate session receives no prior session facts. The assistant stated it could not choose for the user.
- Assessment/chat responses include actual tool names and backend elapsed time.
- The browser loaded the updated TypeScript app (the main entry now explicitly imports `App.tsx`) and showed the correct cost-per-use calculation.
- The animated inference state and disabled chat controls remained visible during a live 18,780.9 ms local-model request; its reasoning-style output was rejected and the UI clearly reported deterministic fallback.
- An aborted browser chat request showed the Retry action; retry succeeded. New conversation cleared chat turns. The interface was checked at 390 × 844 and 1440 × 900; no mobile horizontal overflow was observed.

Five deterministic fallback responses were reviewed against the response criteria and scored in `response_evaluation.csv`. Live Ollama outputs were tested but rejected by application safety checks, so they were not included in the response score sheet.

## 7. PDF verification

The live `POST /api/report/pdf` endpoint returned HTTP 200, `application/pdf`, and an attachment. A direct download was inspected as a readable, non-empty, two-page PDF 1.4 document. `pdfinfo` and `pdftotext` confirmed the expected item and cost-per-use content. Automated integration tests also check the `%PDF-` signature and minimum document size. A browser click triggered the endpoint response, but the browser harness did not expose a native download event or response body.

## 8. Automated testing

Final command: `pytest -q` — **62 passed**.

Frontend command: `cd frontend && npm run build` — TypeScript check and Vite production bundle succeeded. `cd frontend && npm audit --audit-level=moderate` reported **0 vulnerabilities**. An initial browser review found that Vite resolved the old duplicate `App.js`; making the main entry import `App.tsx` explicitly fixed the runtime mismatch.

`python scripts/check_secrets.py` passed. The sentiment evaluator reported 0.767 accuracy on the 30 manually labeled examples. The response summary reported 5/5 manually scored fallback replies passing. Full command outputs and the HTTP/PDF checks are recorded in `TEST_RESULTS.md`.

The Ollama timeout, unavailable provider, malformed response, separate `thinking` field, invented money, chain-of-thought style output, and unsafe-response paths are covered by tests. A live probe found `qwen3:4b` on Ollama 0.40.3 on the inspected Apple Silicon Mac (8 GiB RAM); browser inference took 18,780.9 ms but produced reasoning-style text that was rejected by application checks. The user received deterministic fallback instead. No model weights were downloaded and no fine-tuning was performed.

## 9. Human response evaluation

The five actual deterministic fallback replies from the sequential HTTP demo were read and reviewed. Each satisfied the stated checks for respectfulness/relevance, no more than two questions, neutral decision support, no invented personal facts, no unsupported factual claim, no raw tool-call syntax, and no privacy violation. The score sheet is `response_evaluation.csv`; the summary script reports its pass rate.

## 10. Privacy and ethical safeguards

- No bank or external financial account connection.
- No inference of unprovided income, savings, debt, or disposable funds.
- No direct purchase instruction; factors and assumptions are visible.
- User text is treated as data; retrieved notes are not executable instructions.
- Optional local model has a short timeout and application-level output checks.
- In-memory chat is bounded by turn count, idle expiry, and session count; restart clears it.

## 11. Limitations and future work

- The sentiment lexicon and regret-risk scoring are limited heuristics.
- Automated safety checks cannot guarantee every model response is safe; a human review remains valuable.
- The installed Ollama model did not produce an accepted user-facing reply in the live probe; deterministic fallback is verified and remains available.
- The chat animated in-progress state is implemented and was observed in the browser during local inference. Responses are not streamed.
- The input parser is heuristic and cannot reliably extract every natural-language phrasing; ambiguous or unrecognized values remain unknown rather than being invented.
- No fine-tuning was performed; prompting/retrieval is not described as fine-tuning.
- The educational references may change; users should consult current source pages.
- No original experiment results, historical conversations, or study guide were available to reproduce.
- No screen recording was produced.

## 12. Reproduction

Follow setup commands in `README.md`. Run `pytest -q`, `cd frontend && npm run build`, and `python scripts/evaluate_sentiment.py`. For optional local LLM usage, configure Ollama using `.env.example`. The clean ZIP is produced separately under `deliverables/`.
