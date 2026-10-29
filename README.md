# BeforeYouBuy AI

A neutral, local-first purchase reflection app. It shows which details were supplied, provides transparent calculations, retrieves relevant educational notes, and supports optional follow-up conversation. The chat can show a provisional BUY / WAIT / DON'T BUY decision signal, but this heuristic is not a directive, financial advice, or a validated prediction.

## Architecture

- **Frontend:** React + TypeScript + Vite (`frontend/src/main.tsx` imports `frontend/src/App.tsx` explicitly)
- **Backend:** FastAPI + Pydantic (`backend/main.py`)
- **Assessment:** transparent deterministic heuristic, calculations, and shared BUY / WAIT / DON'T BUY decision rules (`backend/assessment.py`)
- **Conversation assessment:** rule-based extraction with per-session facts, evidence metadata, deterministic scores, and a provisional BUY / WAIT / DON'T BUY decision signal (`backend/main.py`)
- **Sentiment/context analysis:** separate polarity, multi-label emotions, and purchase signals with exact supporting spans (`backend/sentiment.py`). These are transparent weak labels, not psychological, financial, or purchase-decision predictions.
- **Regression scenarios:** small, synthetic baseline set used only for extraction and decision-invariant tests (`backend/data/purchase_scenarios.jsonl`)
- **Purchase dataset:** reproducible synthetic decision corpus and quality report (`backend/data/purchase_decision_dataset.jsonl`, `backend/data/purchase_decision_dataset_quality.json`)
- **Semantic retrieval:** offline hashed word/bigram baseline with version checks, plus an optional local MiniLM dense index when its installed dependency/model is available (`backend/retrieval.py`, `backend/dense_retrieval.py`)
- **RAG knowledge retrieval:** normalized, weighted local Markdown notes are retrieved at chat time, bounded into the Qwen prompt as untrusted evidence, and returned with source provenance (`backend/knowledge.py`, `data/common/`)
- **PDF:** generated server-side by ReportLab at `POST /api/report/pdf`
- **Optional LLM:** installed local Ollama models are discovered at runtime; deterministic fallback remains available
- **Dashboard and journal:** the initial Overview is backed by local SQLite records; journal entries can be reviewed, updated, exported, or removed

The app does not depend on a paid API. Assessment requests are not stored; chat retains at most the latest 12 turns per session in server memory, expires sessions after 30 minutes of inactivity, and is lost on restart. Saved decisions, watchlist entries, and price observations persist in SQLite. Dashboard amounts are grouped by their recorded currency rather than added across currencies; potential savings are not estimated without a verified counterfactual.

## Requirements

- Python 3.11 or newer
- Node.js 20 or newer
- npm
- Optional: Ollama for local model-backed chat

## Setup and launch

From the project root:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r backend/requirements.txt
cp .env.example .env
```

Start the backend in one terminal:

```bash
cd /path/to/beforeyoubuy-ai
source .venv/bin/activate
set -a
source .env
set +a
uvicorn backend.main:app --reload --port 8000
```

Start the frontend in another terminal:

```bash
cd /path/to/beforeyoubuy-ai/frontend
npm install
npm run dev
```

Open <http://localhost:5173>. The Vite development server proxies `/api` and `/health` to `http://127.0.0.1:8000`.

## Ollama and local RAG

If Ollama is installed, start its local service and inspect existing models first:

```bash
ollama list
```

The example configuration enables Ollama with `qwen3:4b`; set `LLM_PROVIDER=none` deliberately when deterministic fallback is desired. The app discovers installed models through `GET /api/models`. Runtime reachability and installed-model state are available from `GET /api/provider/status` and are not inferred from the configured model name.

```dotenv
LLM_PROVIDER=ollama
OLLAMA_MODEL=qwen3:4b
OLLAMA_BASE_URL=http://127.0.0.1:11434
OLLAMA_TIMEOUT_SECONDS=35
```

Restart the backend after changing the environment. Model choices are validated against Ollama's model list. The discovered local model is selectable in the UI; each reply reports the actual provider/model. Ollama is requested to return the `StructuredChatResponse` JSON schema with `think: false`; the backend validates the full response, ignores the separate Ollama `thinking` field, and safely extracts schema-valid content after an explicit closing think marker. Empty, malformed, reasoning-only, unsafe, or invented-money responses fall back explicitly. The default provider timeout is 35 seconds, which leaves bounded room for the observed cold-start latency without keeping the UI waiting indefinitely. No model was fine-tuned or downloaded; do not download additional weights without deciding first.

For a chat request, local Markdown notes are normalized and ranked first. The top notes are passed into the Qwen system context with topic, title, source URL, and bullet evidence. The prompt treats retrieved text as untrusted reference material: it cannot override user facts or the deterministic verdict. The API returns `rag.enabled`, `rag.used`, `rag.evidence_count`, and sanitized source metadata. The frontend labels each assistant turn as Qwen with RAG, Qwen without relevant evidence, or deterministic fallback. This RAG path is educational guidance, not product research or verified pricing.

### Gemini fallback

The default chain is local-first: Ollama is attempted first, Gemini once when Ollama is disabled or unavailable, then the deterministic response. Gemini uses the official `google-genai` SDK and the configured model is checked through the Gemini model-list API when Gemini is selected. The example uses supported `gemini-2.5-flash-lite`; set `GEMINI_API_KEY` and `GEMINI_MODEL` in `.env` without committing the key. Missing credentials, authentication/quota errors, timeouts, unavailable models, and malformed responses remain safe deterministic fallbacks and never expose provider details or secrets.

### Sentiment labels and verification status

The live chat returns `sentiment_analysis` with separate `polarity`, `emotions`, and `purchase_signals`. Each detected label includes evidence spans, a deterministic method, and heuristic confidence; absent signals remain `unknown`. Sentiment never changes the canonical decision. The original `sentiment_labels.csv` has 30 balanced polarity examples and is **PARTIALLY_VERIFIED**: its schema and labels are auditable, but the repository contains no independent annotation record. `sentiment_context_labels.jsonl` contains 10 project-authored examples and is **UNVERIFIED**, not human validation. The acquired public-review sentiment labels are **PARTIALLY_VERIFIED** source annotations: their dataset card attributes them to Mistral classification, not independent human ground truth. Run `python scripts/validate_sentiment_labels.py` and `python scripts/evaluate_sentiment.py` to reproduce schema and baseline reports.

The chat accepts up to 2,000 characters per message, retains a bounded in-memory turn history for follow-ups, supports Enter to send and Shift+Enter for a new line, and can start a separate conversation without clearing the assessment. While a reply is being generated, an accessible animated thinking indicator is shown. Local model replies are not streamed; the indicator remains visible until the response or timeout returns.

When no usable model response is available, deterministic chat recognizes purchase intent and asks at most one primary follow-up at a time. It tracks supported facts, corrections, unresolved conflicts, motivations, usage, current-item condition, and per-fact evidence metadata in temporary per-session memory. The optional LLM drafts conversation wording only; deterministic extraction remains the source of assessed facts and the canonical recommendation. Financial statements such as “I have enough savings” or “I can afford it” are retained as user-reported claims, separately from budget and product price, and are not treated as a full affordability finding. Replacement urgency is tracked separately from whether a specific model has been evaluated. Per-session requests are serialized to prevent interleaved updates. The API calculates separate purchase-pressure and regret-risk reflections, lists unknown inputs, and returns a provisional BUY / WAIT / DON'T BUY decision signal. The decision signal and scores are heuristics, not validated predictions or financial advice; unsupported or conflicting information remains unknown. This bounded fallback is not an unrestricted language model.

## Dashboard and journal

The Overview dashboard is visible at the initial page route and has an empty state when there are no saved decisions. Its loading and API-error states remain distinct from an empty database, with a retry control. Saved decisions appear in the journal, where their status can be changed and individual entries removed. Export downloads local data as JSON; full deletion requires an explicit confirmation. Currency totals are shown separately. Potential savings remain unavailable unless an evidenced comparison supports them.

The primary navigation scrolls to Overview, assessment, chat, product tracking, and the journal sections. Watchlist editing and deletion use API methods enabled by the configured explicit CORS policy.

## Assessment and safety behavior

- Item name and price are required; usage and financial figures are optional and remain missing if not supplied.
- Cost per use uses the entered weekly use count × 52 weeks × entered lifespan. The categorical API field also supports documented frequency estimates.
- A budget comparison uses only the budget and recurring cost explicitly entered. A monthly-surplus comparison shows price divided by that stated figure; it is not independently verified and does not establish affordability.
- Sentiment is a small keyword signal with explicit evidence and low/moderate confidence; it is not a psychological assessment.
- The 0–100 regret-risk value is an explainable design heuristic. Its factors are shown, and it is not scientifically validated or a prediction of actual regret.
- Both the guided assessment and conversation use the same deterministic recommendation rules. They return BUY, WAIT, or DON'T BUY and mark the result provisional while relevant information is missing. The existing text reflection remains available separately from that decision signal.
- The displayed confidence label describes completeness of decision-relevant facts only; it is not a probability, validated confidence interval, or prediction.
- Purchase pressure measures explicit timing and social/emotional pressure; missing information is reported separately and does not itself count as pressure. Regret-risk points consider stated current-item condition, discretionary purpose, pressure, unchecked alternatives or repair, and a known budget overrun.
- Chat recalculates its deterministic reflection scores and missing-input list after each message. Correction language such as “actually” updates supported facts; conflicting figures are withheld from assessment until clarified. Extracted facts expose their source, confidence, and timestamp. Provisional scores do not imply that missing item price or financial information is known.
- `backend/data/purchase_scenarios.jsonl` contains eight project-authored synthetic scenarios, including optional-upgrade and time-sensitive replacement cases. It is labelled synthetic and CC0-1.0; it is not customer data, empirical purchase-outcome evidence, or training data. The scenarios check fact extraction and decision invariants, not fixed verdicts or scientific calibration. No external dataset is used.
- Retrieved note text is reference content, not executable instructions. Sources are linked in the UI and included in the PDF.

## PDF export

Use **Download PDF review** after an assessment, or call the endpoint directly:

```bash
curl -X POST http://127.0.0.1:8000/api/report/pdf \
  -H 'Content-Type: application/json' \
  --data '{"name":"Headphones","price":8000,"currency":"INR","uses_per_week":5,"lifespan_years":2,"scenario":"I am comparing options"}' \
  --output beforeyoubuy-review.pdf
```

The endpoint returns `application/pdf` as `beforeyoubuy-review.pdf`. The document includes supplied item details, calculations, sentiment/tone signals, heuristic factors and limitations, retrieved references, and a UTC timestamp.

## Tests and evaluation

```bash
# Backend unit and HTTP integration tests
source .venv/bin/activate
pytest -q

# Rebuild and validate the local purchase corpus and lexical retrieval index
python scripts/generate_purchase_dataset.py
python scripts/validate_purchase_dataset.py
python scripts/build_purchase_index.py
python scripts/evaluate_purchase_retrieval.py

# Optional dense indexes/evaluation; requires the already-installed local
# sentence-transformers stack and its cached model, and does not download one.
python scripts/build_dense_indexes.py
python scripts/evaluate_dense_retrieval.py

# Frontend type check and production build
cd frontend
npm run build

# Sentiment evaluator over 30 human-labeled examples
cd ..
python scripts/evaluate_sentiment.py

# Human-response evaluation summary (pending rows are not scored)
python scripts/summarize_evaluations.py
```

Pytest sets `BEFOREYOUBUY_DB_PATH` to a disposable temporary SQLite database before application modules are imported and removes that directory at the end of the test session. Tests do not use or delete the local application database.

`sentiment_labels.csv` contains a small manually labeled sample; its metrics describe only that sample and the deterministic lexicon. `response_evaluation.csv` contains five fallback replies from the demo that were manually scored against the stated safety/relevance criteria; it is not an LLM evaluation. Live Ollama probes were attempted, but their reasoning-style output was rejected, so no successful model-response evaluation is claimed. Historical experiment files and metrics from other versions were not present in this rebuild.

The purchase corpus contains 2,048 reproducibly generated synthetic records across 16 categories. It is a retrieval/evaluation fixture, not customer data or empirical evidence. The index uses a local 256-dimensional hashed word/bigram representation; retrieval is optional and cannot override explicit conversation facts. The held-out retrieval report compares it with lexical overlap, but its relevance labels are generated consistency labels rather than human judgments.

The regret-risk weights are source-configured in `backend/assessment.py` and are listed in the result assumptions. They are reflection prompts, not a scientifically validated score. The point-based assessment also produces non-directive comparison text; no score is a purchase instruction.

## Clean source ZIP

From the project root, run `python scripts/package_project.py`. This scans for secrets, excludes `.env`, dependencies, caches, build output, and existing archives, then creates and verifies `deliverables/BeforeYouBuy-AI.zip`.

## API endpoints

- `GET /health` — service and configured chat mode
- `GET /api/models` — installed Ollama model names when enabled
- `GET /api/provider/status` — sanitized configured/enabled/reachable/model status
- `POST /api/assess` — validated assessment result, sources, missing inputs, tools used, and processing time
- `POST /api/chat` — short-lived conversation context, explicit fact extraction/corrections, deterministic assessment, RAG provenance, `decision_state`, and a deterministic or schema-validated Ollama reply
- `POST /api/report/pdf` — downloadable PDF
- `POST /api/products/search` — optional provider-backed India/INR search with explicit live/cache/unavailable states
- `GET /api/watchlist/{id}/history` and `POST /api/watchlist/{id}/refresh` — observations and manual refresh
- `GET /api/data/export` and `DELETE /api/data?confirm=true` — local data management
- `GET /docs` — interactive API documentation

## Workspace, persistence, and product data

Decisions and watchlist entries are stored locally in SQLite at `data/beforeyoubuy.sqlite3` by default. Set `BEFOREYOUBUY_DB_PATH` to choose another local path. The database is created on startup, is not committed, and can be backed up by copying it while the app is stopped. The API exposes `GET/POST /api/decisions`, `GET/POST/DELETE /api/watchlist`, and `GET /api/analytics`.

The dashboard distinguishes planned spending, confirmed spending, postponed decisions, and user-confirmed savings. Monetary totals and spending trends are grouped by currency; combined totals are withheld when multiple currencies are present. Potential savings are not estimated without a verified counterfactual. A new account shows an empty state; watchlist entries are not purchases. Conversation history remains temporary in backend memory and is not written into the journal unless the user explicitly saves an assessment.

Product offers use the provider-neutral normalizer in `backend/market.py`. Offers with invalid prices or unsafe URLs are rejected. The selected optional provider is SerpApi Google Shopping: its documented endpoint supports `engine=google_shopping`, India localization via `gl=in`, and structured shopping fields such as title, source, price, product link, and product id. It is a credentialed external service with account-specific usage limits; this repository does not claim access or pricing for an account. `SERPAPI_KEY` is required and stays on the backend. `PRODUCT_PROVIDER_TIMEOUT_SECONDS` defaults to 12 and `PRODUCT_CACHE_SECONDS` defaults to 900. If `SERPAPI_KEY` is absent, live India/INR search is not operational and manual product entry remains available. Refresh never substitutes a different variant when the requested variant is absent; stale offers are visibly labeled and penalized in shortlist ranking. Amazon India Creators API and Flipkart Affiliate APIs were evaluated but not selected: both require marketplace/affiliate enrollment and access eligibility, and neither was available for verification in this environment. Historical prices and alerts are not claimed unless verified observations and a scheduler are actually configured.

Price observations are stored in the `price_observations` table and linked to watchlist items. A history view requires at least two observations before describing a trend. `GET /api/data/export` exports local JSON; `DELETE /api/data?confirm=true` permanently deletes local decisions, watchlist entries, and observations. Watchlist refresh runs only when manually requested while the application is running. No external notification channel is configured, so the app reports in-app target status but never claims an alert was delivered.

The assessment now returns two separate deterministic heuristics: `purchase_pressure` and `regret_risk`. Their factors and limitations are included in the response. Neither is a validated probability, diagnosis, or instruction to buy or skip. AI interprets context only; it does not set the numeric scores.

## Privacy, assumptions, and known limitations

- The backend does not persist assessment requests or chat. Chat context is process memory only; do not run a multi-user public service without adding appropriate session controls and privacy review.
- The frontend does not store submitted assessments. Browser data is not uploaded except in requests needed for assessment/chat/PDF.
- Tone detection and heuristic scoring are intentionally limited and can miss nuance.
- The sentiment set is small and not a representative benchmark.
- The cited learning notes point to official financial-education resources but are not individualized financial advice or product research.
- The PDF report is generated from request data only; it does not include API keys or environment values.

See [PROJECT_AUDIT.md](./PROJECT_AUDIT.md), [TEST_RESULTS.md](./TEST_RESULTS.md), and [report/experiment_report.md](./report/experiment_report.md) for implementation status, actual validation results, and limitations.
