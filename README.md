# BeforeYouBuy AI

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11+-3776AB?logo=python&logoColor=white" alt="Python 3.11+" />
  <img src="https://img.shields.io/badge/FastAPI-0.110+-009688?logo=fastapi&logoColor=white" alt="FastAPI" />
  <img src="https://img.shields.io/badge/React-18-61DAFB?logo=react&logoColor=black" alt="React 18" />
  <img src="https://img.shields.io/badge/TypeScript-5-3178C6?logo=typescript&logoColor=white" alt="TypeScript 5" />
  <img src="https://img.shields.io/badge/Privacy-Local--First-0A7EA4" alt="Local-first" />
</p>

BeforeYouBuy AI is a local-first purchase reflection app that helps users pause before buying. It combines transparent calculations, source-grounded notes, conversational follow-up, and a lightweight decision dashboard to support smarter buying decisions without depending on a paid cloud service.

## Why this project matters

Impulse purchases are often driven by emotion, missing context, or unclear trade-offs. BeforeYouBuy AI helps users answer the questions that matter:

- How much does this really cost over time?
- What assumptions are hidden in the purchase decision?
- What evidence supports or weakens the decision?
- Is the buying decision based on a rational budget check or pressure and emotion?

## Highlights

- Transparent assessment logic for BUY / WAIT / DON'T BUY decisions
- Deterministic recommendation engine with explainable factors
- Optional local LLM support using Ollama
- RAG-style educational notes grounded in local knowledge sources
- PDF export for review and saving decision context
- Watchlist and journal tracking for product decisions
- Privacy-focused: no paid API required and no permanent assessment storage by default

## Architecture

- Frontend: React + TypeScript + Vite
- Backend: FastAPI + Pydantic
- Assessment layer: deterministic cost, regret-risk, and purchase-pressure heuristics
- Retrieval layer: local keyword retrieval and optional dense index support
- LLM layer: local Ollama integration with deterministic fallback
- Persistence: SQLite for local decision and watchlist data

## Demo workflow

1. Enter product details such as item name, price, usage, and budget.
2. Review the transparent assessment breakdown and supporting reasoning.
3. Ask follow-up questions in chat for clarification or decision support.
4. Save the decision to the local dashboard and review your history.
5. Export a review PDF when you want to keep a record.

## Quick start

### Prerequisites

- Python 3.11+
- Node.js 20+
- npm
- Optional: Ollama for local model-backed chat

### 1) Set up the backend

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r backend/requirements.txt
cp .env.example .env
```

### 2) Run the backend

```bash
source .venv/bin/activate
set -a
source .env
set +a
uvicorn backend.main:app --reload --port 8000
```

### 3) Run the frontend

```bash
cd frontend
npm install
npm run dev
```

Open http://localhost:5173

## Local LLM support

If Ollama is installed, it can be used for local chat and retrieval support. The app checks available models at runtime and falls back gracefully when needed.

```bash
ollama list
```

Example environment configuration:

```dotenv
LLM_PROVIDER=ollama
OLLAMA_MODEL=qwen3:4b
OLLAMA_BASE_URL=http://127.0.0.1:11434
OLLAMA_TIMEOUT_SECONDS=35
```

## Project structure

```text
before-u-buy/
├── backend/
│   ├── data/
│   ├── main.py
│   ├── assessment.py
│   ├── sentiment.py
│   ├── retrieval.py
│   ├── knowledge.py
│   ├── market.py
│   └── requirements.txt
├── frontend/
│   ├── src/
│   ├── package.json
│   ├── vite.config.ts
│   └── tsconfig.json
├── tests/
├── scripts/
├── data/
├── .env.example
├── README.md
├── PROJECT_AUDIT.md
├── TEST_RESULTS.md
├── pytest.ini
└── requirements-retrieval.txt
```

## Main capabilities

- Purchase assessment with transparent decision reasoning
- Follow-up conversational review
- Local knowledge retrieval with source provenance
- Product tracking and spending observations
- Local journal and historical review
- PDF report generation
- Deterministic fallback when model services are unavailable

## API overview

- GET /health
- GET /api/models
- GET /api/provider/status
- POST /api/assess
- POST /api/chat
- POST /api/report/pdf
- GET /api/watchlist/{id}/history
- POST /api/watchlist/{id}/refresh
- GET /api/data/export
- DELETE /api/data?confirm=true
- GET /docs

## Privacy and safety

This project is designed to be transparent and local-first. It does not rely on a paid API for core assessment logic, and the app keeps decision support focused on explainability instead of opaque black-box outputs.

Important notes:

- Assessment and chat are intended for reflection, not financial advice.
- Retrieval sources are educational and reference-based.
- Heuristic scores are explainable decision aids, not validated medical or financial predictions.

## Validation and testing

```bash
source .venv/bin/activate
pytest -q
```

Frontend build:

```bash
cd frontend
npm run build
```

## Why it looks good on GitHub

This repository is built to be easy to understand at a glance:

- Clear project purpose
- Transparent decision-making approach
- Local-first architecture
- Strong documentation and setup flow
- Distinct product and technical value proposition
- Honest limitations and safety framing

## Contributing

Contributions, issues, and feature ideas are welcome. If you want to improve the app, start by opening an issue or proposing a clean, focused change that keeps the project transparent and easy to reason about.

## License

This project does not currently declare a license in the repository metadata. If you plan to share or distribute it publicly, add an appropriate license such as MIT or Apache 2.0.

---

Built for better buying decisions, less regret, and more clarity.
