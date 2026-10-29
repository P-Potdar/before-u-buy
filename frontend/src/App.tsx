import { FormEvent, KeyboardEvent, useEffect, useMemo, useRef, useState } from 'react';

type Source = {
  title: string;
  topic: string;
  url: string;
  points: string[];
};

type Assessment = {
  name: string;
  currency: string;
  price: number;
  recommendation: string;
  decision: 'BUY' | 'WAIT' | "DON'T BUY";
  decision_explanation: string;
  provisional: boolean;
  confidence: { level: string; basis: string };
  summary: string;
  factors: string[];
  factor_weights: number[];
  next_step: string;
  estimated_cost_per_use: number | null;
  estimated_total_uses: number | null;
  recurring_12_month_total: number;
  recurring_monthly_provided: boolean;
  budget_exceeded_by: number | null;
  affordability: {
    budget_provided: boolean;
    budget_comparison_total: number;
    budget_exceeded_by: number | null;
    monthly_surplus_provided: boolean;
    months_of_stated_surplus: number | null;
    note: string;
  };
  sentiment: {
    label: string;
    confidence: string;
    emotional_cues: string[];
    evidence: string[];
    limitation: string;
  };
  regret_risk: {
    score: number;
    scale: string;
    band: string;
    factors: { factor: string; points: number }[];
    limitations: string;
    explanation: string;
  };
  purchase_pressure: {
    score: number;
    band: string;
    factors: { factor: string; points: number }[];
    explanation: string;
    limitations: string;
  };
  missing_inputs: string[];
  knowledge_sources: Source[];
  tools_used: string[];
  processing_ms: number;
  assumptions: string[];
  limitations: string[];
  disclaimer: string;
};

type ChatTurn = {
  role: 'user' | 'assistant';
  text: string;
  provider?: string;
  model?: string | null;
  fallbackReason?: string | null;
  fallbackOccurred?: boolean;
  ragUsed?: boolean;
  evidenceCount?: number;
};
type DecisionScore = {
  score: number;
  band: string;
  explanation: string;
  limitations: string;
  factors: { factor: string; points: number }[];
};
type ChatFinancialContext = {
  product_price: number | null;
  stated_budget: number | null;
  currency: string | null;
  user_affordability: 'reported_affordable' | 'reported_not_affordable' | null;
  savings_adequacy: 'reported_sufficient' | 'reported_insufficient' | null;
  essential_expenses_affected: boolean | null;
  savings_affected: boolean | null;
  upcoming_expenses: boolean | null;
};
type ChatReplacementContext = {
  purchase_timing: 'urgent' | 'not_urgent' | null;
  waiting_period: string | null;
  deadline_consequences: boolean | null;
  usable_alternative_available: boolean | null;
};
type ChatDecisionState = {
  signal: string;
  signal_explanation: string;
  recommendation: 'BUY' | 'WAIT' | "DON'T BUY";
  recommendation_explanation: string;
  confidence: { level: string; basis: string };
  provisional: boolean;
  purchase_pressure: DecisionScore;
  regret_risk: DecisionScore;
  estimated_total_uses: number | null;
  estimated_cost_per_use: number | null;
  cost_per_use_basis: {
    uses_per_week: number | null;
    uses_per_week_source: string | null;
    lifespan_years: number | null;
    lifespan_source: string | null;
    weeks_per_year: number;
    status: string;
  };
  budget_exceeded_by: number | null;
  financial_context: ChatFinancialContext;
  replacement_context: ChatReplacementContext;
  supporting_factors: string[];
  factors_to_consider: string[];
  missing_inputs: string[];
  changed_facts: { field: string; previous: unknown; current: unknown }[];
  limitations: string;
};
type ChatApiResponse = {
  reply: string;
  session_id: string;
  processing_ms: number;
  provider: string;
  model: string | null;
  fallback_reason: string | null;
  fallback_occurred: boolean;
  tools_used: string[];
  knowledge_sources: Source[];
  assessment_update: Assessment | null;
  session_facts: Record<string, unknown>;
  decision_state: ChatDecisionState;
  rag: { enabled: boolean; used: boolean; evidence_count: number };
  sentiment_analysis: {
    polarity: { label: string };
    emotions: { label: string }[];
    purchase_signals: Record<string, { value: boolean | null; evidence: string[] }>;
    insufficient_evidence: boolean;
    provenance: string;
  };
};
type ModelCatalog = {
  provider: string;
  connected: boolean;
  models: string[];
  selected_model: string | null;
};

type ModelStatus = 'checking' | 'connected' | 'offline' | 'disabled' | 'unknown';
type ProviderStatusResponse = {
  configured_provider: string;
  enabled: boolean;
  reachable: boolean;
  models: string[];
  selected_model: string | null;
  status: string;
  reason: string | null;
};
type DecisionRecord = {
  id: string;
  created_at: string;
  name: string;
  category: string;
  price: number;
  currency: string;
  status: string;
  final_decision: string | null;
  confirmed_savings: number;
  data: Record<string, unknown>;
};

const apiBase = import.meta.env.VITE_API_BASE_URL ?? '';

function currency(value: number, code: string): string {
  return new Intl.NumberFormat(undefined, {
    style: 'currency',
    currency: code,
    maximumFractionDigits: 2,
  }).format(value);
}

async function readError(response: Response): Promise<string> {
  try {
    const body = await response.json() as { detail?: string | { msg?: string }[] };
    if (typeof body.detail === 'string') return body.detail;
    if (Array.isArray(body.detail)) return body.detail.map(item => item.msg ?? 'Invalid input').join('; ');
  } catch {
    return `Request failed with status ${response.status}.`;
  }
  return `Request failed with status ${response.status}.`;
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value);
}

function isStringArray(value: unknown): value is string[] {
  return Array.isArray(value) && value.every(item => typeof item === 'string');
}

function assertChatApiResponse(value: unknown): asserts value is ChatApiResponse {
  if (
    !isRecord(value)
    || typeof value.reply !== 'string'
    || !value.reply.trim()
    || typeof value.session_id !== 'string'
    || typeof value.processing_ms !== 'number'
    || typeof value.provider !== 'string'
    || (value.model !== null && typeof value.model !== 'string')
    || (value.fallback_reason !== null && typeof value.fallback_reason !== 'string')
    || typeof value.fallback_occurred !== 'boolean'
    || !isStringArray(value.tools_used)
    || !Array.isArray(value.knowledge_sources)
    || !isRecord(value.session_facts)
    || !isRecord(value.rag)
    || typeof value.rag.enabled !== 'boolean'
    || typeof value.rag.used !== 'boolean'
    || typeof value.rag.evidence_count !== 'number'
    || !isRecord(value.sentiment_analysis)
    || !isRecord(value.sentiment_analysis.polarity)
    || typeof value.sentiment_analysis.polarity.label !== 'string'
    || !Array.isArray(value.sentiment_analysis.emotions)
    || !isRecord(value.sentiment_analysis.purchase_signals)
    || typeof value.sentiment_analysis.insufficient_evidence !== 'boolean'
    || (value.assessment_update !== null && !isRecord(value.assessment_update))
  ) {
    throw new Error('The chat response was missing required conversation data.');
  }
  const state = value.decision_state;
  const financial = isRecord(state) ? state.financial_context : null;
  const replacement = isRecord(state) ? state.replacement_context : null;
  const nullableNumber = (item: unknown) => item === null || typeof item === 'number';
  const nullableBoolean = (item: unknown) => item === null || typeof item === 'boolean';
  const nullableEnum = (item: unknown, allowed: string[]) => item === null || allowed.includes(String(item));
  if (
    !isRecord(state)
    || !['BUY', 'WAIT', "DON'T BUY"].includes(String(state.recommendation))
    || typeof state.recommendation_explanation !== 'string'
    || typeof state.provisional !== 'boolean'
    || !isRecord(state.confidence)
    || typeof state.confidence.level !== 'string'
    || typeof state.confidence.basis !== 'string'
    || !isRecord(financial)
    || !nullableNumber(financial.product_price)
    || !nullableNumber(financial.stated_budget)
    || (financial.currency !== null && !['INR', 'USD', 'GBP', 'EUR'].includes(String(financial.currency)))
    || !nullableEnum(financial.user_affordability, ['reported_affordable', 'reported_not_affordable'])
    || !nullableEnum(financial.savings_adequacy, ['reported_sufficient', 'reported_insufficient'])
    || !nullableBoolean(financial.essential_expenses_affected)
    || !nullableBoolean(financial.savings_affected)
    || !isRecord(replacement)
    || !nullableEnum(replacement.purchase_timing, ['urgent', 'not_urgent'])
    || (replacement.waiting_period !== null && typeof replacement.waiting_period !== 'string')
    || !nullableBoolean(replacement.deadline_consequences)
    || !nullableBoolean(replacement.usable_alternative_available)
    || !isStringArray(state.missing_inputs)
    || !Array.isArray(state.changed_facts)
    || !state.changed_facts.every(item => isRecord(item) && typeof item.field === 'string')
  ) {
    throw new Error('The chat response did not include a valid canonical recommendation.');
  }
  for (const key of ['purchase_pressure', 'regret_risk']) {
    const score = state[key];
    if (
      !isRecord(score)
      || typeof score.score !== 'number'
      || !Array.isArray(score.factors)
      || !score.factors.every(item => isRecord(item) && typeof item.factor === 'string' && typeof item.points === 'number')
    ) {
      throw new Error('The chat response did not include valid decision scores.');
    }
  }
}

function App() {
  const [name, setName] = useState('');
  const [price, setPrice] = useState('');
  const [currencyCode, setCurrencyCode] = useState('INR');
  const [scenario, setScenario] = useState('');
  const [need, setNeed] = useState('');
  const [current, setCurrent] = useState('');
  const [urgency, setUrgency] = useState('');
  const [usesPerWeek, setUsesPerWeek] = useState('');
  const [lifespan, setLifespan] = useState('');
  const [budget, setBudget] = useState('');
  const [monthlySurplus, setMonthlySurplus] = useState('');
  const [savings, setSavings] = useState('');
  const [recurring, setRecurring] = useState('');
  const [result, setResult] = useState<Assessment | null>(null);
  const [error, setError] = useState('');
  const [busy, setBusy] = useState(false);
  const [chatBusy, setChatBusy] = useState(false);
  const [chatText, setChatText] = useState('');
  const [chatTurns, setChatTurns] = useState<ChatTurn[]>([]);
  const [sessionId, setSessionId] = useState<string>(() => crypto.randomUUID());
  const [chatTiming, setChatTiming] = useState<number | null>(null);
  const [chatProvider, setChatProvider] = useState('');
  const [activeModel, setActiveModel] = useState('');
  const [chatFallbackReason, setChatFallbackReason] = useState('');
  const [modelChoices, setModelChoices] = useState<string[]>([]);
  const [selectedModel, setSelectedModel] = useState('');
  const [modelStatus, setModelStatus] = useState<ModelStatus>('checking');
  const [providerStatusText, setProviderStatusText] = useState('Checking Ollama status…');
  const [retryMessage, setRetryMessage] = useState('');
  const [chatTools, setChatTools] = useState<string[]>([]);
  const [followupSources, setFollowupSources] = useState<Source[]>([]);
  const [sessionFacts, setSessionFacts] = useState<Record<string, unknown>>({});
  const [chatDecision, setChatDecision] = useState<ChatDecisionState | null>(null);
  const [chatSentiment, setChatSentiment] = useState<ChatApiResponse['sentiment_analysis'] | null>(null);
  const chatLogRef = useRef<HTMLDivElement>(null);
  const chatBusyRef = useRef(false);
  const decisionGeneration = useRef(0);
  const assessmentGeneration = useRef(0);
  const [assessmentContext, setAssessmentContext] = useState<Record<string, string | number> | null>(null);
  const [assessmentContextPending, setAssessmentContextPending] = useState(false);
  const [activeSection, setActiveSection] = useState('overview');
  const [analytics, setAnalytics] = useState<{ planned_spending: number | null; confirmed_spending: number | null; decision_count: number; tracked_products: number; awaiting_decision: number; confirmed_savings: number | null; potential_savings: number | null; totals_by_currency: Record<string, { planned_spending: number; confirmed_spending: number; confirmed_savings: number }>; average_purchase_pressure: number | null; average_regret_risk: number | null; outcomes: Record<string, number>; assumptions: string[] } | null>(null);
  const [decisions, setDecisions] = useState<DecisionRecord[]>([]);
  const [watchlistItems, setWatchlistItems] = useState<{ id: string; name: string; variant: string; retailer: string; price: number | null; target_price: number | null; currency: string; url: string; source: string; status: string }[]>([]);
  const [workspaceLoading, setWorkspaceLoading] = useState(true);
  const [workspaceError, setWorkspaceError] = useState('');
  const [saveNotice, setSaveNotice] = useState('');
  const [searchQuery, setSearchQuery] = useState('');
  const [searchCategory, setSearchCategory] = useState('');
  const [searchResults, setSearchResults] = useState<{ name: string; retailer: string; price: number; currency: string; variant: string; url: string; source: string; observed_at: string; status: string; offer_id: string }[]>([]);
  const [searchStatus, setSearchStatus] = useState('');
  const [historyByItem, setHistoryByItem] = useState<Record<string, { observed_at: string; price: number; currency: string; status: string }[]>>({});

  useEffect(() => {
    let active = true;
    Promise.all([
      fetch(`${apiBase}/api/models`, { headers: { Accept: 'application/json' } }),
      fetch(`${apiBase}/api/provider/status`, { headers: { Accept: 'application/json' } }),
    ]).then(async ([modelsResponse, statusResponse]) => {
      if (!modelsResponse.ok || !statusResponse.ok) throw new Error('Provider discovery failed.');
      return [await modelsResponse.json() as ModelCatalog, await statusResponse.json() as ProviderStatusResponse] as const;
    }).then(([catalog, status]) => {
        if (!active) return;
        setModelChoices(catalog.models);
        setSelectedModel(catalog.selected_model ?? '');
        setProviderStatusText(status.reason ?? (status.status === 'available' ? 'Ollama enabled and reachable.' : `Ollama ${status.status}.`));
        setModelStatus(status.status === 'disabled' ? 'disabled' : status.status === 'available' ? 'connected' : status.reachable ? 'offline' : 'offline');
      })
      .catch(() => {
        if (active) {
          setModelStatus('unknown');
          setProviderStatusText('Ollama status unknown');
        }
      });
    return () => { active = false; };
  }, []);

  async function refreshWorkspace() {
    setWorkspaceLoading(true);
    setWorkspaceError('');
    try {
      const [analyticsResponse, watchlistResponse, decisionsResponse] = await Promise.all([
        fetch(`${apiBase}/api/analytics`),
        fetch(`${apiBase}/api/watchlist`),
        fetch(`${apiBase}/api/decisions`),
      ]);
      for (const response of [analyticsResponse, watchlistResponse, decisionsResponse]) {
        if (!response.ok) throw new Error(`Workspace data could not be loaded (HTTP ${response.status}).`);
      }
      const [analyticsData, watchlistData, decisionsData] = await Promise.all([
        analyticsResponse.json(),
        watchlistResponse.json(),
        decisionsResponse.json(),
      ]);
      setAnalytics(analyticsData);
      setWatchlistItems(watchlistData.items ?? []);
      setDecisions(decisionsData.items ?? []);
    } catch (cause) {
      setWorkspaceError(cause instanceof Error ? cause.message : 'Workspace data could not be loaded.');
    } finally {
      setWorkspaceLoading(false);
    }
  }

  useEffect(() => { void refreshWorkspace(); }, []);

  useEffect(() => {
    chatLogRef.current?.scrollTo({ top: chatLogRef.current.scrollHeight, behavior: 'auto' });
  }, [chatBusy, chatTurns]);

  const payload = useMemo(() => {
    const data: Record<string, string | number> = {
      name: name.trim(),
      price: Number(price),
      currency: currencyCode,
      scenario: scenario.trim(),
    };
    if (need) data.need = need;
    if (current) data.current = current;
    if (urgency) data.urgency = urgency;
    if (usesPerWeek) data.uses_per_week = Number(usesPerWeek);
    if (lifespan) data.lifespan_years = Number(lifespan);
    if (budget !== '') data.budget = Number(budget);
    if (monthlySurplus !== '') data.monthly_surplus = Number(monthlySurplus);
    if (savings !== '') data.savings_allocated = Number(savings);
    if (recurring !== '') data.recurring_monthly = Number(recurring);
    return data;
  }, [budget, currencyCode, current, lifespan, monthlySurplus, name, need, price, recurring, savings, scenario, urgency, usesPerWeek]);
  const analyticsCurrencyTotals = Object.entries(analytics?.totals_by_currency ?? {});

  function reset() {
    setName('');
    setPrice('');
    setCurrencyCode('INR');
    setScenario('');
    setNeed('');
    setCurrent('');
    setUrgency('');
    setUsesPerWeek('');
    setLifespan('');
    setBudget('');
    setMonthlySurplus('');
    setSavings('');
    setRecurring('');
    setResult(null);
    setError('');
    setChatText('');
    setChatTurns([]);
    setSessionId(crypto.randomUUID());
    setChatTiming(null);
    setChatProvider('');
    setActiveModel('');
    setChatFallbackReason('');
    setRetryMessage('');
    setChatTools([]);
    setFollowupSources([]);
    setSessionFacts({});
    setChatDecision(null);
    setChatSentiment(null);
    setAssessmentContext(null);
    setAssessmentContextPending(false);
  }

  function newConversation() {
    decisionGeneration.current += 1;
    setChatTurns([]);
    setSessionId(crypto.randomUUID());
    setChatTiming(null);
    setChatProvider('');
    setActiveModel('');
    setChatFallbackReason('');
    setChatTools([]);
    setFollowupSources([]);
    setSessionFacts({});
    setChatDecision(null);
    setChatSentiment(null);
    // A new conversation must not inherit facts from a guided assessment or
    // another chat. The assessment result remains visible in its own section,
    // but it is not sent as context to the new chat session.
    setAssessmentContext(null);
    setAssessmentContextPending(false);
    setRetryMessage('');
    setError('');
  }

  function handleChatKeyDown(event: KeyboardEvent<HTMLTextAreaElement>) {
    if (event.key === 'Enter' && !event.shiftKey && !event.nativeEvent.isComposing) {
      event.preventDefault();
      event.currentTarget.form?.requestSubmit();
    }
  }

  async function runAssessment(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const requestGeneration = ++decisionGeneration.current;
    const assessmentRequest = ++assessmentGeneration.current;
    setBusy(true);
    setError('');
    try {
      const response = await fetch(`${apiBase}/api/assess`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', Accept: 'application/json' },
        body: JSON.stringify(payload),
      });
      if (!response.ok) throw new Error(await readError(response));
      const assessment = await response.json() as Assessment;
      if (requestGeneration !== decisionGeneration.current) return;
      setResult(assessment);
      setAssessmentContext(payload);
      setAssessmentContextPending(true);
      setSessionId(crypto.randomUUID());
      // Keep the structured assessment in the assessment section. It is not a
      // conversational assistant message and must not appear in chat history.
      setChatTurns([]);
      setChatDecision(null);
      setSessionFacts({});
    } catch (cause) {
      if (requestGeneration === decisionGeneration.current) {
        setError(cause instanceof Error ? cause.message : 'Assessment could not be completed.');
      }
    } finally {
      if (assessmentRequest === assessmentGeneration.current) setBusy(false);
    }
  }

  async function downloadPdf() {
    setBusy(true);
    setError('');
    try {
      const response = await fetch(`${apiBase}/api/report/pdf`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', Accept: 'application/pdf' },
        body: JSON.stringify(payload),
      });
      if (!response.ok) throw new Error(await readError(response));
      const blob = await response.blob();
      if (blob.size < 1000 || blob.type !== 'application/pdf') throw new Error('The server did not return a valid PDF.');
      const url = URL.createObjectURL(blob);
      const anchor = document.createElement('a');
      anchor.href = url;
      anchor.download = 'beforeyoubuy-review.pdf';
      document.body.appendChild(anchor);
      anchor.click();
      anchor.remove();
      window.setTimeout(() => URL.revokeObjectURL(url), 1000);
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : 'PDF export failed.');
    } finally {
      setBusy(false);
    }
  }

  async function saveCurrentDecision(status = 'awaiting') {
    if (!result) return;
    setSaveNotice('Saving locally…');
    try {
      const response = await fetch(`${apiBase}/api/decisions`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ name: result.name, category: 'Uncategorised', price: result.price, currency: result.currency, status, data: result }) });
      if (!response.ok) throw new Error('Could not save this decision.');
      setSaveNotice('Saved to your local decision journal.');
      await refreshWorkspace();
    } catch (cause) { setSaveNotice(cause instanceof Error ? cause.message : 'Could not save this decision.'); }
  }

  async function updateDecisionStatus(decision: DecisionRecord, status: string) {
    try {
      const response = await fetch(`${apiBase}/api/decisions`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ ...decision, status, final_decision: status, data: decision.data }),
      });
      if (!response.ok) throw new Error(await readError(response));
      await refreshWorkspace();
    } catch (cause) {
      setWorkspaceError(cause instanceof Error ? cause.message : 'Decision status could not be updated.');
    }
  }

  async function removeDecision(decision: DecisionRecord) {
    if (!window.confirm(`Remove the journal entry for "${decision.name}"?`)) return;
    try {
      const response = await fetch(`${apiBase}/api/decisions/${encodeURIComponent(decision.id)}`, { method: 'DELETE' });
      if (!response.ok) throw new Error(await readError(response));
      await refreshWorkspace();
    } catch (cause) {
      setWorkspaceError(cause instanceof Error ? cause.message : 'Journal entry could not be removed.');
    }
  }

  async function searchProducts(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (!searchQuery.trim()) return;
    setSearchStatus('Searching…'); setSearchResults([]);
    try {
      const response = await fetch(`${apiBase}/api/products/search`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ query: searchQuery.trim(), category: searchCategory }) });
      const data = await response.json();
      if (!response.ok) throw new Error(data.detail ?? 'Search failed.');
      setSearchResults(data.offers ?? []);
      setSearchStatus(data.message || `${data.offers?.length ?? 0} verified offer(s) returned · ${data.status}.`);
    } catch (cause) { setSearchStatus(cause instanceof Error ? cause.message : 'Search failed.'); }
  }

  async function addOfferToWatchlist(offer: typeof searchResults[number]) {
    try {
      const response = await fetch(`${apiBase}/api/watchlist`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ name: offer.name, variant: offer.variant, retailer: offer.retailer, price: offer.price, currency: offer.currency, url: offer.url, source: offer.source, status: offer.status, observation: offer }) });
      if (!response.ok) throw new Error(await readError(response));
      setSearchStatus('Saved to your local watchlist.');
      await refreshWorkspace();
    } catch (cause) { setSearchStatus(cause instanceof Error ? cause.message : 'Offer could not be tracked.'); }
  }

  async function removeTrackedItem(id: string) {
    if (!window.confirm('Remove this product from your watchlist and price history?')) return;
    try {
      const response = await fetch(`${apiBase}/api/watchlist/${encodeURIComponent(id)}`, { method: 'DELETE' });
      if (!response.ok) throw new Error(await readError(response));
      await refreshWorkspace();
    } catch (cause) { setSearchStatus(cause instanceof Error ? cause.message : 'Product could not be removed.'); }
  }

  async function refreshTrackedItem(id: string) {
    try {
      const response = await fetch(`${apiBase}/api/watchlist/${encodeURIComponent(id)}/refresh`, { method: 'POST' });
      if (!response.ok) throw new Error(await readError(response));
      const data = await response.json();
      setSearchStatus(data.result?.message || data.notification || 'Watchlist refreshed.');
      await refreshWorkspace();
    } catch (cause) { setSearchStatus(cause instanceof Error ? cause.message : 'Watchlist refresh failed.'); }
  }

  async function editTargetPrice(item: typeof watchlistItems[number]) {
    const next = window.prompt('Target price in INR (leave blank to remove):', item.target_price === null ? '' : String(item.target_price));
    if (next === null) return;
    const target = next.trim() === '' ? null : Number(next);
    if (target !== null && (!Number.isFinite(target) || target < 0)) {
      setSearchStatus('Enter a valid non-negative target price.');
      return;
    }
    try {
      const response = await fetch(`${apiBase}/api/watchlist/${encodeURIComponent(item.id)}`, { method: 'PATCH', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ target_price: target }) });
      if (!response.ok) throw new Error(await readError(response));
      await refreshWorkspace();
    } catch (cause) { setSearchStatus(cause instanceof Error ? cause.message : 'Target price could not be updated.'); }
  }

  async function showHistory(id: string) {
    try {
      const response = await fetch(`${apiBase}/api/watchlist/${encodeURIComponent(id)}/history`);
      if (!response.ok) throw new Error(await readError(response));
      const data = await response.json();
      setHistoryByItem(previous => ({ ...previous, [id]: data.observations ?? [] }));
    } catch (cause) { setSearchStatus(cause instanceof Error ? cause.message : 'Price history could not be loaded.'); }
  }

  async function exportJournal() {
    try {
      const response = await fetch(`${apiBase}/api/data/export`, { headers: { Accept: 'application/json' } });
      if (!response.ok) throw new Error(await readError(response));
      const blob = new Blob([JSON.stringify(await response.json(), null, 2)], { type: 'application/json' });
      const url = URL.createObjectURL(blob);
      const anchor = document.createElement('a');
      anchor.href = url;
      anchor.download = 'beforeyoubuy-data.json';
      anchor.click();
      window.setTimeout(() => URL.revokeObjectURL(url), 1000);
    } catch (cause) { setWorkspaceError(cause instanceof Error ? cause.message : 'Journal export failed.'); }
  }

  async function deleteLocalData() {
    if (!window.confirm('Delete all local decisions, watchlist entries, and price history? This cannot be undone.')) return;
    try {
      const response = await fetch(`${apiBase}/api/data?confirm=true`, { method: 'DELETE' });
      if (!response.ok) throw new Error(await readError(response));
      setSearchStatus('Local data deleted.');
      await refreshWorkspace();
    } catch (cause) { setWorkspaceError(cause instanceof Error ? cause.message : 'Local data could not be deleted.'); }
  }

  async function sendChat(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (chatBusyRef.current) return;
    const message = chatText.trim();
    if (!message) return;
    setChatTurns(previous => [...previous, { role: 'user', text: message }]);
    setChatText('');
    await sendChatMessage(message);
  }

  async function sendChatMessage(message: string) {
    if (chatBusyRef.current) return;
    const requestGeneration = ++decisionGeneration.current;
    const requestSessionId = sessionId;
    chatBusyRef.current = true;
    setChatBusy(true);
    setError('');
    try {
      const response = await fetch(`${apiBase}/api/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', Accept: 'application/json' },
        body: JSON.stringify({
          message,
          session_id: requestSessionId,
          ...(selectedModel ? { model: selectedModel } : {}),
          ...(assessmentContext && assessmentContextPending ? { assessment_context: assessmentContext } : {}),
        }),
      });
      if (!response.ok) throw new Error(await readError(response));
      const data: unknown = await response.json();
      assertChatApiResponse(data);
      if (requestGeneration !== decisionGeneration.current) return;
      setSessionId(data.session_id);
      setAssessmentContextPending(false);
      setChatTiming(data.processing_ms);
      setChatProvider(data.provider);
      setActiveModel(data.model ?? '');
      setChatFallbackReason(data.provider === 'deterministic_fallback' ? (data.fallback_reason ?? '') : '');
      setChatTools(data.tools_used);
      setFollowupSources(data.knowledge_sources);
      setSessionFacts(data.session_facts);
      setChatDecision(data.decision_state);
      setChatSentiment(data.sentiment_analysis);
      setRetryMessage('');
      if (data.assessment_update) setResult(data.assessment_update);
      setChatTurns(previous => [...previous, {
        role: 'assistant',
        text: data.reply.trim(),
        provider: data.provider,
        model: data.model,
        fallbackReason: data.fallback_reason,
        fallbackOccurred: data.fallback_occurred,
        ragUsed: data.rag.used,
        evidenceCount: data.rag.evidence_count,
      }]);
    } catch (cause) {
      if (requestGeneration === decisionGeneration.current) {
        setError(cause instanceof Error ? cause.message : 'The conversation reply failed.');
        setRetryMessage(message);
      }
    } finally {
      chatBusyRef.current = false;
      setChatBusy(false);
    }
  }

  async function retryChat() {
    if (!retryMessage || chatBusyRef.current) return;
    const message = retryMessage;
    setRetryMessage('');
    await sendChatMessage(message);
  }

  return (
    <main className="app">
      <header className="topbar">
        <a className="brand" href="#" onClick={event => { event.preventDefault(); reset(); }}>
          <span className="brand-mark">B</span>
          <span>BeforeYouBuy <b>AI</b></span>
        </a>
        <div className="top-note"><span className="online-dot" /> Reflective decision support · Your choice stays yours</div>
        <button className="button button-quiet" type="button" onClick={reset}>New assessment</button>
      </header>

      <nav className="app-nav" aria-label="Primary navigation">
        {[['overview', 'Overview', 'overview'], ['assess', 'Assess a purchase', 'assessment-form'], ['chat', 'Talk it through', 'chat-title'], ['watchlist', 'Watchlist', 'watchlist-panel'], ['journal', 'Decision journal', 'journal']].map(([id, label, target]) => (
          <button key={id} type="button" className={activeSection === id ? 'nav-active' : ''} onClick={() => { setActiveSection(id); document.getElementById(target)?.scrollIntoView({ behavior: 'smooth' }); }}>{label}</button>
        ))}
      </nav>

      <section id="overview" className="overview-panel" aria-labelledby="overview-title">
        <div className="section-heading"><span className="step-mark">00</span><div><h2 id="overview-title">Your purchase picture</h2><p>Only saved records appear here. Nothing is fabricated for a new account.</p></div></div>
        {workspaceLoading ? <p role="status">Loading your saved purchase picture…</p> : workspaceError ? <div className="error-banner" role="alert">{workspaceError} <button type="button" className="button button-secondary" onClick={() => void refreshWorkspace()}>Retry</button></div> : analytics?.decision_count ? <div className="metric-grid">
          <div><span>Planned</span><strong>{analyticsCurrencyTotals.map(([code, totals]) => <small className="metric-currency" key={`planned-${code}`}>{currency(totals.planned_spending, code)}</small>)}</strong></div><div><span>Confirmed spend</span><strong>{analyticsCurrencyTotals.map(([code, totals]) => <small className="metric-currency" key={`spent-${code}`}>{currency(totals.confirmed_spending, code)}</small>)}</strong></div><div><span>Confirmed savings</span><strong>{analyticsCurrencyTotals.map(([code, totals]) => <small className="metric-currency" key={`saved-${code}`}>{currency(totals.confirmed_savings, code)}</small>)}</strong></div><div><span>Decisions</span><strong>{analytics.decision_count}</strong></div><div><span>Pressure avg.</span><strong>{analytics.average_purchase_pressure ?? '—'}<small>/100</small></strong></div><div><span>Regret-risk avg.</span><strong>{analytics.average_regret_risk ?? '—'}<small>/100</small></strong></div>
        </div> : <div className="empty-state"><strong>Your dashboard is ready when you are.</strong><p>Assess a purchase and save it to build a private journal. Planned, postponed, and skipped purchases stay distinct from confirmed spending.</p><button className="button button-primary" type="button" onClick={() => { setActiveSection('assess'); document.getElementById('assessment-form')?.scrollIntoView({ behavior: 'smooth' }); }}>Assess your first purchase</button></div>}
        {!workspaceLoading && !workspaceError && analytics && analytics.decision_count > 0 && <p className="small dashboard-note">{analytics.awaiting_decision} awaiting a final decision · {analytics.tracked_products} tracked product{analytics.tracked_products === 1 ? '' : 's'}. Confirmed savings are user-entered only.</p>}
      </section>

      <section className="intro">
        <p className="eyebrow">A CALM MOMENT BEFORE A PURCHASE</p>
        <h1>Think it through,<br /><em>on your terms.</em></h1>
        <p className="intro-copy">Share only what you want to. See transparent calculations, the assumptions behind them, and relevant educational sources — without a buy-or-don't-buy verdict.</p>
      </section>

      {error && <div className="error-banner" role="alert">{error}</div>}

      <div className="layout">
        <section id="assessment-form" className="card form-card">
          <div className="section-heading">
            <span className="step-mark">01</span>
            <div><h2>What are you considering?</h2><p>Only the item and price are required.</p></div>
          </div>
          <form onSubmit={runAssessment}>
            <div className="field-grid">
              <label>Item name
                <input required maxLength={200} value={name} onChange={event => setName(event.target.value)} placeholder="e.g. Headphones" />
              </label>
              <div className="field-row">
                <label>Price
                  <input required type="number" min="0.01" max="100000000" step="0.01" value={price} onChange={event => setPrice(event.target.value)} placeholder="8000" />
                </label>
                <label>Currency
                  <select value={currencyCode} onChange={event => setCurrencyCode(event.target.value)}>
                    <option value="INR">INR ₹</option><option value="USD">USD $</option><option value="GBP">GBP £</option><option value="EUR">EUR €</option>
                  </select>
                </label>
              </div>
              <label className="full">What is on your mind? <span className="optional">Optional</span>
                <textarea maxLength={2000} rows={3} value={scenario} onChange={event => setScenario(event.target.value)} placeholder="Describe the reason, timing, or anything you are weighing." />
              </label>
              <div className="field-row full">
                <label>Purpose
                  <select value={need} onChange={event => setNeed(event.target.value)}>
                    <option value="">Not specified</option>
                    <option>Essential need</option><option>Solves a real problem</option><option>Useful improvement</option><option>Mostly a want</option>
                  </select>
                </label>
                <label>Current alternative
                  <select value={current} onChange={event => setCurrent(event.target.value)}>
                    <option value="">Not specified</option>
                    <option>Mine is broken or unusable</option><option>It works, but has limitations</option><option>I have something that works</option>
                  </select>
                </label>
              </div>
              <label className="full">Timing context
                <select value={urgency} onChange={event => setUrgency(event.target.value)}>
                  <option value="">Not specified</option>
                  <option>No deadline</option><option>A genuine deadline</option><option>A sale or countdown</option><option>Social pressure or FOMO</option>
                </select>
              </label>
            </div>

            <details className="optional-details">
              <summary>Optional usage and financial figures</summary>
              <p className="hint">Enter only figures you know and choose to share. Blank fields remain unknown.</p>
              <div className="field-grid">
                <div className="field-row full">
                  <label>Planned uses per week
                    <input type="number" min="0.01" max="500" step="0.01" value={usesPerWeek} onChange={event => setUsesPerWeek(event.target.value)} placeholder="e.g. 5" />
                  </label>
                  <label>Expected lifespan (years)
                    <input type="number" min="0.01" max="100" step="0.01" value={lifespan} onChange={event => setLifespan(event.target.value)} placeholder="e.g. 2" />
                  </label>
                </div>
                <label>Budget comparison figure ({currencyCode})
                  <input type="number" min="0" max="100000000" step="0.01" value={budget} onChange={event => setBudget(event.target.value)} placeholder="Optional" />
                </label>
                <label>Monthly surplus you stated ({currencyCode})
                  <input type="number" min="0" max="100000000" step="0.01" value={monthlySurplus} onChange={event => setMonthlySurplus(event.target.value)} placeholder="Optional" />
                </label>
                <label>Allocated savings ({currencyCode})
                  <input type="number" min="0" max="100000000" step="0.01" value={savings} onChange={event => setSavings(event.target.value)} placeholder="Optional" />
                </label>
                <label>Recurring monthly cost ({currencyCode})
                  <input type="number" min="0" max="10000000" step="0.01" value={recurring} onChange={event => setRecurring(event.target.value)} placeholder="Optional" />
                </label>
              </div>
            </details>
            <button className="button button-primary" type="submit" disabled={busy}>
              {busy ? 'Working…' : 'Reflect on this purchase'}
            </button>
          </form>
        </section>

        <aside className="card privacy-card">
          <div className="privacy-icon">◎</div>
          <h2>Private by design</h2>
          <p>Assessment requests are not stored by the backend. The optional Ollama chat keeps only a short, temporary session in server memory.</p>
          <div className="privacy-rule" />
          <p className="small">No bank connections. No inferred income or savings. No product research claims.</p>
        </aside>
      </div>

      {result && (
        <section className="results" aria-live="polite">
          <div className="result-heading">
            <div><p className="eyebrow">YOUR REFLECTION</p><h2>{result.name}</h2><p className="result-summary"><b>{result.decision}{result.provisional ? ' · Provisional' : ''}.</b> {result.decision_explanation}</p><p className="small">Evidence completeness: {result.confidence.level}. {result.confidence.basis}</p><p>{result.summary}</p></div>
            <button className="button button-secondary" type="button" onClick={downloadPdf} disabled={busy}>{busy ? 'Preparing…' : 'Download PDF review'}</button>
          </div>

          <div className="result-grid">
            <article className="card result-card">
              <h3>Purchase pressure</h3>
              <div className="score-row"><strong>{result.purchase_pressure.score}<small>/100</small></strong><span>{result.purchase_pressure.band}</span></div>
              <div className="meter"><span style={{ width: `${result.purchase_pressure.score}%` }} /></div>
              <p>{result.purchase_pressure.explanation}</p>
              <ul>{result.purchase_pressure.factors.map((factor, index) => <li key={`${factor.factor}-${index}`}>{factor.factor} <b>+{factor.points}</b></li>)}</ul>
              <p className="caveat">{result.purchase_pressure.limitations}</p>
            </article>
            <article className="card result-card">
              <h3>Regret-risk reflection</h3>
              <div className="score-row"><strong>{result.regret_risk.score}<small>/100</small></strong><span>{result.regret_risk.band}</span></div>
              <div className="meter"><span style={{ width: `${result.regret_risk.score}%` }} /></div>
              <p>{result.regret_risk.explanation}</p>
              <ul>{result.regret_risk.factors.map((factor, index) => <li key={`${factor.factor}-${index}`}>{factor.factor} <b>+{factor.points}</b></li>)}</ul>
              <p className="caveat">{result.regret_risk.limitations}</p>
            </article>
            <article className="card result-card">
              <h3>What the text indicates</h3>
              <p className="sentiment-label">{result.sentiment.label} sentiment <span>· {result.sentiment.confidence} confidence</span></p>
              <p>Emotional cues explicitly mentioned: {result.sentiment.emotional_cues.join(', ') || 'none detected'}</p>
              {result.sentiment.evidence.length > 0 && <p className="small">Keyword evidence: {result.sentiment.evidence.join(', ')}</p>}
              <p className="caveat">{result.sentiment.limitation}</p>
            </article>
            <article className="card result-card">
              <h3>Cost and figures entered</h3>
              <p>Price: <b>{currency(result.price, result.currency)}</b></p>
              {result.estimated_cost_per_use === null
                ? <p>Cost per use not calculated — enter both uses per week and expected lifespan.</p>
                : <p>Cost per use: <b>{currency(result.estimated_cost_per_use, result.currency)}</b> · {result.estimated_total_uses} estimated uses (52 weeks/year)</p>}
              <p>Total compared (price + 12-month recurring cost): {currency(result.affordability.budget_comparison_total, result.currency)} · Recurring amount {result.recurring_monthly_provided ? 'was supplied' : 'not supplied (treated as zero)'}</p>
              {result.affordability.budget_provided && <p>Entered budget: {currency(Number(budget), result.currency)} · {result.budget_exceeded_by && result.budget_exceeded_by > 0 ? `exceeds by ${currency(result.budget_exceeded_by, result.currency)}` : 'not above the entered figure'}</p>}
              {!result.affordability.budget_provided && <p>No budget comparison figure was supplied.</p>}
              {result.affordability.monthly_surplus_provided && <p>Price equals {result.affordability.months_of_stated_surplus ?? '—'} month(s) of your stated surplus figure; it is not independently verified.</p>}
              <p className="small">{result.affordability.note}</p>
            </article>
            <article className="card result-card factors-card">
              <h3>Factors and missing information</h3>
              <p className="outcome">{result.recommendation}</p>
              <ul>{result.factors.map((factor, index) => <li key={`${factor}-${index}`}>{factor} <b>({result.factor_weights[index] > 0 ? '+' : ''}{result.factor_weights[index]})</b></li>)}</ul>
              {result.missing_inputs.length > 0 && <><h4>Not provided</h4><ul>{result.missing_inputs.map(item => <li key={item}>{item}</li>)}</ul></>}
              <p className="small">Processing time: {result.processing_ms} ms · Tools used: {result.tools_used.join(', ')}</p>
            </article>
          </div>

          <div className="save-row">
            <button className="button button-primary" type="button" onClick={() => void saveCurrentDecision()}>Save to decision journal</button>
            <button className="button button-secondary" type="button" onClick={() => void saveCurrentDecision('postponed')}>Save as postponed</button>
            {saveNotice && <span className="small" role="status">{saveNotice}</span>}
          </div>

          <section className="card source-card">
            <h3>Relevant educational notes</h3>
            {result.knowledge_sources.length === 0 ? <p>No matching note was retrieved for this information.</p> : result.knowledge_sources.map(source => (
              <article className="source" key={source.url + source.topic}>
                <div><h4>{source.topic}</h4><p>{source.points.slice(0, 3).join(' ')}</p></div>
                <a href={source.url} target="_blank" rel="noreferrer">{source.title} ↗</a>
              </article>
            ))}
          </section>

          <details className="card assumptions-card">
            <summary>Calculation assumptions and limitations</summary>
            <ul>{result.assumptions.map((assumption, index) => <li key={`assumption-${index}`}>{assumption}</li>)}</ul>
            <ul>{result.limitations.map((limitation, index) => <li key={`limitation-${index}`}>{limitation}</li>)}</ul>
          </details>
          <p className="disclaimer">{result.disclaimer} {result.regret_risk.limitations}</p>
        </section>
      )}

      <section className="card chat-card" aria-labelledby="chat-title">
        <div className="chat-header">
          <div className="section-heading"><span className="step-mark">02</span><div><h2 id="chat-title">Talk it through</h2><p>Chat naturally; an assessment is optional.</p></div></div>
          <button className="button button-quiet" type="button" onClick={newConversation} disabled={chatBusy}>New conversation</button>
        </div>
        <div className="chat-controls">
          <label className="model-picker">Local response model
            <select aria-label="Local response model" value={selectedModel} onChange={event => setSelectedModel(event.target.value)} disabled={modelStatus === 'checking'}>
              <option value="">Automatic / configured default</option>
              {modelChoices.map(model => <option key={model} value={model}>{model}</option>)}
            </select>
          </label>
          <span className={`model-status ${modelStatus}`} role="status">
            <i />{modelStatus === 'checking'
              ? 'Checking local models…'
              : modelStatus === 'connected'
                ? `${modelChoices.length} local model${modelChoices.length === 1 ? '' : 's'} available`
                : modelStatus === 'disabled'
                  ? 'Ollama disabled'
                  : modelStatus === 'unknown'
                    ? 'Ollama status unknown'
                    : 'No Ollama model available · fallback ready'}
          </span>
        </div>
        <div className="chat-log" ref={chatLogRef} aria-live="polite" aria-relevant="additions text">
          {chatTurns.length === 0 && !chatBusy && <p className="chat-empty">Share a thought, ask a question, or add context in your own words. You can start chatting without filling in the assessment.</p>}
          {chatTurns.map((turn, index) => (
            <div className={`chat-bubble ${turn.role}`} key={`${turn.role}-${index}`}>
              <p>{turn.text}</p>
              {turn.role === 'assistant' && turn.provider && (
                <small className="chat-reply-provenance" role="status">
                  {turn.provider === 'ollama'
                    ? `Qwen response · Ollama${turn.model ? ` · ${turn.model}` : ''}${turn.ragUsed ? ` · RAG (${turn.evidenceCount} source${turn.evidenceCount === 1 ? '' : 's'})` : ' · No relevant evidence retrieved'}`
                    : turn.provider === 'gemini'
                      ? `Gemini response${turn.model ? ` · ${turn.model}` : ''}${turn.fallbackOccurred ? ' · Ollama fallback' : ''}${turn.ragUsed ? ` · RAG (${turn.evidenceCount} source${turn.evidenceCount === 1 ? '' : 's'})` : ' · No relevant evidence retrieved'}`
                    : turn.fallbackReason?.toLowerCase().includes('disabled')
                      ? 'Ollama disabled'
                      : `Deterministic fallback · ${turn.fallbackReason?.replace('; deterministic fallback is active.', '') || 'Ollama response unavailable'}`}
                </small>
              )}
            </div>
          ))}
          {chatBusy && (
            <div className="chat-bubble assistant thinking" role="status" aria-live="polite">
              <span>{selectedModel ? `${selectedModel} is thinking` : 'Thinking through your message'}</span>
              <span className="thinking-dots" aria-hidden="true"><i /><i /><i /></span>
              <small>Local inference can take a little while. Your message is being processed.</small>
            </div>
          )}
        </div>
        {chatDecision && (
          <section className="chat-decision" aria-live="polite" aria-label="Updated purchase reflection">
            <div className="chat-decision-heading">
              <div><p className="eyebrow">DECISION SUMMARY FROM THIS CONVERSATION</p><h3>{chatDecision.recommendation}</h3></div>
              {chatDecision.provisional && <span className="provisional-label">Provisional</span>}
            </div>
            <p>{chatDecision.recommendation_explanation}</p>
            <details className="chat-assessment-details">
              <summary>Show structured assessment details</summary>
            <p className="small"><b>Evidence completeness: {chatDecision.confidence.level}.</b> {chatDecision.confidence.basis}</p>
            {(chatDecision.financial_context.product_price !== null
              || chatDecision.financial_context.stated_budget !== null
              || chatDecision.financial_context.user_affordability !== null
              || chatDecision.financial_context.savings_adequacy !== null
              || chatDecision.financial_context.essential_expenses_affected !== null
              || chatDecision.financial_context.savings_affected !== null
              || chatDecision.financial_context.upcoming_expenses !== null) && (
              <p className="small"><b>Financial details stated:</b> {[
                chatDecision.financial_context.product_price !== null
                  ? `Product price: ${currency(chatDecision.financial_context.product_price, chatDecision.financial_context.currency ?? 'INR')}`
                  : null,
                chatDecision.financial_context.stated_budget !== null
                  ? `Budget: ${currency(chatDecision.financial_context.stated_budget, chatDecision.financial_context.currency ?? 'INR')}`
                  : null,
                chatDecision.financial_context.user_affordability === 'reported_affordable'
                  ? 'You report that it is affordable'
                  : chatDecision.financial_context.user_affordability === 'reported_not_affordable'
                    ? 'You report that it is not affordable'
                    : null,
                chatDecision.financial_context.savings_adequacy === 'reported_sufficient'
                  ? 'You report sufficient savings'
                  : chatDecision.financial_context.savings_adequacy === 'reported_insufficient'
                    ? 'You report insufficient savings'
                    : null,
                chatDecision.financial_context.essential_expenses_affected === false
                  ? 'You report no effect on rent or bills'
                  : chatDecision.financial_context.essential_expenses_affected === true
                    ? 'You report an effect on rent or bills'
                    : null,
                chatDecision.financial_context.savings_affected === false
                  ? 'You report no effect on savings'
                  : chatDecision.financial_context.savings_affected === true
                    ? 'You report an effect on savings'
                    : null,
                chatDecision.financial_context.upcoming_expenses === true
                  ? 'You report upcoming expenses'
                  : null,
              ].filter((item): item is string => item !== null).join(' · ')}
                {chatDecision.financial_context.stated_budget !== null && chatDecision.financial_context.product_price === null
                  ? ' A budget is not the price of a specific product.'
                  : ''}
              </p>
            )}
            {(chatDecision.replacement_context.purchase_timing !== null
              || chatDecision.replacement_context.waiting_period !== null
              || chatDecision.replacement_context.usable_alternative_available !== null) && (
              <p className="small"><b>Timing and alternatives:</b> {[
                chatDecision.replacement_context.purchase_timing === 'urgent'
                  ? 'You described a time-sensitive replacement need'
                  : chatDecision.replacement_context.purchase_timing === 'not_urgent'
                    ? 'You said the purchase is not urgent'
                    : null,
                chatDecision.replacement_context.waiting_period
                  ? `You can wait ${chatDecision.replacement_context.waiting_period}`
                  : null,
                chatDecision.replacement_context.usable_alternative_available === false
                  ? 'No usable alternative was reported'
                  : chatDecision.replacement_context.usable_alternative_available === true
                    ? 'A usable alternative was reported'
                    : null,
              ].filter((item): item is string => item !== null).join(' · ')}</p>
            )}
            {chatDecision.changed_facts.length > 0 && (
              <p className="small"><b>Changed this turn:</b> {chatDecision.changed_facts.map(change => `${change.field.split('_').join(' ')}${change.previous === undefined ? '' : ` (${String(change.previous)} →)`}: ${String(change.current)}`).join(' · ')}</p>
            )}
            <div className="chat-score-row">
              <span>Purchase pressure <b>{chatDecision.purchase_pressure.score}/100</b></span>
              <span>Regret-risk reflection <b>{chatDecision.regret_risk.score}/100</b></span>
            </div>
            {chatDecision.estimated_total_uses !== null && chatDecision.estimated_cost_per_use !== null && (
              <p className="small">Estimated uses: {chatDecision.estimated_total_uses.toLocaleString()} · Cost per use: {chatDecision.estimated_cost_per_use.toFixed(2)} ({chatDecision.cost_per_use_basis.uses_per_week_source ?? 'usage input'}, {chatDecision.cost_per_use_basis.weeks_per_year} weeks/year, {chatDecision.cost_per_use_basis.lifespan_source ?? 'lifespan input'})</p>
            )}
            {(chatDecision.supporting_factors.length > 0 || chatDecision.factors_to_consider.length > 0) && (
              <div className="chat-decision-factors">
                {chatDecision.supporting_factors.length > 0 && <p><b>Factors supporting the stated purpose:</b> {chatDecision.supporting_factors.join(' · ')}</p>}
                {chatDecision.factors_to_consider.length > 0 && <p><b>Factors to weigh:</b> {chatDecision.factors_to_consider.join(' · ')}</p>}
              </div>
            )}
            {chatDecision.missing_inputs.length > 0 && <p className="small"><b>Still unknown:</b> {chatDecision.missing_inputs.join(' ')}</p>}
            <p className="caveat">{chatDecision.limitations}</p>
            </details>
          </section>
        )}
        <form className="chat-form" onSubmit={sendChat}>
          <textarea
            id="chat-input"
            maxLength={2000}
            rows={2}
            value={chatText}
            disabled={chatBusy}
            onChange={event => setChatText(event.target.value)}
            onKeyDown={handleChatKeyDown}
            placeholder="Write a message… (Enter to send, Shift+Enter for a new line)"
            aria-label="Chat message"
          />
          <button className="button button-primary" type="submit" disabled={chatBusy || !chatText.trim()}>{chatBusy ? 'Thinking…' : 'Send'}</button>
        </form>
        {retryMessage && <div className="chat-retry" role="alert"><span>That message did not get a reply.</span><button className="button button-secondary" type="button" onClick={retryChat} disabled={chatBusy}>Retry</button></div>}
        <p className="small chat-meta">
          {chatTiming !== null
            ? `Reply processing: ${chatTiming} ms · Provider: ${chatProvider}${activeModel ? ` · Model: ${activeModel}` : ''} · Tools used: ${chatTools.join(', ') || 'none'}`
            : providerStatusText}
        </p>
          {chatFallbackReason && <p className="small chat-fallback-note" role="status">{chatFallbackReason} A deterministic response was used instead.</p>}
        {followupSources.length > 0 && <div className="chat-sources"><b>Sources for this reply:</b> {followupSources.map((source, index) => <span key={`${source.url}-${index}`}>{index > 0 && ' · '}<a href={source.url} target="_blank" rel="noreferrer">{source.title}</a></span>)}</div>}
        {chatSentiment && <p className="small chat-sentiment" role="status">
          <b>Message analysis:</b> {chatSentiment.polarity.label}
          {chatSentiment.emotions.length > 0 && ` · emotions: ${chatSentiment.emotions.map(item => item.label).join(', ')}`}
          {(() => {
            const explicit = Object.entries(chatSentiment.purchase_signals)
              .filter(([, signal]) => signal.value === true)
              .map(([label]) => label.split('_').join(' '));
            return explicit.length > 0 ? ` · signals: ${explicit.join(', ')}` : '';
          })()}
          {chatSentiment.insufficient_evidence ? ' · insufficient sentiment evidence' : ''}
        </p>}
        {Object.entries(sessionFacts).some(([key]) => key !== 'fact_evidence') && <p className="small">Facts recognized from your messages: {Object.entries(sessionFacts).filter(([key]) => key !== 'fact_evidence').map(([key, value]) => `${key.split('_').join(' ')}: ${String(value)}`).join(' · ')}</p>}
        <p className="small chat-privacy">Conversation context is temporary and held in backend memory; it clears on restart or when you start a new conversation.</p>
      </section>

      <section id="watchlist-panel" className="card workspace-card" aria-labelledby="watchlist-title">
        <div className="section-heading"><span className="step-mark">04</span><div><h2 id="watchlist-title">Search, compare & track</h2><p>India-first search is provider-backed only when credentials are configured; cached results are labeled.</p></div></div>
        <form className="search-form" onSubmit={searchProducts}>
          <label>Product or model<input value={searchQuery} onChange={event => setSearchQuery(event.target.value)} placeholder="e.g. Sony WH-1000XM5" /></label>
          <label>Category <span className="optional">Optional</span><input value={searchCategory} onChange={event => setSearchCategory(event.target.value)} placeholder="Headphones" /></label>
          <button className="button button-primary" type="submit">Search India offers</button>
        </form>
        {searchStatus && <p className="small search-status" role="status">{searchStatus}</p>}
        {searchResults.length > 0 && <div className="offer-list">{searchResults.map((offer, index) => <div className="offer-row" key={`${offer.offer_id}-${offer.url}-${index}`}><div><strong>{offer.name}</strong><span>{offer.retailer || offer.source} · {offer.variant || 'Variant not supplied'} · {offer.status}</span></div><div className="offer-actions"><b>{currency(offer.price, offer.currency)}</b><a href={offer.url} target="_blank" rel="noreferrer">Open source ↗</a><button className="button button-secondary" type="button" onClick={() => void addOfferToWatchlist(offer)}>Track</button></div></div>)}</div>}
        {watchlistItems.length === 0 ? <div className="empty-state"><strong>No products are being tracked.</strong><p>Add a verified offer above or use a manual watchlist entry through the API. Automated notifications are not configured.</p></div> : watchlistItems.map(item => <div className="watch-row" key={item.id}><div><strong>{item.name}</strong><span>{item.retailer || 'Manual'} · {item.variant || 'Variant not specified'} · {item.status}</span>{historyByItem[item.id] && (historyByItem[item.id].length < 2 ? <small>Price history is not available yet; one observation is not a trend.</small> : <div className="history-list">{historyByItem[item.id].map(observation => <span key={observation.observed_at}>{new Date(observation.observed_at).toLocaleDateString()} · {currency(observation.price, observation.currency)} · {observation.status}</span>)}</div>)}</div><div className="offer-actions">{item.price === null ? 'No price observed' : currency(item.price, item.currency)} {item.target_price !== null && <small>· target {currency(item.target_price, item.currency)}</small>}<button className="button button-secondary" type="button" onClick={() => void editTargetPrice(item)}>Target</button><button className="button button-secondary" type="button" onClick={() => void refreshTrackedItem(item.id)}>Refresh</button><button className="button button-secondary" type="button" onClick={() => void showHistory(item.id)}>History</button><button className="button button-quiet" type="button" onClick={() => void removeTrackedItem(item.id)}>Remove</button></div></div>)}
      </section>

      <section id="journal" className="card workspace-card" aria-labelledby="journal-title">
        <div className="section-heading"><span className="step-mark">05</span><div><h2 id="journal-title">Decision journal</h2><p>Saved locally in SQLite, with explicit export and deletion controls.</p></div></div>
        <p className="small">The journal keeps your assessment inputs, deterministic scores, recommendation context, and final status together. Personal history is not used to train a model.</p>
        {workspaceLoading ? <p role="status">Loading journal entries…</p> : workspaceError ? <div className="error-banner" role="alert">{workspaceError} <button type="button" className="button button-secondary" onClick={() => void refreshWorkspace()}>Retry</button></div> : decisions.length === 0 ? <div className="empty-state"><strong>No saved decisions yet.</strong><p>Your entries will appear here after you save an assessment.</p></div> : <div className="journal-list">
          {decisions.map(decision => <article className="journal-row" key={decision.id}>
            <div><strong>{decision.name}</strong><span>{decision.category} · {new Date(decision.created_at).toLocaleDateString()} · {currency(decision.price, decision.currency)}</span></div>
            <label>Status
              <select value={decision.status} onChange={event => void updateDecisionStatus(decision, event.target.value)}>
                <option value="awaiting">Awaiting decision</option><option value="planned">Planned</option><option value="postponed">Postponed</option><option value="completed">Completed</option><option value="skipped">Skipped</option><option value="cancelled">Cancelled</option>
              </select>
            </label>
            {decision.confirmed_savings > 0 && <span>Confirmed savings: {currency(decision.confirmed_savings, decision.currency)}</span>}
            <button type="button" className="button button-quiet" onClick={() => void removeDecision(decision)}>Remove</button>
          </article>)}
        </div>}
        <div className="save-row"><button className="button button-secondary" type="button" onClick={exportJournal}>Export local data</button><button className="button button-quiet" type="button" onClick={() => void deleteLocalData()}>Delete all local data</button></div>
      </section>

      <footer>BeforeYouBuy AI · A reflection aid, not financial advice or a prediction of regret.</footer>
    </main>
  );
}

export default App;
