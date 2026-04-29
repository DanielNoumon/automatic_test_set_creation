// ========================================
// Types — mirrors the test set JSON structure
// ========================================

interface QuestionMetadata {
  generated_by: string;
  question_type: string;
  rlm_model: string;
  recursive_model: string;
  chapter?: string;
  subchapters?: string[];
  start_pages?: Record<string, number>;
  end_pages?: Record<string, number>;
}

interface Question {
  id: string;
  type: string;
  question: string;
  expected_behavior: string;
  golden_answer: string;
  golden_context?: string;
  source_documents: string[];
  difficulty?: string | null;
  generation_prompt?: string;
  metadata: QuestionMetadata;
}

interface TestSetMetrics {
  generation_time: number;
  questions_by_type: Record<string, number>;
  total_questions: number;
}

interface TestSetConfig {
  rlm_model: string;
  enabled_types: string[];
  total_questions: number;
}

interface Summary {
  total_questions: number;
  total_requested: number;
  yield_ratio: number;
  llm_calls: number;
  llm_efficiency: number;
  enabled_types: number;
  timings: {
    total_seconds: number;
    parse_seconds: number;
    index_seconds: number;
    avg_per_question_seconds: number;
  };
  quality: {
    avg_context_match: number;
    min_context_match: number;
    answer_grounded_ratio: number;
    duplicate_questions: number;
  };
  context_stats: { min_chars: number; max_chars: number; avg_chars: number };
  answer_stats: { min_chars: number; max_chars: number; avg_chars: number };
  difficulty_distribution: Record<string, number>;
  questions_by_type: Record<string, number>;
  source_documents: Record<string, number>;
  multi_source_questions: number;
}

interface TestSetData {
  metadata: {
    created_at: number;
    config: TestSetConfig;
    metrics: TestSetMetrics;
  };
  summary?: Summary;
  questions: Question[];
}

// ========================================
// State
// ========================================

let testSetData: TestSetData | null = null;

// ========================================
// DOM references
// ========================================

const jsonInput = document.getElementById("json-input") as HTMLInputElement;
const metadataPanel = document.getElementById("metadata-panel")!;
const controlsSection = document.getElementById("controls")!;
const typeFilter = document.getElementById("type-filter") as HTMLSelectElement;
const difficultyFilter = document.getElementById(
  "difficulty-filter"
) as HTMLSelectElement;
const searchInput = document.getElementById(
  "search-input"
) as HTMLInputElement;
const resultCount = document.getElementById("result-count")!;
const tableContainer = document.getElementById("table-container")!;
const tbody = document.getElementById("questions-tbody")!;
const emptyState = document.getElementById("empty-state")!;
const summaryToggle = document.getElementById("summary-toggle")!;
const summaryPanel = document.getElementById("summary-panel")!;
const modalOverlay = document.getElementById("modal-overlay")!;
const modalTitle = document.getElementById("modal-title")!;
const modalBody = document.getElementById("modal-body")!;
const modalClose = document.getElementById("modal-close")!;

// ========================================
// File loading
// ========================================

jsonInput.addEventListener("change", (event: Event) => {
  const target = event.target as HTMLInputElement;
  const file = target.files?.[0];
  if (!file) return;

  const reader = new FileReader();
  reader.onload = (e: ProgressEvent<FileReader>) => {
    try {
      const data = JSON.parse(e.target?.result as string) as TestSetData;
      testSetData = data;
      initViewer();
    } catch (err) {
      alert("Invalid JSON file. Please select a valid test set JSON.");
    }
  };
  reader.readAsText(file);
});

// ========================================
// Initialize viewer after data loads
// ========================================

function initViewer(): void {
  if (!testSetData) return;

  renderMetadata();
  renderSummary();
  populateTypeFilter();
  renderTable(testSetData.questions);

  metadataPanel.classList.remove("hidden");
  if (testSetData.summary) {
    summaryToggle.classList.remove("hidden");
  }
  controlsSection.classList.remove("hidden");
  tableContainer.classList.remove("hidden");
  emptyState.classList.add("hidden");
}

// ========================================
// Metadata panel
// ========================================

function renderMetadata(): void {
  if (!testSetData) return;

  const { config, metrics } = testSetData.metadata;
  const createdDate = new Date(testSetData.metadata.created_at * 1000);

  const items: { label: string; value: string }[] = [
    { label: "Created", value: formatDate(createdDate) },
    { label: "Model", value: config.rlm_model },
    { label: "Total Questions", value: String(metrics.total_questions) },
    {
      label: "Generation Time",
      value: `${metrics.generation_time.toFixed(1)}s`,
    },
    {
      label: "Question Types",
      value: String(config.enabled_types.length),
    },
  ];

  metadataPanel.innerHTML = items
    .map(
      (item) => `
      <div class="meta-item">
        <span class="meta-label">${item.label}</span>
        <span class="meta-value">${item.value}</span>
      </div>`
    )
    .join("");
}

function formatDate(date: Date): string {
  return date.toLocaleDateString("en-GB", {
    day: "2-digit",
    month: "short",
    year: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  });
}

// ========================================
// Summary panel
// ========================================

summaryToggle.addEventListener("click", () => {
  const isOpen = !summaryPanel.classList.contains("hidden");
  summaryPanel.classList.toggle("hidden");
  const chevron = summaryToggle.querySelector(".chevron");
  if (chevron) chevron.textContent = isOpen ? "▸" : "▾";
});

function renderSummary(): void {
  if (!testSetData?.summary) return;
  const s = testSetData.summary;

  const pct = (v: number) => `${(v * 100).toFixed(1)}%`;

  // --- Top-level KPI cards ---
  const kpis = [
    { label: "Questions", value: `${s.total_questions} / ${s.total_requested}` },
    { label: "Yield", value: pct(s.yield_ratio) },
    { label: "LLM Calls", value: String(s.llm_calls) },
    { label: "LLM Efficiency", value: pct(s.llm_efficiency) },
    { label: "Enabled Types", value: String(s.enabled_types) },
    { label: "Total Time", value: `${s.timings.total_seconds}s` },
    { label: "Avg / Question", value: `${s.timings.avg_per_question_seconds}s` },
  ];

  // --- Quality ---
  const qualityRows = [
    { label: "Avg Context Match", value: pct(s.quality.avg_context_match) },
    { label: "Min Context Match", value: pct(s.quality.min_context_match) },
    { label: "Answer Grounded", value: pct(s.quality.answer_grounded_ratio) },
    { label: "Duplicate Questions", value: String(s.quality.duplicate_questions) },
  ];

  // --- Stats ---
  const statsRows = [
    { label: "Context (chars)", value: `${s.context_stats.min_chars} – ${s.context_stats.max_chars} (avg ${s.context_stats.avg_chars})` },
    { label: "Answer (chars)", value: `${s.answer_stats.min_chars} – ${s.answer_stats.max_chars} (avg ${s.answer_stats.avg_chars})` },
    { label: "Multi-source Questions", value: String(s.multi_source_questions) },
    { label: "Parse Time", value: `${s.timings.parse_seconds}s` },
    { label: "Index Time", value: `${s.timings.index_seconds}s` },
  ];

  // --- Build HTML ---
  const kpiHtml = kpis.map(k => `
    <div class="summary-kpi">
      <span class="summary-kpi-value">${k.value}</span>
      <span class="summary-kpi-label">${k.label}</span>
    </div>
  `).join("");

  const tableHtml = (rows: {label: string; value: string}[]) => rows.map(r => `
    <tr><td class="summary-row-label">${r.label}</td><td class="summary-row-value">${r.value}</td></tr>
  `).join("");

  const distHtml = (record: Record<string, number>) =>
    Object.entries(record).map(([k, v]) => `
      <div class="summary-dist-item">
        <span class="summary-dist-label">${formatTypeName(k)}</span>
        <span class="summary-dist-value">${v}</span>
      </div>
    `).join("");

  summaryPanel.innerHTML = `
    <div class="summary-kpis">${kpiHtml}</div>

    <div class="summary-grid">
      <div class="summary-card">
        <h4>Quality</h4>
        <table class="summary-table">${tableHtml(qualityRows)}</table>
      </div>
      <div class="summary-card">
        <h4>Statistics</h4>
        <table class="summary-table">${tableHtml(statsRows)}</table>
      </div>
      <div class="summary-card">
        <h4>Difficulty Distribution</h4>
        <div class="summary-dist">${distHtml(s.difficulty_distribution)}</div>
      </div>
      <div class="summary-card">
        <h4>Questions by Type</h4>
        <div class="summary-dist">${distHtml(s.questions_by_type)}</div>
      </div>
      <div class="summary-card">
        <h4>Source Documents</h4>
        <div class="summary-dist">${distHtml(s.source_documents)}</div>
      </div>
    </div>
  `;
}

// ========================================
// Filters
// ========================================

function populateTypeFilter(): void {
  if (!testSetData) return;

  const types = [...new Set(testSetData.questions.map((q) => q.type))];
  typeFilter.innerHTML = '<option value="all">All types</option>';
  types.forEach((type) => {
    const option = document.createElement("option");
    option.value = type;
    option.textContent = formatTypeName(type);
    typeFilter.appendChild(option);
  });
}

function formatTypeName(type: string): string {
  return type
    .split("_")
    .map((word) => word.charAt(0).toUpperCase() + word.slice(1))
    .join(" ");
}

function getFilteredQuestions(): Question[] {
  if (!testSetData) return [];

  const typeVal = typeFilter.value;
  const diffVal = difficultyFilter.value;
  const searchVal = searchInput.value.toLowerCase().trim();

  return testSetData.questions.filter((q) => {
    if (typeVal !== "all" && q.type !== typeVal) return false;
    if (diffVal !== "all" && (q.difficulty || "N/A") !== diffVal) return false;
    if (searchVal) {
      const haystack =
        `${q.question} ${q.golden_answer} ${q.golden_context || ""} ${q.id}`.toLowerCase();
      if (!haystack.includes(searchVal)) return false;
    }
    return true;
  });
}

function applyFilters(): void {
  const filtered = getFilteredQuestions();
  renderTable(filtered);
}

typeFilter.addEventListener("change", applyFilters);
difficultyFilter.addEventListener("change", applyFilters);
searchInput.addEventListener("input", applyFilters);

// ========================================
// Modal
// ========================================

function openModal(title: string, content: string): void {
  modalTitle.textContent = title;
  modalBody.textContent = content;
  modalOverlay.classList.remove("hidden");
}

function closeModal(): void {
  modalOverlay.classList.add("hidden");
}

modalClose.addEventListener("click", closeModal);
modalOverlay.addEventListener("click", (e: Event) => {
  if (e.target === modalOverlay) closeModal();
});
document.addEventListener("keydown", (e: KeyboardEvent) => {
  if (e.key === "Escape") closeModal();
});

// ========================================
// Render table
// ========================================

function renderTable(questions: Question[]): void {
  const count = questions.length;
  resultCount.textContent = `Showing ${count} question${count !== 1 ? "s" : ""}`;

  if (count === 0) {
    tbody.innerHTML = `<tr><td colspan="9" style="text-align:center; padding:2rem; color:var(--text-muted);">No questions match your filters.</td></tr>`;
    return;
  }

  tbody.innerHTML = questions.map((q, i) => renderRow(q, i)).join("");

  // Attach click handlers for expandable cells
  tbody.querySelectorAll(".cell-text").forEach((el) => {
    el.addEventListener("click", () => {
      const title = el.getAttribute("data-title") || "Details";
      const content = el.getAttribute("data-full") || el.textContent || "";
      openModal(title, content);
    });
  });
}

function renderRow(q: Question, index: number): string {
  const sources = q.source_documents
    .map((doc) => `<span class="source-tag">${escapeHtml(doc)}</span>`)
    .join("");

  const meta = q.metadata;
  const chapter = meta.chapter || "N/A";
  const subchapters = (meta.subchapters || []).join(", ") || "N/A";
  const startPages = formatPages(meta.start_pages);
  const endPages = formatPages(meta.end_pages);

  const metaHtml = [
    `<div class="meta-line"><strong>Chapter:</strong> ${escapeHtml(chapter)}</div>`,
    `<div class="meta-line"><strong>Subchapters:</strong> ${escapeHtml(subchapters)}</div>`,
    `<div class="meta-line"><strong>Start pages:</strong> ${escapeHtml(startPages)}</div>`,
    `<div class="meta-line"><strong>End pages:</strong> ${escapeHtml(endPages)}</div>`,
  ].join("");

  const goldenContext = q.golden_context || "";
  const genPrompt = q.generation_prompt || "";

  return `<tr>
    <td><span class="row-num">${index + 1}</span></td>
    <td><span class="badge badge-type">${formatTypeName(q.type)}</span></td>
    <td>${q.difficulty ? `<span class="badge badge-${q.difficulty}">${q.difficulty}</span>` : '<span style="color:var(--text-muted)">-</span>'}</td>
    <td class="cell-expandable">
      <div class="cell-text" data-title="Question" data-full="${attr(q.question)}">${escapeHtml(q.question)}</div>
    </td>
    <td class="cell-expandable">
      <div class="cell-text" data-title="Golden Context" data-full="${attr(goldenContext)}">${escapeHtml(goldenContext) || '<span style="color:var(--text-muted)">-</span>'}</div>
    </td>
    <td class="cell-expandable">
      <div class="cell-text" data-title="Golden Answer" data-full="${attr(q.golden_answer)}">${escapeHtml(q.golden_answer)}</div>
    </td>
    <td>${sources}</td>
    <td>${metaHtml}</td>
    <td class="cell-expandable">
      <div class="cell-text" data-title="Generation Prompt" data-full="${attr(genPrompt)}">${escapeHtml(truncate(genPrompt, 120)) || '<span style="color:var(--text-muted)">-</span>'}</div>
    </td>
  </tr>`;
}

function formatPages(
  pages: Record<string, number> | undefined
): string {
  if (!pages || Object.keys(pages).length === 0) return "N/A";
  return Object.entries(pages)
    .map(([doc, page]) => `${doc}: ${page}`)
    .join(", ");
}

// ========================================
// Utility
// ========================================

function escapeHtml(text: string): string {
  const div = document.createElement("div");
  div.textContent = text;
  return div.innerHTML;
}

function attr(text: string): string {
  return text
    .replace(/&/g, "&amp;")
    .replace(/"/g, "&quot;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;");
}

function truncate(text: string, maxLen: number): string {
  if (text.length <= maxLen) return text;
  return text.slice(0, maxLen) + "...";
}
