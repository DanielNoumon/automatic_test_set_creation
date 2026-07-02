/* ========================================
   Test Set Viewer v3 — plain JS (no build step)
   Reads the v3 test-set JSON (metadata / summary / questions).
   ======================================== */

let data = null;

// DOM refs
const $ = (id) => document.getElementById(id);
const jsonInput = $("json-input");
const metadataPanel = $("metadata-panel");
const controls = $("controls");
const tierFilter = $("tier-filter");
const intentFilter = $("intent-filter");
const personaFilter = $("persona-filter");
const typeFilter = $("type-filter");
const verifFilter = $("verif-filter");
const searchInput = $("search-input");
const resultCount = $("result-count");
const tableContainer = $("table-container");
const tbody = $("questions-tbody");
const emptyState = $("empty-state");
const summaryToggle = $("summary-toggle");
const summaryPanel = $("summary-panel");
const modalOverlay = $("modal-overlay");
const modalTitle = $("modal-title");
const modalBody = $("modal-body");
const modalClose = $("modal-close");
const flowToggle = $("flow-toggle");
const flowPanel = $("flow-panel");

// ── "How it works" flow panel (rendered from embedded FLOW_MD) ──
if (flowPanel && window.FLOW_MD) {
  flowPanel.innerHTML = renderMarkdown(window.FLOW_MD);
}
flowToggle.addEventListener("click", () => {
  const open = !flowPanel.classList.contains("hidden");
  flowPanel.classList.toggle("hidden");
  const chev = flowToggle.querySelector(".chevron");
  if (chev) chev.textContent = open ? "▸" : "▾";
});

// ── File loading ────────────────────────────────────────
jsonInput.addEventListener("change", (e) => {
  const file = e.target.files && e.target.files[0];
  if (!file) return;
  const reader = new FileReader();
  reader.onload = (ev) => {
    try {
      data = JSON.parse(ev.target.result);
      initViewer();
    } catch (err) {
      alert("Invalid JSON file. Please select a valid v3 test set JSON.");
    }
  };
  reader.readAsText(file);
});

function initViewer() {
  if (!data) return;
  renderMetadata();
  renderSummary();
  populateFilters();
  renderTable(data.questions || []);
  metadataPanel.classList.remove("hidden");
  if (data.summary) summaryToggle.classList.remove("hidden");
  controls.classList.remove("hidden");
  tableContainer.classList.remove("hidden");
  emptyState.classList.add("hidden");
}

// ── Metadata ────────────────────────────────────────────
function renderMetadata() {
  const m = data.metadata || {};
  const cfg = m.config || {};
  const calls = (m.metrics && m.metrics.llm_calls) || {};
  const created = m.created_at ? new Date(m.created_at * 1000) : null;
  const items = [
    { label: "Pipeline", value: m.pipeline_version || "?" },
    { label: "Language", value: (m.language || "?").toUpperCase() },
    { label: "Questions", value: String((data.questions || []).length) },
    { label: "Generator", value: cfg.generator_model || "?" },
    { label: "Solver", value: cfg.solver_model || "?" },
    { label: "Judge", value: cfg.judge_model || "?" },
    { label: "Elapsed", value: m.metrics ? `${m.metrics.elapsed_seconds}s` : "?" },
    {
      label: "LLM calls",
      value: `gen ${calls.generator || 0} · solver ${calls.solver || 0} · judge ${calls.judge || 0}`,
    },
    { label: "Created", value: created ? formatDate(created) : "?" },
  ];
  metadataPanel.innerHTML = items
    .map(
      (i) => `<div class="meta-item">
        <span class="meta-label">${i.label}</span>
        <span class="meta-value">${escapeHtml(i.value)}</span></div>`
    )
    .join("");
}

function formatDate(d) {
  return d.toLocaleDateString("en-GB", {
    day: "2-digit", month: "short", year: "numeric",
    hour: "2-digit", minute: "2-digit",
  });
}

// ── Summary ─────────────────────────────────────────────
summaryToggle.addEventListener("click", () => {
  const open = !summaryPanel.classList.contains("hidden");
  summaryPanel.classList.toggle("hidden");
  const chev = summaryToggle.querySelector(".chevron");
  if (chev) chev.textContent = open ? "▸" : "▾";
});

function renderSummary() {
  const s = data.summary;
  if (!s) return;
  const pct = (v) => `${(v * 100).toFixed(0)}%`;
  const floors = s.tier_floors || {};

  const kpis = [
    { label: "Questions", value: String(s.total_questions) },
    { label: "Grounded", value: pct(s.grounded_ratio || 0) },
    {
      label: "Solver reproduced",
      value: s.solver_ran != null
        ? `${s.solver_reproduced || 0}/${s.solver_ran}`
        : pct(s.solver_reproduced_ratio || 0),
    },
    { label: "Multi-source", value: String(s.multi_source_questions || 0) },
  ];
  const kpiHtml = kpis
    .map(
      (k) => `<div class="summary-kpi">
        <span class="summary-kpi-value">${k.value}</span>
        <span class="summary-kpi-label">${k.label}</span></div>`
    )
    .join("");

  const distCard = (title, rec) => {
    if (!rec || Object.keys(rec).length === 0) return "";
    const max = Math.max(...Object.values(rec), 1);
    const rows = Object.entries(rec)
      .sort((a, b) => b[1] - a[1])
      .map(
        ([k, v]) => `<div class="bar-row">
          <span class="summary-dist-label" style="min-width:110px">${escapeHtml(pretty(k))}</span>
          <span class="bar-track"><span class="bar-fill" style="width:${(v / max) * 100}%"></span></span>
          <span class="summary-dist-value">${v}</span></div>`
      )
      .join("");
    return `<div class="summary-card"><h4>${title}</h4><div class="summary-dist">${rows}</div></div>`;
  };

  const floorHtml = floors.t1
    ? `<div class="summary-card"><h4>Tier Floors</h4><div class="summary-dist">
        <div class="summary-dist-item"><span class="summary-dist-label">T1 baseline</span>
          <span class="${floors.t1.ok ? "floor-ok" : "floor-low"}">${floors.t1.actual} / ${floors.t1.floor} ${floors.t1.ok ? "OK" : "LOW"}</span></div>
        <div class="summary-dist-item"><span class="summary-dist-label">T3+T4 discriminating</span>
          <span class="${floors.t3_t4.ok ? "floor-ok" : "floor-low"}">${floors.t3_t4.actual} / ${floors.t3_t4.floor} ${floors.t3_t4.ok ? "OK" : "LOW"}</span></div>
      </div></div>`
    : "";

  summaryPanel.innerHTML = `
    <div class="summary-kpis">${kpiHtml}</div>
    <div class="summary-grid">
      ${distCard("By Tier", s.by_tier)}
      ${distCard("By Intent", s.by_intent)}
      ${distCard("By Persona", s.by_persona)}
      ${distCard("By Type", s.by_type)}
      ${distCard("Verification Method", s.verification_by_method)}
      ${floorHtml}
    </div>`;
}

// ── Filters ─────────────────────────────────────────────
function populateFilters() {
  const qs = data.questions || [];
  fill(tierFilter, uniq(qs.map((q) => q.locality_tier)), "All tiers");
  fill(intentFilter, uniq(qs.map((q) => q.intent_cluster)), "All intents");
  fill(personaFilter, uniq(qs.map((q) => q.persona)), "All personas");
  fill(typeFilter, uniq(qs.map((q) => q.type)), "All types");
}

function fill(sel, values, allLabel) {
  sel.innerHTML = `<option value="all">${allLabel}</option>`;
  values.forEach((v) => {
    const o = document.createElement("option");
    o.value = v;
    o.textContent = pretty(v);
    sel.appendChild(o);
  });
}

function filtered() {
  const qs = data.questions || [];
  const t = tierFilter.value, i = intentFilter.value, p = personaFilter.value;
  const ty = typeFilter.value, vf = verifFilter.value;
  const search = searchInput.value.toLowerCase().trim();
  return qs.filter((q) => {
    if (t !== "all" && q.locality_tier !== t) return false;
    if (i !== "all" && q.intent_cluster !== i) return false;
    if (p !== "all" && q.persona !== p) return false;
    if (ty !== "all" && q.type !== ty) return false;
    if (vf !== "all") {
      const v = q.verification || {};
      if (vf === "reproduced" && v.solver_reproduced !== true) return false;
      if (vf === "diverged" && v.solver_reproduced !== false) return false;
      if (vf === "deterministic" && v.method !== "deterministic") return false;
      if (vf === "ungrounded" && (q.supporting_spans || []).length > 0) return false;
    }
    if (search) {
      const hay = `${q.question} ${q.golden_answer} ${q.id} ${q.intended_subject || ""}`.toLowerCase();
      if (!hay.includes(search)) return false;
    }
    return true;
  });
}

[tierFilter, intentFilter, personaFilter, typeFilter, verifFilter].forEach((el) =>
  el.addEventListener("change", () => renderTable(filtered()))
);
searchInput.addEventListener("input", () => renderTable(filtered()));

// ── Modal ───────────────────────────────────────────────
function openModal(title, html) {
  modalTitle.textContent = title;
  modalBody.innerHTML = html;
  modalOverlay.classList.remove("hidden");
}
function closeModal() { modalOverlay.classList.add("hidden"); }
modalClose.addEventListener("click", closeModal);
modalOverlay.addEventListener("click", (e) => { if (e.target === modalOverlay) closeModal(); });
document.addEventListener("keydown", (e) => { if (e.key === "Escape") closeModal(); });

// ── Table ───────────────────────────────────────────────
function renderTable(qs) {
  resultCount.textContent = `Showing ${qs.length} question${qs.length !== 1 ? "s" : ""}`;
  if (qs.length === 0) {
    tbody.innerHTML = `<tr><td colspan="8" style="text-align:center;padding:2rem;color:var(--text-muted)">No questions match your filters.</td></tr>`;
    return;
  }
  tbody.innerHTML = qs.map((q, i) => renderRow(q, i)).join("");

  tbody.querySelectorAll("[data-modal]").forEach((el) => {
    el.addEventListener("click", () => {
      const idx = Number(el.getAttribute("data-idx"));
      const kind = el.getAttribute("data-modal");
      openDetail(qs[idx], kind);
    });
  });
}

function renderRow(q, i) {
  const tier = q.locality_tier || "?";
  const variant = q.underspecified_variant
    ? `<div class="variant"><span class="variant-label">underspecified variant</span>${escapeHtml(q.underspecified_variant)}</div>`
    : "";
  const turn2 = q.question_turn_2
    ? `<div class="turn2"><span class="turn2-label">turn 2</span>${escapeHtml(q.question_turn_2)}</div>`
    : "";

  const v = q.verification || {};
  const spans = q.supporting_spans || [];

  return `<tr>
    <td><span class="row-num">${i + 1}</span></td>
    <td><span class="badge badge-tier-${tier}">${tier}</span></td>
    <td><span class="badge badge-type">${pretty(q.type)}</span></td>
    <td><div class="persona-intent">
      <span class="badge badge-soft">${pretty(q.persona)}</span>
      <span class="badge badge-soft">${pretty(q.intent_cluster)}</span>
    </div></td>
    <td class="cell-expandable">
      <div class="cell-text" data-modal="question" data-idx="${i}">${escapeHtml(q.question)}</div>
      ${turn2}${variant}
    </td>
    <td class="cell-expandable">
      <div class="cell-text" data-modal="answer" data-idx="${i}">${escapeHtml(q.golden_answer || "")}</div>
    </td>
    <td>${renderVerif(v, i)}</td>
    <td>${renderSpans(spans, i)}</td>
  </tr>`;
}

function renderVerif(v, i) {
  const method = v.method || "?";
  let status;
  if (v.solver_reproduced === true)
    status = `<span class="verif-status verif-ok">✓ solver reproduced</span>`;
  else if (v.solver_reproduced === false)
    status = `<span class="verif-status verif-diverge">⚠ solver diverged</span>`;
  else
    status = `<span class="verif-status verif-na">— n/a</span>`;
  const ev = v.evidence_docs ? `<span class="verif-sub">${v.evidence_docs} evidence doc(s)</span>` : "";
  const dm = v.deterministic_matches
    ? `<span class="verif-sub">${v.deterministic_matches.length} match(es)</span>` : "";
  return `<div class="verif-box" data-modal="verif" data-idx="${i}">
    <span class="verif-method">${escapeHtml(method)}</span>
    ${status}${ev}${dm}
  </div>`;
}

function renderSpans(spans, i) {
  const n = spans.length;
  const docs = uniq(spans.map((s) => baseName(s.document)));
  const tags = docs.slice(0, 3).map((d) => `<span class="source-tag">${escapeHtml(d)}</span>`).join("");
  const more = docs.length > 3 ? `<span class="verif-sub"> +${docs.length - 3} more</span>` : "";
  return `<div class="spans-box" data-modal="spans" data-idx="${i}">
    <span class="spans-count ${n === 0 ? "zero" : ""}">${n} span${n !== 1 ? "s" : ""} · ${docs.length} doc${docs.length !== 1 ? "s" : ""}</span>
    <div>${tags}${more}</div>
  </div>`;
}

// ── Detail modals ───────────────────────────────────────
function openDetail(q, kind) {
  if (kind === "question") {
    let html = `<div class="modal-plain">${escapeHtml(q.question)}</div>`;
    if (q.question_turn_2) {
      html += `<div class="vdetail" style="margin-top:1rem">
        <div><div class="vrow-label">Turn 1 answer</div><div class="vblock">${escapeHtml(q.golden_answer || "")}</div></div>
        <div><div class="vrow-label">Turn 2 question</div><div class="vblock">${escapeHtml(q.question_turn_2)}</div></div>
        <div><div class="vrow-label">Turn 2 answer</div><div class="vblock">${escapeHtml(q.answer_turn_2 || "")}</div></div></div>`;
    }
    if (q.intended_subject || q.intended_scope) {
      html += `<div class="vdetail" style="margin-top:1rem">
        <div class="vmeta"><span><b>Subject:</b> ${escapeHtml(q.intended_subject || "—")}</span>
        <span><b>Scope:</b> ${escapeHtml(q.intended_scope || "—")}</span></div></div>`;
    }
    if (q.underspecified_variant) {
      html += `<div class="vdetail" style="margin-top:1rem"><div>
        <div class="vrow-label">Underspecified variant (robustness)</div>
        <div class="vblock">${escapeHtml(q.underspecified_variant)}</div></div></div>`;
    }
    openModal("Question", html);
  } else if (kind === "answer") {
    openModal("Golden Answer", `<div class="modal-plain">${escapeHtml(q.golden_answer || "")}</div>`);
  } else if (kind === "verif") {
    openModal("Verification", verifDetailHtml(q));
  } else if (kind === "spans") {
    openModal("Supporting Spans", spansDetailHtml(q));
  }
}

function verifDetailHtml(q) {
  const v = q.verification || {};
  const rows = [];
  rows.push(`<div class="vmeta">
    <span><b>Method:</b> ${escapeHtml(v.method || "?")}</span>
    <span><b>Reproduced:</b> ${v.solver_reproduced === true ? "yes" : v.solver_reproduced === false ? "no" : "n/a"}</span>
    <span><b>Evidence docs:</b> ${v.evidence_docs != null ? v.evidence_docs : "—"}</span>
    <span><b>Expected behavior:</b> ${escapeHtml(q.expected_behavior || "—")}</span>
  </div>`);
  if (v.proposed_answer || v.solver_answer) {
    rows.push(`<div><div class="vrow-label">Proposed answer (generator)</div>
      <div class="vblock proposed">${escapeHtml(v.proposed_answer || "—")}</div></div>`);
    rows.push(`<div><div class="vrow-label">Independent solver answer</div>
      <div class="vblock solver">${escapeHtml(v.solver_answer || "—")}</div></div>`);
  } else {
    rows.push(`<div class="vrow-label">This record uses the earlier verification format (no proposed/solver answer captured). Re-run to get the full audit trail.</div>`);
  }
  if (v.deterministic_matches && v.deterministic_matches.length) {
    rows.push(`<div><div class="vrow-label">Deterministic matches (${v.deterministic_matches.length})</div>
      <div class="vblock">${v.deterministic_matches.map(baseName).map(escapeHtml).join("\n")}</div></div>`);
  }
  if (v.notes) rows.push(`<div class="vmeta"><span><b>Notes:</b> ${escapeHtml(v.notes)}</span></div>`);
  return `<div class="vdetail">${rows.join("")}</div>`;
}

function spansDetailHtml(q) {
  const spans = q.supporting_spans || [];
  if (!spans.length) return `<div class="modal-plain">No supporting spans (e.g. hallucination/absence questions have none by design).</div>`;
  return spans
    .map(
      (s) => `<div class="span-item">
        <div class="span-head">${escapeHtml(s.document)} · page ${s.page}</div>
        <div>${escapeHtml(s.full_text || s.text || "")}</div></div>`
    )
    .join("");
}

// ── Utils ───────────────────────────────────────────────
function uniq(arr) { return [...new Set(arr.filter((x) => x != null))]; }
function baseName(p) { return String(p).split("/").pop(); }
function pretty(s) {
  if (s == null) return "";
  return String(s).split("_").map((w) => w.charAt(0).toUpperCase() + w.slice(1)).join(" ");
}
function escapeHtml(t) {
  const d = document.createElement("div");
  d.textContent = t == null ? "" : String(t);
  return d.innerHTML;
}

// ── Minimal markdown renderer (headings, lists, bold, inline code,
//    and fenced blocks delimited by ``` or :::) ──
function renderMarkdown(md) {
  const lines = String(md).replace(/\r\n/g, "\n").split("\n");
  let html = "", inCode = false, code = [], inList = false, para = [];
  const flushPara = () => { if (para.length) { html += "<p>" + inline(para.join(" ")) + "</p>"; para = []; } };
  const flushList = () => { if (inList) { html += "</ul>"; inList = false; } };
  for (const line of lines) {
    const t = line.trim();
    if (t === "```" || t === ":::") {
      if (inCode) { html += `<pre class="flow-pre">${escapeHtml(code.join("\n"))}</pre>`; code = []; inCode = false; }
      else { flushPara(); flushList(); inCode = true; }
      continue;
    }
    if (inCode) { code.push(line); continue; }
    if (t === "") { flushPara(); flushList(); continue; }
    let m;
    if ((m = t.match(/^(#{1,3})\s+(.*)$/))) {
      flushPara(); flushList();
      const lvl = m[1].length + 2;
      html += `<h${lvl} class="flow-h">${inline(m[2])}</h${lvl}>`;
      continue;
    }
    if ((m = t.match(/^[-*]\s+(.*)$/))) {
      flushPara();
      if (!inList) { html += `<ul class="flow-ul">`; inList = true; }
      html += `<li>${inline(m[1])}</li>`;
      continue;
    }
    para.push(t);
  }
  flushPara(); flushList();
  if (inCode) html += `<pre class="flow-pre">${escapeHtml(code.join("\n"))}</pre>`;
  return html;
}

function inline(text) {
  let s = escapeHtml(text);
  s = s.replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>");
  s = s.replace(/`([^`]+)`/g, "<code>$1</code>");
  return s;
}
