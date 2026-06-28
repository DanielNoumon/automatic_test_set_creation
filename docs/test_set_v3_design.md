# Test Set v3 — Design Doc

**Goal:** Build a DSL test set that fairly compares answering architectures
(RAG, file-search, Recursive Language Models, and others) on answer quality —
not on whether a single chunk happens to be retrievable.

**Status:** Design. No code written yet. This doc is the spec to implement against.

**Scope decisions (confirmed with stakeholder):**
- The chatbot-under-test has access to the **full corpus (all 37 files)** at query time.
  → cross-corpus aggregation questions are fair and in-scope.
- CVs: use a **representative subset for single-document (T1/T2) questions**, but the
  **full CV population for cross-corpus aggregation (T3/T4)**.

---

## 1. Why v2 must be replaced (root cause)

v2 is **passage-first** ([pipeline/selection/strategies.py](../pipeline/selection/strategies.py)):
a chunk is scored by entity density, selected, and *becomes* the `golden_context`;
the LLM is then told to write a question answerable **entirely from that chunk**
([prompts.py:364](../pipeline/generation/prompts.py#L364)); the validator confirms the
answer's keywords appear back in the same chunk.

Every question is therefore reverse-engineered from one small, retrievable chunk that
is also the gold context. This **structurally favors RAG** (retrieve one chunk → win)
and never creates the conditions — distributed evidence, evidence far from query
keywords, whole-corpus aggregation — where RLM / file-search / long-context differ.
Even v2's "hard" types stay chunk-sized (`multi_hop` = two ~800-char snippets;
`long_context_synthesis` ≤ 4000 chars within one doc).

**v3 principle:** *Decouple question generation from chunking.* Generate from whole
documents and from the whole corpus; locate grounding/provenance **after** generation;
organize the set around retrieval-locality, the axis that actually separates
architectures.

### Secondary v2 defects v3 also fixes
- Over-stuffed, unnatural phrasing from the self-containedness rules
  ([prompts.py:373](../pipeline/generation/prompts.py#L373)).
- "Type" faked via prompt over a generic chunk (`ambiguous`, `multi_turn`,
  `adversarial`, `prompt_injection` all reuse generic/paraphrase strategies —
  [strategies.py:887](../pipeline/selection/strategies.py#L887)).
- `temporal_questions` broken for this corpus (no versioned pairs → gold answers like
  "Dat is niet vast te stellen").
- Adversarial types skip quality scoring entirely
  ([pipeline.py:212](../pipeline/pipeline.py#L212)).
- CV genre dominance → interchangeable, trivially retrievable questions.

---

## 2. The locality-tier model (core of the design)

Every question is tagged with a **locality tier** = how much of the corpus must be read
to answer it. This is the primary axis we report results along.

| Tier | Definition | `min_docs_required` | Discriminates | Corpus examples |
|------|------------|---------------------|---------------|-----------------|
| **T1 Local** | answerable from one contiguous span | 1 (1 span) | baseline; RAG should ace this | ziekmelding-bedenktijd; één budgetcel |
| **T2 Intra-doc distributed** | ≥3 separated places in one doc, or a whole-doc read | 1 (≥3 spans) | small top-k RAG starts to miss | samenvatting studiekosten-/amortisatieregels (Handboek); volledige projectflow-volgorde |
| **T3 Cross-doc aggregation** | scan many documents | ≥3 docs | naive top-k RAG fails; RLM/agentic/file-search win | "welke consultants kennen Databricks?" (alle CV's); "totaal 2026-budget Data Engineering" (Excel) |
| **T4 Global/structural** | whole-corpus count/enumerate | most/all docs | only agentic/RLM | "hoeveel mensen met master in data-vakgebied?"; "hoeveel rollen noemen een jaarplan?" |

**Target distribution (tune later):** T1 35% / T2 30% / T3 25% / T4 10%.
Rationale: enough T1 to anchor a baseline, heavy T2/T3 because that is where the
architectures separate.

### Type → tier remap
Keep the 14 question types but re-anchor each to a tier and the new generation method:

| v2 type | Tier | Generation source |
|---------|------|-------------------|
| direct_lookup, paraphrase_lookup, specific_jargon, needle_in_haystack | T1 | whole-doc gen |
| lists_extraction, multi_hop_within_corpus | T2 | whole-doc gen |
| multi_hop_between_documents, cross_document_conflict, long_context_synthesis | T3 | corpus-level gen + verification |
| (new) global_count / structural | T4 | corpus-level gen + deterministic verification |
| temporal_questions | — | **dropped for now** (type kept in enum, disabled). Re-enable later by authoring versioned document pairs. |
| hallucination_test | any | corpus-wide absence check |
| adversarial_aggro, prompt_injection | any | graft onto a verified T1/T2 question |
| ambiguous_questions | T1/T2 | dedicated gen (genuine referential/scope ambiguity), not "X or Y?" |
| multi_turn_followup | any | dedicated gen (turn-2 unanswerable without turn-1) |

---

## 2.5 Personas & intents (top-down steering)

Bottom-up generation produces *answerable* but not necessarily *realistic* questions
(the v2 "questions feel off" problem). We steer generation top-down with personas and
intents, which (a) makes questions realistic and (b) sets the question-mix weights.

**Happy alignment:** the predominant real use case — *"do we have experience in
project type X / industry Y / technique Z, and how much?"* — is inherently
**cross-corpus (T3/T4)**: it combines the **CVs** (who knows what) with the **Excel
project history** (`Company Domain`, `Techniques`, `Budget`, dates). So the questions the
business cares about are the same ones that are hard for naïve RAG and that separate
the architectures. Persona steering therefore *also* weights the set toward the
discriminating tiers.

**Personas:** Sales / Business Development · Strategy / Leadership · Technical
consultant · All-staff (policy/HR).

**Intent clusters → tier / stage / verification:**

| Intent | Example | Tier | Source / verification |
|--------|---------|------|----------------------|
| **A. Capability discovery** | "Hebben we ervaring met computer vision in de zorg?" | T3/T4 | CVs + Excel; corpus search |
| **B. Capability depth / quantity** | "Hoeveel projecten in de publieke sector?" / "Wie heeft de meeste NLP-ervaring?" | T3/T4 | count/sum over CVs + Excel (code) |
| **C. Staffing / intersection** | "Wie kan een project met Databricks én overheidservaring bemensen?" | T3 | set-intersection across CVs |
| **D. Reference / evidence** | "Welke klantprojecten gebruikten een knowledge graph?" | T1–T3 | Excel + projectflow |
| **E. Policy / HR** | "Hoe werkt de studiekostenregeling?" | T1/T2 | Handboek |
| **F. Process** | "Wat zijn de stappen van een externe kick-off?" | T2 | projectflow |

**Default intent weighting** (capability cluster dominant, matching real usage):
A+B+C ≈ 50% · D ≈ 20% · E ≈ 20% · F ≈ 10%. This naturally pushes the mix toward T3/T4.

**Two bonuses:**
- Persona questions yield **realistic negative/hallucination tests for free** — e.g.
  "Hebben we ervaring met quantum computing?" is a real sales question with no corpus
  support; correct answer = "not found". Far better than v2's synthetic hallucinations.
- "How much" intents force **quantification**, exactly what Stage 2 deterministic
  verification handles.

Personas + intents are **inputs to Stage 1/2 generation** ("given this content,
generate questions a *Sales colleague discovering capability* would ask"), and the
weights replace v2's even split across the 14 types.

---

## 3. Pipeline stages

**Global rule — language:** ALL generated questions, variants, and gold answers are in
**Dutch**, regardless of the source document's language (one CV is in English; questions
about it are still Dutch). The chatbot's users are Dutch-speaking colleagues. This is
enforced in every generation prompt and re-checked by the isolation/quality judges.

### Stage 0 — Parse (reuse, fix provenance)
Reuse the parsers ([pipeline/parsing/](../pipeline/parsing/)). Keep **two views** per doc:
1. **Full text** (for generation).
2. **Section/page map** (heading → page → char-span) for *provenance only*, never to
   bound a question.
Excel: keep as a real table (rows/columns), not flattened prose, so deterministic
verification can run against it.

### Stage 1 — Per-document generation (T1/T2)
For each selected document (CV subset + all policy/process docs + Excel):
- Feed the **entire document** to a strong long-context model.
- Ask for N natural user questions at varied locality (mix of T1 and T2), each with a
  grounded gold answer and the type label.
- **Locate supporting spans** post-hoc by string/semantic match against the section map
  → `supporting_spans`. Reject if a span can't be found (answer not grounded).
- Phrasing rule (replaces v2 over-stuffing): questions are **short and natural**; the
  disambiguating subject goes into `intended_subject` metadata, not the question text.
  Generate an optional `underspecified_variant` for robustness testing.

### Stage 2 — Corpus-level generation (T3/T4)
- Build a **corpus index**: per-doc summary + key entities + the Excel as a structured
  table.
- Generate aggregation/structural questions from the index (e.g. "which consultants
  have X", "total budget for Y", "how many roles mention Z").
- **Deterministically verify** every T3/T4 gold answer:
  - Excel questions → compute against the parsed table (pandas).
  - "who knows/did X" → corpus-wide search; gold answer = the verified hit set.
  - counting → programmatic count over the corpus, not LLM guess.
- `supporting_spans` = the union of contributing spans across all source docs.

### Stage 3 — Independent answerability gate (replaces v2 validator)
A **separate solver agent** (model: **GPT 5.4**, kept independent from the architectures
under test to avoid grading bias) with full-corpus access must reproduce the gold answer
(closed against external knowledge, open against the corpus). Compare solver output to
gold:
- Match → keep.
- Mismatch → flag for review / drop.
This is the real filter: it guarantees the question is answerable from the corpus and
the gold answer is correct, **without** tying the question to a chunk.

### Stage 4 — Realism & the self-containedness gate
A question must be understandable with **zero knowledge of where it came from**. v2 had
prompt rules for this and they still failed (and over-corrected into stilted phrasing),
so v3 enforces it with an **LLM-as-judge gate that evaluates the question in isolation**,
while keeping questions short and natural (self-contained ≠ verbose).

Three layers:
1. **Isolation judge** — an LLM judge is shown **only the question (no passage)** and
   must resolve the referent. Catches both obvious deixis ("dit document", "deze rol")
   and *bare definite references* ("the personal profile" / "het profiel"). Differs from
   v2, which scored self-containedness *with the passage visible* — defeating the
   purpose. (No regex blocklist: brittle, false-positive-prone, and the judge subsumes
   it.)
2. **Repair** — failed questions are rewritten using the known `intended_subject`
   metadata (e.g. "What does this document say about the personal profile?" → "Wat is
   het profiel van [Persoon X]?"), then re-judged. Repair (not discard) preserves yield.
3. **Solver-agent backstop (Stage 3)** — a question that says "this document" is
   unanswerable for an independent full-corpus solver, so anything that slips through is
   caught there.

Other Stage 4 outputs:
- `intended_subject`, `intended_scope` recorded as fields, not baked into the question.
- `underspecified_variant` (Dutch) — a deliberately terse/vague rephrasing of the same
  intent (e.g. "Welke consultants bij DSL hebben ervaring met Databricks?" → "wie kent
  databricks?"). **Graded as a separate robustness sub-benchmark** (see §5): each
  architecture is run on both the clean question and the variant, and we report the
  performance drop on vague queries.

### Stage 5 — Behavioral types grafted onto verified questions
- `prompt_injection` / `adversarial_aggro`: take an already-verified T1/T2 question,
  wrap with the injection/aggro envelope. Factual core stays grounded; expected
  behavior = refusal/de-escalation **then** the correct factual answer.
- `hallucination_test`: pick a plausible in-domain detail, verify by **corpus-wide
  search** that it is absent everywhere; gold answer = "not available in the documents".
- Run quality scoring on these too (remove the v2 `_SKIP_QUALITY` bypass) using a
  behavior-appropriate rubric.

### Stage 6 — Output, provenance & grading schema
See §4.

---

## 4. Output schema (per question)

```jsonc
{
  "id": "v3_<type>_<n>",
  "type": "multi_hop_between_documents",
  "locality_tier": "T3",
  "min_docs_required": 4,
  "question": "Welke consultants hebben ervaring met Databricks?",
  "intended_subject": "consultants with Databricks experience",
  "intended_scope": "all CVs",
  "underspecified_variant": "Wie kent Databricks?",
  "golden_answer": "…verified…",
  "answer_type": "set | scalar | span | refusal | enumeration",
  "supporting_spans": [
    {"document": "DSL_data/HR/CV/CV X.docx", "page": 1, "text": "…Databricks…"}
  ],
  "source_documents": ["…", "…"],
  "verification": {
    "method": "deterministic | solver_agent | corpus_search",
    "solver_reproduced": true,
    "notes": ""
  },
  "expected_behavior": "…",
  "grading_rubric": {
    "answer_correctness": "LLM-judge vs golden_answer (0-1)",
    "completeness": "fraction of gold set items recovered (for set/enumeration)",
    "retrieval_recall": "fraction of supporting_spans surfaced by the system"
  },
  "metadata": { "generated_by": "v3", "llm_model": "…", "tier_rationale": "…" }
}
```

**Key change vs v2:** `golden_context` (one chunk) → `supporting_spans` (a list across
docs). This enables **retrieval-recall** scoring per architecture, alongside
**answer-correctness**, so the comparison is fair across RAG / file-search / RLM.

---

## 5. How the eval uses this (the payoff)

For each architecture, report **answer-correctness by tier**:

```
              T1     T2     T3     T4
RAG          0.95   0.70   0.40   0.20
file-search  0.93   0.85   0.78   0.55
RLM          0.92   0.88   0.86   0.80
```

The interesting signal lives in T2–T4. T1 confirms nobody is broken on the easy case.
`retrieval_recall` explains *why* an architecture missed (didn't surface the spans vs
surfaced but reasoned wrong).

**Robustness sub-benchmark (`underspecified_variant`):** each architecture is also run
on the terse/vague variant of every question. We report the correctness **drop** from
clean → vague phrasing — a separate axis that shows how well each architecture tolerates
real-world sloppy queries.

---

## 6. Corpus curation
- **CV subset for T1/T2:** sample ~6–8 representative CVs (vary role/seniority/format:
  include some `.pdf` and some `.docx`).
- **All 24 CVs for T3/T4:** breadth is the point for aggregation/counting.
- Ensure non-CV coverage: Handboek (rich for T2 policy synthesis), Rolomschrijvingen
  (roles → T2/T4 counting), projectflow (sequence/process → T2/T3), Excel (numeric
  aggregation → T3/T4 with deterministic verification).

---

## 7. Build order (when we implement)
1. Schema + tier enum + config knobs (tier distribution, CV subset list, persona/intent
   weights from §2.5).
2. Stage 0 provenance map + Excel-as-table.
3. Stage 1 persona/intent-driven whole-doc generation + span locator.
4. Stage 4 self-containedness gate (isolation judge + repair) — applied to Stage 1
   output.
5. Stage 3 solver-agent answerability gate (validates Stage 1/4 before going further).
6. Stage 2 corpus index + deterministic verification (Excel/search/counting).
7. Stage 5 behavioral grafting + corpus-wide absence check.
8. Stage 6 output writer + summary/metrics by tier.
9. Optional: prototype Stage 2 first on Excel+CVs to de-risk (per stakeholder, a slice
   PoC is a fallback if Stage 1 underperforms).

---

## 8. Decisions & open questions

**Resolved**
- `temporal`: **dropped for now**, type retained (disabled) so it can be re-enabled
  later by authoring versioned document pairs.
- Solver-agent (Stage 3): **independent**, model **GPT 5.4**.
- Distribution: **hybrid, 100 questions total, behavioral quota carved out *inside* the
  100** (80 content + 20 behavioral). See §9.
- `underspecified_variant`: **graded as a separate robustness sub-benchmark** (clean vs
  vague). See §5.
- Language: **all questions and answers in Dutch** (global rule in §3).

**Open**
- _None blocking. Spec is ready to implement._

---

## 9. Distribution (resolved)

**Approach:** hybrid — drive *content* questions by persona/intent (realism + business
value, which naturally loads the discriminating T3/T4 tiers), add a tier floor, and
carve out a fixed behavioral quota. **Total = 100, behavioral quota is inside the 100.**

So: **80 content + 20 behavioral = 100.**

### Knob 1 — Content questions (80), by persona/intent

Intent weights applied to the 80 content questions:

| Intent cluster | Persona | Weight | Count |
|----------------|---------|--------|-------|
| Capability discovery + depth + staffing (A+B+C) | Sales / Strategy / Technical | 50% | **40** |
| Reference / evidence (D) | Sales / Technical | 20% | **16** |
| Policy / HR (E) | All-staff | 20% | **16** |
| Process (F) | Technical / All-staff | 10% | **8** |
| | | | **80** |

### Knob 2 — Behavioral quota (20), fixed

| Type | Count |
|------|-------|
| hallucination_test (incl. realistic negatives, e.g. "ervaring met quantum computing?") | 6 |
| adversarial_aggro | 4 |
| prompt_injection | 4 |
| ambiguous_questions | 3 |
| multi_turn_followup | 3 |
| | **20** |

### Emergent tier distribution (derived, then verified)

Tiers are not set directly — they fall out of the intent mix above:

| Tier | ~Count (of 80 content) | Mainly from |
|------|------------------------|-------------|
| T1 Local | ~12 | policy, some process |
| T2 Intra-doc | ~22 | policy synthesis, process, some reference |
| T3 Cross-doc | ~36 | capability, reference |
| T4 Global | ~10 | capability counting/enumeration |

Behavioral (20) are graded on behavior; tier tagged where it applies, otherwise N/A.

**Tier floor:** enforce **T1 ≥ 12** and **T3+T4 ≥ 40** so the baseline and the
discriminating zone are both well-populated. After generation, report the actual
tier/type/persona breakdown and nudge weights if a cell is too thin.
