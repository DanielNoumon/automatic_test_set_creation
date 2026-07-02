# Test Set Viewer — v3

A visual inspector for **v3** test sets (`data/test_sets/DSL_corpus_v3/…json`),
tailored to the v3 schema: locality tiers, personas/intents, verification audit
trail, supporting spans, and underspecified variants.

## Usage

No build step. Two options:

1. **Just open it** — double-click `index.html` (opens `file://`), then use the
   file picker to load a v3 test-set JSON.
2. **Serve it** (nicer for some browsers):
   ```bash
   npx serve .        # or: python -m http.server
   ```
   then open the printed URL and load a JSON via the picker.

## What it shows

- **Metadata bar** — pipeline version, language, generator/solver/judge models,
  elapsed time, LLM-call counts.
- **Summary** (collapsible) — KPIs (grounded ratio, solver-reproduced x/y,
  multi-source) plus distributions by tier / intent / persona / type /
  verification method, and the tier-floor checks.
- **Table** — one row per question:
  - **Tier** (T1–T4 / behavioral, colour-coded) · **Type** · **Persona / Intent**
  - **Question** (Dutch) with its underspecified variant and any turn-2 followup
  - **Golden Answer**
  - **Verification** — method + whether the independent solver reproduced the
    answer; click to see the **proposed vs. solver answer side by side**,
    evidence-doc count, and deterministic matches.
  - **Supporting Spans** — count + docs; click to read the actual evidence
    passages (doc + page).
- **Filters** — tier, intent, persona, type, verification status, and free-text
  search. Click any truncated cell to open the full content in a modal.

Works with both the enriched verification format and older test-set files
(missing fields degrade gracefully).
