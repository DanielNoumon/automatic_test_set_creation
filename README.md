# Test Set Creator

Automated golden Q&A test set generation for RAG evaluation. Given a corpus of documents, this pipeline produces a diverse set of question-answer pairs — each grounded in a specific passage — that can be used to benchmark retrieval and generation quality of a RAG system.

The key design principle is **minimal LLM usage**: parsing, indexing, passage selection, and validation are all deterministic. The LLM is called exactly once per question (for generation) and optionally once more for quality scoring, making runs fast and reproducible.

## Project Structure

```
pipeline/          ← active pipeline (Parse → Index → Select → Generate → Validate → Score)
deprecated/        ← v1 RLM-based pipeline (preserved for reference)
test-set-viewer/   ← web UI for reviewing generated test sets
data/              ← input documents + output test sets
main.py            ← entry point and configuration
```

See `pipeline/WORKFLOW.md` for a detailed Mermaid diagram of every phase.

---

## How the Test Set Is Created

The pipeline runs in six phases. Each phase is fully deterministic except for the LLM generation and quality scoring steps.

### Phase 0 — Parse Documents

The pipeline recursively discovers all files in the input folder and converts them into a uniform `Document → Section` tree:

- **PDF** (`PyMuPDF`) — extracts text page by page, detects headings via heuristics (numbered chapters like "3.2.1", title-case lines, font-size changes), and falls back to page-based splitting when no headings are found.
- **Word** (`python-docx`) — maps heading styles (`Heading 1`, `Heading 2`, etc.) to a section hierarchy; body text under each heading becomes section content.
- **Excel** (`openpyxl`) — each worksheet becomes a single section with all rows rendered as pipe-separated text. No chunking or header detection is applied; the full sheet is preserved so the LLM receives complete tabular context.
- **TXT / Markdown** — splits on markdown headings (`#`, `##`, ...) or blank-line-separated paragraphs.

Every section is then tagged with structural metadata: `has_list`, `has_table`, `has_dates`, `has_definitions`, etc. These tags drive passage selection in later phases.

### Phase 1 — Index the Corpus

Two lightweight indexes are built over all sections:

- **BM25 search index** — tokenises every section and computes term frequencies and inverse document frequencies. Used later for hallucination validation (checking whether a supposedly unanswerable topic actually appears elsewhere in the corpus) and for entity-overlap scoring.
- **Entity extractor** — regex-based extraction of dates (Dutch + English formats), monetary amounts, phone numbers, emails, URLs, defined terms (e.g. "X means ..."), and key noun phrases. Entity counts per section feed into the passage selection scoring.

No embeddings or vector databases are involved — the indexing is pure text processing, keeping it fast and dependency-light.

### Phase 2 — Select Passages (Per Question Type)

For each enabled question type, a **dedicated selection strategy** picks the best candidate passages from the indexed corpus. This is where most of the intelligence lives — different question types need fundamentally different source material:

| Strategy | Question Type(s) | How It Selects |
|---|---|---|
| Factual density scoring | Direct Lookup | Ranks sections by concentration of numbers, dates, monetary values, and contact info — sections rich in hard facts produce better lookup questions. |
| Descriptive + definition filtering | Paraphrase Lookup, Ambiguous, Multi-turn | Prefers longer sections with definitions and explanatory text; filters out boilerplate (footers, copyright notices). |
| Defined-term targeting | Specific Jargon | Finds sections containing explicit term definitions and abbreviations. |
| Intra-document pairing | Multi-hop Within | Pairs two sections from the same document that share meaningful entities, so the question must combine information from both. |
| Cross-document pairing | Multi-hop Between, Temporal | Pairs sections from different documents with overlapping topics. Uses a "meaningful shared entities" filter that strips corpus-common stop words (e.g. company name, generic role titles) to ensure the connection is substantive, not trivial. |
| Low-density entity sections | Needle in Haystack | Deliberately picks sections where facts are sparse and buried — the opposite of direct lookup. |
| List detection | Lists Extraction | Sections containing bullet or numbered lists, preferring longer lists. |
| Multi-section spans | Long Context Synthesis | Assembles 4 consecutive sections into a single passage (up to 4000 chars), using section-aware assembly that never cuts mid-section — preventing factually broken golden answers. |
| Random diverse sections | Hallucination Test | Selects random sections as context for questions about topics *not* in the corpus. |
| Content-rich generic | Adversarial, Prompt Injection | Picks information-dense sections by word + entity count; filters boilerplate. |

A **diversity tracker** prevents section reuse across question types, ensuring the test set covers a broad spread of the corpus.

**Truncation handling**: passages are truncated at sentence boundaries to fit context windows, *except* for Excel sheets which are always passed in full (the LLM receives the entire worksheet).

### Phase 3 — Generate Q&A (Single LLM Call)

Each selected passage is sent to the LLM (Azure OpenAI) with a **type-specific prompt** that includes:
- The passage text and source document metadata
- Instructions tailored to the question type (e.g. "ask about a specific detail buried in a long section" for needle-in-haystack, or "respond with a refusal prefix" for prompt injection)
- A difficulty hint (easy / medium / hard)

The LLM returns a JSON `{question, answer}` pair. For multi-turn follow-up questions, it returns a 4-field JSON with `{question, answer, followup_question, followup_answer}` where the follow-up uses pronouns to test conversational memory.

Only **one LLM call** is made per question — there is no iterative refinement loop.

### Phase 4 — Validate (Deterministic)

Every generated Q&A pair passes through a chain of deterministic checks before it is accepted:

1. **Context match** — at least 30% of the passage must be found verbatim in the source documents (guards against hallucinated context).
2. **Table-of-contents rejection** — detects and discards passages that are just ToC entries.
3. **Answer grounding** — at least 25% of answer keywords must appear in the passage. Skipped for question types where the answer is intentionally *not* in the passage (hallucination tests, adversarial, prompt injection).
4. **Hallucination BM25 check** — for hallucination-test questions only: searches the full corpus to verify the question's topic truly doesn't appear elsewhere. Uses BM25 scoring with a threshold of 3.0 and keyword overlap checks.
5. **Context length** — passages shorter than 30 characters are rejected.
6. **Multi-hop diversity** — for cross-document questions, verifies the two source sections actually come from different documents.

If any check fails, the question is discarded and the pipeline tries the next candidate passage.

### Phase 5 — Quality Scoring (LLM-as-Judge, Optional)

Accepted Q&A pairs are optionally scored by a second LLM call acting as a judge. The judge evaluates four dimensions on a 1–5 scale:

- **Self-containedness** — can the question be understood without seeing the passage?
- **Answer accuracy** — is the answer correct given the passage?
- **Naturalness** — does the Q&A read like something a real user would ask?
- **Difficulty alignment** — does the actual difficulty match the intended level?

A composite score is computed and pairs below the threshold (default 3.5) are filtered out.

### Phase 6 — Save & Track

The final test set is saved as a JSON file in `data/test_sets/{corpus_name}/`:
- Each entry includes: question, golden answer, golden context passage, source documents (with subfolder paths), page numbers, question type, difficulty, and quality scores.
- **MLflow** logs per-type question counts, yield ratios, context match statistics, timing, and LLM call counts to a local SQLite database (`mlflow.db`).

---

## Question Types (15 enabled / 20 defined)

| # | Type | Difficulty | Description |
|---|---|---|---|
| 1 | Direct Lookup | easy | Exact fact retrieval |
| 2 | Paraphrase Lookup | medium | Rephrased fact retrieval |
| 3 | Specific Jargon | medium | Domain term definitions |
| 4 | Multi-hop Within | hard | Combine info from same document |
| 5 | Multi-hop Between | hard | Combine info across documents |
| 6 | Temporal | medium | Document versioning / recency |
| 7 | Pinpointing/Quoting | medium | Source location identification |
| 8 | Long Context Synthesis | hard | Counting/summarizing across sections |
| 9 | Needle in Haystack | hard | Hidden detail retrieval |
| 10 | Ambiguous Questions | medium | Intentionally vague wording |
| 11 | Lists Extraction | easy | Items from bullet/numbered lists |
| 12 | Hallucination Test | medium | Unanswerable questions (BM25-validated) |
| 13 | Adversarial/Aggro | hard | Aggressive tone with de-escalation |
| 14 | Prompt Injection | hard | Jailbreak resistance |
| 15 | Multi-turn Followup | medium | Conversational memory |

## Installation

```bash
conda create -n rlm_test_set_creation python=3.11
conda activate rlm_test_set_creation
pip install -e .
```

## Configuration

Create a `.env` file in the project root:
```
AZURE_OPENAI_API_KEY=your_key
AZURE_OPENAI_ENDPOINT_GPT5_MINI=https://your-resource.api.cognitive.microsoft.com/...
DEPLOYMENT_NAME_GPT5_MINI=gpt-5-mini
API_VERSION_GPT5_MINI=2025-04-01-preview
```

Edit question types, counts, and other settings in `main.py`:
```python
COUNT_OVERRIDE = 1    # Set to None to use per-type counts below
QUESTION_TYPES = {
    QuestionType.DIRECT_LOOKUP: QuestionConfig(
        enabled=True, count=3, difficulty="easy"
    ),
    # ...
}
```

## Usage

```bash
python main.py
```

### Input

Place documents in `data/files_for_test_set/` (subfolders are traversed recursively):
- `.pdf`, `.docx`, `.xlsx`, `.xls`, `.txt`, `.md`

### Output

Test sets saved to `data/test_sets/{corpus_name}/{corpus_name}_{timestamp}.json` with:
- Questions with golden answers and context
- Per-question metadata (source docs, pages, match ratios, quality scores)
- MLflow experiment tracking (SQLite: `mlflow.db`)

## Test Set Viewer

A browser-based UI for reviewing generated test sets. Load a JSON output file to browse questions in a filterable table with search, type/difficulty filters, and expandable cells for long content.

```bash
cd test-set-viewer
npm install
npm start
```

This compiles the TypeScript and launches a local server (default `http://localhost:3000`). Use the file picker to load a test set JSON from `data/test_sets/`.

## License

MIT License — see LICENSE file for details.
