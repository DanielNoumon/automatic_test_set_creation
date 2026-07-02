# How the v3 test set is built — full walkthrough

This explains, step by step, exactly what the generator does: which steps are
plain Python vs. LLM calls, what goes INTO each LLM call and roughly how much,
and what comes out. If you've never seen this before, read top to bottom.

## The one big idea

Old test sets cut documents into small chunks and wrote a question per chunk —
which secretly favours chunk-retrieval systems (RAG). v3 instead:

- generates questions from **whole documents** and **the whole corpus**,
- finds the supporting evidence **afterwards** (provenance, not the seed),
- and **independently verifies every answer** with a second model.

That makes the questions fair for comparing RAG, file-search and RLM.

## Three LLM roles (all GPT 5.4)

Every LLM call is one of three roles. Same model, different jobs:

```
  ROLE        JOB                                        SEES
  ---------   ----------------------------------------   -----------------------
  GENERATOR   writes the questions + a first answer      a whole doc / corpus map
  JUDGE       checks a question is self-contained        ONLY the question text
  SOLVER      re-answers independently to verify         question + evidence
```

Legend used below:  [PY] = plain Python, no LLM.   [LLM:role] = one model call.

## 1. The whole pipeline at a glance

```
                    +-----------------------------------------------+
                    |  INPUT: 37 source files                       |
                    |  25 CVs · HR handboek · rolomschrijvingen     |
                    |  7 projectflow docs · 1 project-history Excel |
                    +-----------------------------------------------+
                                       |
                                       v
   [PY]  STAGE 0 -- PARSE   (no LLM)
         - read every file            -> full plain text
         - build a section/page map   -> used later for provenance
         - load the Excel as a TABLE  -> rows x columns (for exact counts)
                                       |
            +--------------------------+---------------------------+
            |                          |                           |
            v                          v                           v
  [LLM:gen] STAGE 1           [LLM:gen] STAGE 2          [LLM:gen] STAGE 5
  PER-DOCUMENT                CORPUS-LEVEL               BEHAVIORAL
  policy + process            capability + reference     halluc / ambiguous /
  tiers T1 / T2               tiers T3 / T4              multi-turn / graft
  IN : ONE whole document     IN : corpus map + Excel    IN : corpus map OR a
       (FULL; ~69k max here)       (~28,000 chars)            verified question
  OUT: natural T1/T2 Qs       OUT: cross-doc T3/T4 Qs    OUT: 20 behavioral Qs
            |                          |                           |
            +--------------------------+---------------------------+
                                       |
                                       v
  [LLM:judge] STAGE 4 -- SELF-CONTAINEDNESS GATE
         IN : ONLY the question text (NO document is shown!)
         - can a chatbot understand the referent on its own?
         - FAIL -> repair using intended_subject -> re-judge
                                       |
                                       v
  [PY]  GROUND -- locate supporting spans   (no LLM)
         - find the exact doc+page sections that back the answer
         - rank by anchor tokens (emails / phones / numbers / names)
         - snippet is windowed around the match, not the section start
                                       |
                                       v
  [LLM:solver] + [PY]  STAGE 3 -- VERIFY
         IN : the question + evidence (spans + full Excel for T3/T4)
         - independent GPT 5.4 answers from the EVIDENCE ONLY
         - proposed answer  --compared with-->  solver answer  = gold answer
         - counts/aggregations double-checked in Python (deterministic)
         - dedup: at most one question per topic (key_term)
                                       |
                                       v
  [PY]  STAGE 6 -- OUTPUT
         one JSON { metadata + summary + questions[] }  ->  this viewer
```

## 2. Stage 0 -- Parse   [PY, no LLM]

Reads all 37 files with format-specific parsers (PDF / DOCX / XLSX). Produces
two views of every document plus a structured table:

```
   file.pdf/.docx  --parse-->  full text        (fed to the generator)
                               section/page map  (heading -> page -> text)
   project.xlsx    --parse-->  rows x columns    (exact counts / sums)
```

Nothing is chunked. The section map is only used later to point back at evidence.

## 3. Stage 1 -- Per-document generation   [LLM:generator]

Used for **policy** and **process** questions (tiers T1/T2). The generator is
handed **one entire document** and asked for natural questions a colleague would
ask, each with a first-draft answer and a short verbatim quote.

```
   INPUT to the LLM                              SIZE
   ------------------------------------------    -----------------------------
   the FULL document text (never chunked)        largest here ~69,000 chars;
                                                 hard safety cap 400,000 chars
   persona + intent instruction                  a few lines
   "return N questions as JSON" schema           a few lines

   OUTPUT from the LLM (JSON)
   ------------------------------------------
   question, first-draft answer, answer_type,
   locality_tier (T1/T2), intended_subject,
   supporting_quote, underspecified_variant
```

## 4. Stage 2 -- Corpus-level generation   [LLM:generator]

Used for **capability** and **reference** questions (tiers T3/T4) -- the
cross-document "do we have experience with X?" questions. The generator can't
read all 37 files at once, so it gets a compact **corpus map** instead:

```
   THE CORPUS MAP (built in Python, ~28,000 chars ~= 7k tokens):
   ------------------------------------------------------------
   for each non-Excel doc:  "[category] filename: first 220 chars"
   + the FULL project Excel table (up to 20,000 chars)

   INPUT to the LLM = corpus map + persona/intent + "avoid these topics" list
   OUTPUT (JSON per question):
     question, key_term (a searchable term, e.g. "Databricks"),
     proposed_answer, answer_type, locality_tier (T3/T4),
     intended_subject, underspecified_variant
```

The "avoid these topics" list holds key_terms already used, so questions spread
across different techniques / industries / clients instead of all asking about
the same popular tool.

## 5. Stage 4 -- Self-containedness gate   [LLM:judge]

A separate model is shown **only the question -- never the source document** --
and must decide if the referent is understandable on its own. This is the exact
opposite of the generator's view, which is the point.

```
   IN : just the question string  (e.g. 50-300 chars). NO passage.
   ---------------------------------------------------------------
   PASS  ->  keep the question
   FAIL  ->  repair: rewrite naming the intended_subject, then re-judge
             (e.g. "wat staat er in dit document?"  ->  "wat is X bij DSL?")
```

The judge knows the chatbot has the whole corpus, so "de projectentabel" is
fine, but "dit document" / "deze rol" is not.

## 6. Ground -- locate supporting spans   [PY, no LLM]

Now that the question exists, Python searches the source docs for the sections
that actually contain the answer -- this becomes the provenance for grading.

```
   answer  --extract anchors-->  emails, phone numbers, money, numbers,
                                 quoted phrases, proper names
   score each section = 6 x (anchor hits) + (keyword hits)
   keep sections near the top score (drops weak / irrelevant matches)
   snippet = a window CENTERED on the match (not the section start)
```

## 7. Stage 3 -- Verify   [LLM:solver] + [PY]

The trust step. An **independent** GPT 5.4 answers the question from the
evidence only, and its answer becomes the gold answer.

```
   EVIDENCE given to the solver:
   -----------------------------
   the located spans (up to ~25 short snippets)
   + for T3/T4: the FULL project Excel table (up to 60,000 chars)

   proposed_answer (from generator)  --compare-->  solver_answer (independent)
                                                   = golden_answer
   solver_reproduced = did they agree?

   PLUS deterministic Python checks for counts / "which docs mention X"
   (never trust an LLM to count) -> recorded in deterministic_matches
```

If the solver can't answer from the evidence, the question is dropped. A dedup
guard also drops a question whose topic (key_term) was already used.

## 8. Stage 5 -- Behavioral questions   [LLM:generator]

The 20 behavioral questions are made specially:

```
   hallucination  -> ask a plausible question whose answer is ABSENT;
                     Python confirms the key_term appears NOWHERE in the corpus
   ambiguous      -> ask a genuinely multi-interpretation question about one doc
   multi-turn     -> a 2-turn chat; turn 2 is unanswerable without turn 1
   injection      -> wrap a VERIFIED question in "ignore all instructions..."
   aggro          -> wrap a VERIFIED question in an angry tone
                     (both keep the real answer, add refusal / de-escalation)
```

## 9. Stage 6 -- Output   [PY]

Everything is written to one JSON file with three parts:

```
   metadata  : models used, timings, LLM-call counts
   summary   : counts by tier / intent / persona / type, grounded %,
               solver-reproduced %, tier-floor checks
   questions : the full list (see the record shape below)
```

## How much goes into each LLM call (context budget)

The single most important thing to understand about cost/scale:

```
   CALL                     ROLE      MAIN INPUT                    ~SIZE
   ----------------------   -------   ---------------------------   ------------
   per-doc generation       gen       one FULL document            ~69k max here
   corpus generation        gen       corpus map + full Excel       ~28,000 ch
   self-containedness gate  judge     the question only             ~0.1-0.3k ch
   repair (only on fail)    judge     question + intended_subject   ~0.5k ch
   verify (T1/T2)           solver    question + a few spans        ~1-3k ch
   verify (T3/T4)           solver    question + spans + Excel       ~40-60k ch
   behavioral graft         gen       one verified Q + its answer   ~0.5k ch
```

Rough call budget for a full 100-question run: generation is batched (tens of
calls), then ~1-2 judge calls per surviving question and ~1 solver call per
content question -> a few hundred calls total.

## Locality tiers -- the axis that separates architectures

```
   T1  Local       answer sits in one passage            (RAG baseline)
   T2  Intra-doc   >=3 spots in one doc / whole-doc read
   T3  Cross-doc   combine many documents                (naive RAG fails)
   T4  Global      corpus-wide count / enumerate         (agentic / RLM)
```

## Distribution (100 questions)

```
   100  =  90 content  +  10 behavioral      (90 / 10 split)

   content by intent          behavioral quota
   -----------------          ----------------
   capability .... 45         hallucination .. 3
   reference ..... 18         adversarial .... 2
   policy ........ 18         injection ...... 2
   process ....... 9          ambiguous ...... 2
                              multi-turn ..... 1
```

## What one question record contains

```
   question, golden_answer, underspecified_variant   (all Dutch)
   locality_tier, type, persona, intent_cluster
   supporting_spans[]  -> { document, page, text }    (provenance)
   verification        -> { method, solver_reproduced,
                            proposed_answer, solver_answer,
                            evidence_docs, deterministic_matches }
   intended_subject, intended_scope, expected_behavior
```
