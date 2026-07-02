"""
Stage 0 — corpus loading & views.

Reuses the v2 parsers but exposes two views per document:
  1. full text (for whole-document generation),
  2. a section/page map (for provenance only — never to bound a question).

Also classifies documents, picks the CV subset for T1/T2, exposes the Excel as a
structured table for deterministic verification, and provides a lightweight
corpus-wide search used for grounding and capability/counting checks.
"""
from __future__ import annotations

import random
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from pipeline.parsing.document import Document, Section
from pipeline.parsing.pdf_parser import parse_pdf
from pipeline.parsing.docx_parser import parse_docx
from pipeline.parsing.excel_parser import parse_excel
from pipeline.parsing.text_parser import parse_text_file


# Document category by relative path
CAT_CV = "cv"
CAT_POLICY = "policy"
CAT_PROCESS = "process"
CAT_PROJECTS = "projects"
CAT_OTHER = "other"


@dataclass
class SectionRef:
    """A section with its provenance, used to locate supporting spans."""
    document: str
    heading: str
    text: str
    page_start: int
    page_end: int


@dataclass
class CorpusDoc:
    document: Document
    filename: str
    category: str
    sections: List[SectionRef] = field(default_factory=list)

    @property
    def full_text(self) -> str:
        return self.document.raw_text

    @property
    def page_count(self) -> int:
        return self.document.page_count


def _classify(rel_path: str) -> str:
    p = rel_path.lower()
    if "/cv/" in p or p.endswith(("cv.docx", "cv.pdf")):
        return CAT_CV
    if "projecten" in p or p.endswith((".xlsx", ".xls")):
        return CAT_PROJECTS
    if "projectflow" in p:
        return CAT_PROCESS
    if "/hr/" in p or "operations" in p:
        return CAT_POLICY
    return CAT_OTHER


def _parse_one(fpath: Path, rel: str) -> Optional[Document]:
    suffix = fpath.suffix.lower()
    try:
        if suffix == ".pdf":
            return parse_pdf(str(fpath))
        if suffix == ".docx":
            return parse_docx(str(fpath))
        if suffix in (".xlsx", ".xls"):
            return parse_excel(str(fpath))
        if suffix in (".txt", ".md"):
            return parse_text_file(str(fpath))
    except Exception as e:  # noqa: BLE001 — keep loading the rest of the corpus
        print(f"  [corpus] error parsing {rel}: {e}")
    return None


class Corpus:
    """The loaded DSL corpus with all views the v3 pipeline needs."""

    def __init__(self, docs: List[CorpusDoc], cv_subset: List[str]):
        self.docs = docs
        self.by_name = {d.filename: d for d in docs}
        self.cv_subset = cv_subset

    # ── Loading ─────────────────────────────────────────
    @classmethod
    def load(cls, input_path: str, cv_subset_size: int, seed: int = 42,
             cv_subset_override: Optional[List[str]] = None) -> "Corpus":
        root = Path(input_path)
        if not root.exists():
            raise FileNotFoundError(f"Input path not found: {root}")

        docs: List[CorpusDoc] = []
        for fpath in sorted(root.rglob("*")):
            if not fpath.is_file():
                continue
            rel = str(fpath.relative_to(root)).replace("\\", "/")
            doc = _parse_one(fpath, rel)
            if doc is None:
                continue
            doc.filename = rel
            refs = [
                SectionRef(
                    document=rel,
                    heading=s.heading,
                    text=s.full_text,
                    page_start=s.page_start,
                    page_end=s.page_end,
                )
                for s in doc.all_sections_flat()
                if s.full_text.strip()
            ]
            docs.append(CorpusDoc(
                document=doc, filename=rel,
                category=_classify(rel), sections=refs,
            ))

        cv_subset = cls._pick_cv_subset(
            docs, cv_subset_size, seed, cv_subset_override)
        return cls(docs, cv_subset)

    @staticmethod
    def _pick_cv_subset(docs: List[CorpusDoc], n: int, seed: int,
                        override: Optional[List[str]] = None) -> List[str]:
        """Pick the CV subset. If `override` is given, use those CVs (matched by
        basename/relative path); backfill any shortfall with the auto-picker.
        Otherwise pick a representative set, preferring format variety."""
        cvs = [d.filename for d in docs if d.category == CAT_CV]
        if len(cvs) <= n and not override:
            return cvs

        if override:
            chosen = [
                c for c in cvs
                if any(c == o or c.endswith("/" + o) or c.endswith(o)
                       for o in override)
            ]
            unmatched = [o for o in override
                         if not any(c == o or c.endswith(o) for c in cvs)]
            for o in unmatched:
                print(f"  [corpus] cv_subset_override entry not found: {o}")
            if len(chosen) >= n:
                return sorted(chosen)[:n]
            # backfill remaining slots from the auto-pick, skipping already chosen
            remaining = [c for c in cvs if c not in chosen]
            fill = Corpus._auto_pick(remaining, n - len(chosen), seed)
            return sorted(chosen + fill)

        return Corpus._auto_pick(cvs, n, seed)

    @staticmethod
    def _auto_pick(cvs: List[str], n: int, seed: int) -> List[str]:
        if len(cvs) <= n:
            return sorted(cvs)
        rng = random.Random(seed)
        pdfs = [c for c in cvs if c.lower().endswith(".pdf")]
        docx = [c for c in cvs if c.lower().endswith(".docx")]
        rng.shuffle(pdfs)
        rng.shuffle(docx)
        n_pdf = min(len(pdfs), max(1, n // 3)) if pdfs else 0
        chosen = pdfs[:n_pdf] + docx[: n - n_pdf]
        return sorted(chosen)[:n]

    # ── Accessors ───────────────────────────────────────
    def by_category(self, category: str) -> List[CorpusDoc]:
        return [d for d in self.docs if d.category == category]

    def cvs(self, subset_only: bool = False) -> List[CorpusDoc]:
        if subset_only:
            return [self.by_name[n] for n in self.cv_subset]
        return self.by_category(CAT_CV)

    def excel_doc(self) -> Optional[CorpusDoc]:
        projects = self.by_category(CAT_PROJECTS)
        return projects[0] if projects else None

    def all_section_refs(self) -> List[SectionRef]:
        refs: List[SectionRef] = []
        for d in self.docs:
            refs.extend(d.sections)
        return refs

    # ── Excel as a structured table ─────────────────────
    def excel_table(self) -> Tuple[List[str], List[Dict[str, str]]]:
        """Return (headers, rows) for the project-history sheet.

        First non-empty row is treated as the header. Used by deterministic
        verification of budget/project aggregation questions.
        """
        ed = self.excel_doc()
        if ed is None:
            return [], []
        # excel_parser renders each row as "a | b | c"
        text = ed.full_text
        lines = [ln for ln in text.splitlines() if "|" in ln]
        if not lines:
            return [], []
        headers = [h.strip() for h in lines[0].split("|")]
        rows: List[Dict[str, str]] = []
        for ln in lines[1:]:
            cells = [c.strip() for c in ln.split("|")]
            if not any(cells):
                continue
            row = {headers[i]: (cells[i] if i < len(cells) else "")
                   for i in range(len(headers))}
            rows.append(row)
        return headers, rows

    # ── Lightweight corpus search ───────────────────────
    def search_term(self, term: str, categories: Optional[List[str]] = None
                    ) -> List[Tuple[str, SectionRef]]:
        """Find every section that mentions `term` (case-insensitive,
        word-ish). Returns (document, section) hits. Used for capability /
        counting verification and for grounding."""
        pattern = re.compile(re.escape(term.lower()))
        hits: List[Tuple[str, SectionRef]] = []
        for d in self.docs:
            if categories and d.category not in categories:
                continue
            for s in d.sections:
                if pattern.search(s.text.lower()):
                    hits.append((d.filename, s))
        return hits

    def documents_mentioning(self, term: str,
                             categories: Optional[List[str]] = None) -> List[str]:
        """Distinct documents that mention `term`."""
        seen: List[str] = []
        for fname, _ in self.search_term(term, categories):
            if fname not in seen:
                seen.append(fname)
        return seen
