"""
Excel (.xlsx / .xls) parser: converts spreadsheets to plain text.

Each worksheet becomes a single Section — no chunking, no header
detection heuristics.  Every row is rendered as a pipe-separated
line so the full sheet is preserved as-is for the LLM context
window during Q+A generation.
"""
from pathlib import Path

from openpyxl import load_workbook

from .document import Document, Section
from .pdf_parser import _tag_section


def parse_excel(filepath: str) -> Document:
    """Parse an Excel file into a structured Document.

    One Section per worksheet.  All rows are kept intact
    so the LLM receives the complete sheet as context.
    """
    path = Path(filepath)
    wb = load_workbook(
        str(path), read_only=True, data_only=True,
    )

    sections = []
    raw_parts = []

    for sheet_name in wb.sheetnames:
        ws = wb[sheet_name]
        rows = list(ws.iter_rows(values_only=True))
        if not rows:
            continue

        # Convert every row to a pipe-separated text line
        lines = []
        for row in rows:
            cells = [
                str(c).strip() if c is not None else ""
                for c in row
            ]
            # Skip fully empty rows
            if not any(cells):
                continue
            lines.append(" | ".join(cells))

        if not lines:
            continue

        content = "\n".join(lines)
        raw_parts.append(
            f"[Sheet: {sheet_name}]\n{content}"
        )

        sections.append(Section(
            heading=sheet_name.strip(),
            content=content,
            level=0,
            page_start=1,
            page_end=1,
        ))

    wb.close()

    raw_text = "\n\n".join(raw_parts)

    for sec in sections:
        _tag_section(sec)

    return Document(
        filename=path.name,
        raw_text=raw_text,
        sections=sections,
        page_count=1,
        doc_type="excel",
    )
