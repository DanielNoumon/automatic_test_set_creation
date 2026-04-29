"""
Excel (.xlsx / .xls) parser: extracts structured sections from
spreadsheet files.

Each worksheet becomes a top-level Section. Rows are converted to
a readable text representation. Header rows (first row) are used
as column labels for subsequent rows.
"""
from pathlib import Path

from openpyxl import load_workbook

from .document import Document, Section
from .pdf_parser import _tag_section


def parse_excel(filepath: str) -> Document:
    """Parse an Excel file into a structured Document."""
    path = Path(filepath)
    wb = load_workbook(str(path), read_only=True, data_only=True)

    sections = []
    raw_parts = []

    for sheet_name in wb.sheetnames:
        ws = wb[sheet_name]
        rows = list(ws.iter_rows(values_only=True))

        if not rows:
            continue

        # First row as headers
        headers = [
            str(c).strip() if c is not None else ""
            for c in rows[0]
        ]

        # Build text representation of the sheet
        sheet_lines = []
        for row_idx, row in enumerate(rows[1:], start=2):
            cells = [
                str(c).strip() if c is not None else ""
                for c in row
            ]
            # Skip fully empty rows
            if not any(cells):
                continue
            # Format as "Header: Value" pairs
            pairs = []
            for h, v in zip(headers, cells):
                if v:
                    label = h if h else f"col_{headers.index(h)}"
                    pairs.append(f"{label}: {v}")
            if pairs:
                sheet_lines.append(" | ".join(pairs))

        if not sheet_lines:
            continue

        content = "\n".join(sheet_lines)
        header_line = " | ".join(
            h for h in headers if h
        )
        raw_parts.append(
            f"[Sheet: {sheet_name}]\n"
            f"{header_line}\n{content}"
        )

        # Split large sheets into chunks of ~30 rows
        # to create manageable sections
        chunk_size = 30
        for i in range(0, len(sheet_lines), chunk_size):
            chunk = sheet_lines[i:i + chunk_size]
            chunk_label = (
                f"{sheet_name}"
                if len(sheet_lines) <= chunk_size
                else f"{sheet_name} (rows {i+1}-"
                     f"{min(i+chunk_size, len(sheet_lines))})"
            )
            sections.append(Section(
                heading=chunk_label,
                content="\n".join(chunk),
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
