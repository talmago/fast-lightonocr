"""Pure Python OCR post-processing helpers."""

from __future__ import annotations

from dataclasses import dataclass
import re

from bs4 import BeautifulSoup
from bs4.element import Tag


@dataclass(frozen=True)
class TableCell:
    """A parsed OCR table cell."""

    text: str
    header: bool = False
    colspan: int = 1
    rowspan: int = 1


@dataclass(frozen=True)
class Table:
    """A parsed OCR table."""

    rows: list[list[TableCell]]
    html: str

    @property
    def text_rows(self) -> list[list[str]]:
        """Return table rows as plain cell text."""

        return [[cell.text for cell in row] for row in self.rows]


@dataclass(frozen=True)
class PostProcessResult:
    """Post-processed OCR text and extracted tables."""

    clean_text: str
    tables: list[Table]


def post_process_text(text: str) -> PostProcessResult:
    """Strip HTML tags from OCR text and extract HTML tables."""

    soup = BeautifulSoup(text, "html.parser")
    tables = [_parse_table(table) for table in soup.find_all("table")]
    return PostProcessResult(
        clean_text=_normalize_text(soup.get_text(separator="\n")),
        tables=tables,
    )


def _parse_table(table: Tag) -> Table:
    rows = []
    for row in table.find_all("tr"):
        cells = [_parse_cell(cell) for cell in row.find_all(["th", "td"], recursive=False)]
        if cells:
            rows.append(cells)

    return Table(rows=rows, html=str(table))


def _parse_cell(cell: Tag) -> TableCell:
    return TableCell(
        text=_normalize_inline_text(cell.get_text(separator=" ")),
        header=cell.name == "th",
        colspan=_positive_int(cell.get("colspan"), default=1),
        rowspan=_positive_int(cell.get("rowspan"), default=1),
    )


def _normalize_text(text: str) -> str:
    lines = [_normalize_inline_text(line) for line in text.splitlines()]
    output = []
    pending_blank = False

    for line in lines:
        if line:
            if pending_blank and output:
                output.append("")
            output.append(line)
            pending_blank = False
        elif output:
            pending_blank = True

    return "\n".join(output).strip()


def _normalize_inline_text(text: str) -> str:
    return re.sub(r"[ \t]+", " ", text).strip()


def _positive_int(value: object, *, default: int) -> int:
    try:
        parsed = int(str(value))
    except (TypeError, ValueError):
        return default

    return parsed if parsed > 0 else default
