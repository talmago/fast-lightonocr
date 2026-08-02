"""Parse OCR Markdown containing embedded HTML tables."""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Literal, TypeAlias

from bs4 import BeautifulSoup
from bs4.element import Tag
from tabulate import tabulate

# ============================================================================
# Configuration
# ============================================================================

TableFormat: TypeAlias = Literal[
    "grid",
    "github",
    "simple",
    "plain",
    "pipe",
    "rounded_grid",
    "fancy_grid",
]


@dataclass(frozen=True)
class ParserOptions:
    """Controls parsing and rendering of OCR documents."""

    #: Tabulate table format.
    table_format: TableFormat = "grid"

    #: Separator inserted between rendered document blocks.
    block_separator: str = "\n\n"

    #: Normalize whitespace in Markdown text.
    normalize_whitespace: bool = True


# ============================================================================
# Tables
# ============================================================================


@dataclass(frozen=True)
class TableCell:
    """A parsed HTML table cell."""

    text: str
    header: bool = False
    colspan: int = 1
    rowspan: int = 1


@dataclass(frozen=True)
class Table:
    """A parsed HTML table."""

    rows: list[list[TableCell]]
    html: str

    @property
    def text_rows(self) -> list[list[str]]:
        return [[cell.text for cell in row] for row in self.rows]

    def render(self, table_format: TableFormat) -> str:
        """Render the table using tabulate."""

        headers, rows = self._tabular_data()

        if not headers and not rows:
            return ""

        return tabulate(
            rows,
            headers=headers,
            tablefmt=table_format,
            disable_numparse=True,
        )

    def _tabular_data(self) -> tuple[list[str], list[list[str]]]:
        rows = self.text_rows

        if not rows:
            return [], []

        width = max(len(row) for row in rows)

        rows = [row + [""] * (width - len(row)) for row in rows]

        header_count = 0

        for row in self.rows:
            if row and all(cell.header for cell in row):
                header_count += 1
            else:
                break

        if header_count == 0:
            return [], rows

        header_rows = rows[:header_count]
        body_rows = rows[header_count:]

        headers = [
            "\n".join(row[column] for row in header_rows if row[column])
            for column in range(width)
        ]

        return headers, body_rows


# ============================================================================
# Blocks
# ============================================================================


class DocumentBlock(ABC):
    """Base class for parsed document blocks."""

    @abstractmethod
    def render(self, options: ParserOptions) -> str: ...


@dataclass(frozen=True)
class TextBlock(DocumentBlock):
    """Markdown text."""

    text: str

    def render(self, options: ParserOptions) -> str:
        return self.text


@dataclass(frozen=True)
class TableBlock(DocumentBlock):
    """Structured table."""

    table: Table

    def render(self, options: ParserOptions) -> str:
        return self.table.render(options.table_format)


# ============================================================================
# Parsed document
# ============================================================================


@dataclass(frozen=True)
class Document:
    """Parsed OCR document."""

    blocks: list[DocumentBlock]
    options: ParserOptions

    @property
    def text(self) -> str:
        """Render the document."""

        return self.render()

    @property
    def tables(self) -> list[Table]:
        return [block.table for block in self.blocks if isinstance(block, TableBlock)]

    def render(self) -> str:
        rendered = [block.render(self.options).strip() for block in self.blocks]

        return self.options.block_separator.join(block for block in rendered if block)


# ============================================================================
# Parsing
# ============================================================================


_TABLE_RE = re.compile(
    r"<table\b.*?</table>",
    flags=re.DOTALL | re.IGNORECASE,
)


def parse_document(
    text: str,
    options: ParserOptions | None = None,
) -> Document:
    """Parse OCR Markdown into a structured document."""

    if options is None:
        options = ParserOptions()

    blocks: list[DocumentBlock] = []

    position = 0

    for match in _TABLE_RE.finditer(text):
        markdown = text[position : match.start()]

        if options.normalize_whitespace:
            markdown = _normalize_text(markdown)

        if markdown:
            blocks.append(TextBlock(markdown))

        soup = BeautifulSoup(match.group(0), "html.parser")
        table = soup.find("table")

        if isinstance(table, Tag):
            blocks.append(
                TableBlock(
                    _parse_table(table),
                )
            )

        position = match.end()

    markdown = text[position:]

    if options.normalize_whitespace:
        markdown = _normalize_text(markdown)

    if markdown:
        blocks.append(TextBlock(markdown))

    return Document(
        blocks=blocks,
        options=options,
    )


# ============================================================================
# HTML parsing
# ============================================================================


def _parse_table(table: Tag) -> Table:
    rows: list[list[TableCell]] = []

    for row in table.find_all("tr"):
        cells = [
            _parse_cell(cell) for cell in row.find_all(["th", "td"], recursive=False)
        ]

        if cells:
            rows.append(cells)

    return Table(
        rows=rows,
        html=str(table),
    )


def _parse_cell(cell: Tag) -> TableCell:
    return TableCell(
        text=_normalize_inline_text(
            cell.get_text(separator=" "),
        ),
        header=cell.name == "th",
        colspan=_positive_int(
            cell.get("colspan"),
            default=1,
        ),
        rowspan=_positive_int(
            cell.get("rowspan"),
            default=1,
        ),
    )


# ============================================================================
# Utilities
# ============================================================================


def _normalize_text(text: str) -> str:
    lines = [_normalize_inline_text(line) for line in text.splitlines()]

    output: list[str] = []
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
        value = int(str(value))
    except (TypeError, ValueError):
        return default

    return value if value > 0 else default
