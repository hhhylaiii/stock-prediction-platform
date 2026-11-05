"""Table cells of the plain HTML pages MOPS serves."""

from html.parser import HTMLParser


class TableCells(HTMLParser):
    """Rows of cell text, each tagged with its table; a cell ends at either </td> or </th>."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.rows: list[tuple[int, list[str]]] = []
        self._tables: list[int] = []
        self._opened = 0
        self._cell: list[str] | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "table":
            self._opened += 1
            self._tables.append(self._opened)
        elif tag == "tr":
            self.rows.append((self._tables[-1] if self._tables else 0, []))
        elif tag in ("td", "th") and self.rows:
            self._close()
            self._cell = []

    def handle_endtag(self, tag: str) -> None:
        if tag in ("td", "th"):
            self._close()
        elif tag == "table" and self._tables:
            self._tables.pop()

    def handle_data(self, data: str) -> None:
        if self._cell is not None:
            self._cell.append(data)

    def _close(self) -> None:
        if self._cell is not None:
            self.rows[-1][1].append("".join(self._cell).strip())
            self._cell = None
