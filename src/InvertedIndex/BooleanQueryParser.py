import re
from typing import Set


class BooleanQueryParser:
    _token_re = re.compile(r"\(|\)|AND|OR|NOT|[^()\s]+", flags=re.IGNORECASE)

    def __init__(self, index: "InvertedIndexProcessor") -> None:
        self.index = index
        self.tokens: list[str] = []
        self.pos = 0

    def tokenize(self, query: str) -> list[str]:
        return [tok for tok in self._token_re.findall(query) if tok.strip()]

    def peek(self) -> str | None:
        if self.pos < len(self.tokens):
            return self.tokens[self.pos].upper()
        return None

    def consume(self, expected: str | None = None) -> str:
        if self.pos >= len(self.tokens):
            raise ValueError("Unexpected end of query")
        tok = self.tokens[self.pos]
        self.pos += 1
        if expected and tok.upper() != expected:
            raise ValueError(f"Expected '{expected}', found '{tok}'")
        return tok

    def parse(self, query: str) -> Set[str]:
        self.tokens = self.tokenize(query)
        self.pos = 0
        result = self._parse_expr()
        if self.pos != len(self.tokens):
            raise ValueError("Extra tokens at end of query")
        return result

    def _parse_expr(self) -> Set[str]:
        result = self._parse_term()
        while self.peek() == 'OR':
            self.consume('OR')
            right = self._parse_term()
            result = result | right
        return result

    def _parse_term(self) -> Set[str]:
        result = self._parse_factor()
        while self.peek() == 'AND':
            self.consume('AND')
            right = self._parse_factor()
            result = result & right
        return result

    def _parse_factor(self) -> Set[str]:
        tok = self.peek()
        if tok == 'NOT':
            self.consume('NOT')
            return self.index.all_docs - self._parse_factor()
        elif tok == '(':
            self.consume('(')
            result = self._parse_expr()
            self.consume(')')
            return result
        else:
            word = self.consume()
            return set(self.index.index.get(word.lower(), []))