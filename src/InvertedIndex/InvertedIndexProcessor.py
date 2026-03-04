import os
from typing import Set, Dict
from InvertedIndex import BooleanQueryParser
from Tokenizer import TokenizerUtils


class InvertedIndexProcessor:
    def __init__(self) -> None:
        self.index: Dict[str, Set[str]] = {}
        self.all_docs: Set[str] = set()

    def build(self, tokenized_dir: str) -> None:
        if not os.path.isdir(tokenized_dir):
            raise FileNotFoundError(f"tokenized directory '{tokenized_dir}' not found")

        for entry in sorted(os.listdir(tokenized_dir), key=lambda v: int(v) if v.isdigit() else v):
            subdir = os.path.join(tokenized_dir, entry)
            if not os.path.isdir(subdir):
                continue
            lemmas_file = os.path.join(subdir, TokenizerUtils.LEMMAS_FILE)
            if not os.path.isfile(lemmas_file):
                continue
            self.all_docs.add(entry)
            with open(lemmas_file, 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split()
                    if not parts:
                        continue
                    lemma = parts[0].lower()
                    self.index.setdefault(lemma, set()).add(entry)

    def save(self, filename: str) -> None:
        with open(filename, 'w', encoding='utf-8') as f:
            for term in sorted(self.index):
                docs = sorted(self.index[term], key=int)
                f.write(term + " " + " ".join(docs) + "\n")

    def load(self, filename: str) -> None:
        self.index.clear()
        self.all_docs.clear()
        with open(filename, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split()
                if not parts:
                    continue
                term = parts[0]
                docs = set(parts[1:])
                self.index[term] = docs
                self.all_docs |= docs

    def query(self, query_string: str) -> Set[str]:
        parser = BooleanQueryParser(self)
        return parser.parse(query_string)

