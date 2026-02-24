import os
from typing import Dict, Set

from Crawler import CrawlerUtils
from Tokenizer.Lemmatizer import Lemmatizer
from Tokenizer.Tokenizer import Tokenizer


class HtmlProcessor:    
    def __init__(self, processing_dir: str = "output", tokens_file: str = "tokens.txt", lemmas_file: str = "lemmas.txt"):
        self.processing_dir = processing_dir
        self.tokens_file = tokens_file
        self.lemmas_file = lemmas_file
        self.tokenizer = Tokenizer()
        self.lemmatizer = Lemmatizer()
    
    def process_documents(self) -> None:
        all_tokens = set()
        
        files = [f for f in os.listdir(self.processing_dir) 
                if f.endswith('.txt')]
        
        print(f"Найдено файлов: {len(files)}")
        
        for i, filename in enumerate(files, 1):
            print(f"Обработка {i}/{len(files)}: {filename}")
            
            with open(os.path.join(self.processing_dir, filename), 
                        'r', encoding='utf-8') as f:
                html = f.read()
            
            text = self.tokenizer.extract_text_from_html(html)
            tokens = self.tokenizer.tokenize(text)
            all_tokens.update(tokens)
                
        
        lemma_dict = self.lemmatizer.lemmatize_tokens(all_tokens)
        
        self._save_results(all_tokens, lemma_dict)
    
    def _save_results(self, tokens: Set[str], lemma_dict: Dict[str, Set[str]]):        
        with open(self.tokens_file, 'w', encoding='utf-8') as f:
            for token in sorted(tokens):
                f.write(f"{token}\n")
        print(f"\nТокены сохранены в {self.tokens_file} ({len(tokens)} шт.)")
        
        with open(self.lemmas_file, 'w', encoding='utf-8') as f:
            for lemma, token_set in sorted(lemma_dict.items()):
                tokens_str = ' '.join(sorted(token_set))
                f.write(f"{lemma} {tokens_str}\n")
        print(f"Леммы сохранены в {self.lemmas_file} ({len(lemma_dict)} шт.)")
        