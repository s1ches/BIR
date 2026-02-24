import os
from typing import Dict, Set

from Tokenizer import TokenizerUtils
from Tokenizer.Lemmatizer import Lemmatizer
from Tokenizer.Tokenizer import Tokenizer


class HtmlProcessor:    
    def __init__(self, processing_dir: str = "output", output_dir: str = "output_tokenized", tokens_file: str = "tokens.txt", lemmas_file: str = "lemmas.txt"):
        self.processing_dir = processing_dir
        self.tokens_file = tokens_file
        self.lemmas_file = lemmas_file
        self.output_dir = output_dir
        self.tokenizer = Tokenizer()
        self.lemmatizer = Lemmatizer()
    
    def process_document(self, processing_file_index: str) -> None:
        all_tokens = set()
            
        with open(os.path.join(self.processing_dir, processing_file_index+'.txt'), 
                    'r', encoding='utf-8') as f:
            html = f.read()
            
            text = self.tokenizer.extract_text_from_html(html)
            tokens = self.tokenizer.tokenize(text)
            all_tokens.update(tokens)
        
        lemma_dict = self.lemmatizer.lemmatize_tokens(all_tokens)
        
        self._save_results(processing_file_index, all_tokens, lemma_dict)
    
    def _save_results(self, result_dir: str, tokens: Set[str], lemma_dict: Dict[str, Set[str]]): 
        print(f'Обработан файл с индексом {result_dir}')  
        if not os.path.exists(os.path.join(self.output_dir, result_dir)):
            os.makedirs(os.path.join(self.output_dir, result_dir))  

        with open(os.path.join(self.output_dir, result_dir, TokenizerUtils.TOKENS_FILE), 'w', encoding='utf-8') as f:
            for token in sorted(tokens):
                f.write(f"{token}\n")
        
        with open(os.path.join(self.output_dir, result_dir, TokenizerUtils.LEMMAS_FILE), 'w', encoding='utf-8') as f:
            for lemma, token_set in sorted(lemma_dict.items()):
                tokens_str = ' '.join(sorted(token_set))
                f.write(f"{lemma} {tokens_str}\n")
        