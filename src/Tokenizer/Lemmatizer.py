from typing import Dict, Set
import pymorphy3


class Lemmatizer:    
    def __init__(self):
        self.morph = pymorphy3.MorphAnalyzer()
    
    def lemmatize_tokens(self, tokens: Set[str]) -> Dict[str, Set[str]]:
        lemma_dict = {}
        
        for token in sorted(tokens):
            try:
                lemma = self.morph.parse(token)[0].normal_form
                
                if lemma not in lemma_dict:
                    lemma_dict[lemma] = set()
                lemma_dict[lemma].add(token)
            except:
                if token not in lemma_dict:
                    lemma_dict[token] = set()
                lemma_dict[token].add(token)
        
        return lemma_dict


