import string
import razdel
from typing import Set
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
import nltk
import re

from Tokenizer import TokenizerUtils


try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

class Tokenizer:    
    def __init__(self):
        self.stop_words = set(stopwords.words('russian'))
        custom_stopwords = TokenizerUtils.get_custom_stopwords(TokenizerUtils.STOPWORDS_FILE)
        print(custom_stopwords)
        self.stop_words.update(TokenizerUtils.get_custom_stopwords())
        self.russian_pattern = re.compile(r'^[а-яА-ЯёЁ]+$')
    
    def extract_text_from_html(self, html_content: str) -> str:
        soup = BeautifulSoup(html_content, 'lxml')
        
        for element in soup(["script", "style", "meta", "noscript", "nav", "header", "footer"]):
            element.decompose()
        
        text = soup.get_text(separator=' ', strip=True)
        
        text = re.sub(r'&[a-z]+;', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()
    
    def is_valid_token(self, token: str) -> bool:
        if len(token) <= 1:
            return False
            
        if not self.russian_pattern.match(token):
            return False
            
        if token.lower() not in self.stop_words:
            return True
            
        return False
    
    def tokenize(self, text: str) -> Set[str]:        
        tokens = [t.text for t in razdel.tokenize(text.lower())]
        
        valid_tokens = set()
        for token in tokens:
            token = token.strip(string.punctuation)
            
            if '-' in token:
                for part in token.split('-'):
                    if part and self.is_valid_token(part):
                        valid_tokens.add(part)
            else:
                if token and self.is_valid_token(token):
                    valid_tokens.add(token)
        
        return valid_tokens
