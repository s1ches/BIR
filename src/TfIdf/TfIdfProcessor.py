import math
import os
import re
import string
from typing import Dict, Set, List, Tuple

import razdel
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
import nltk

from Tokenizer import TokenizerUtils

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')


class TfIdfProcessor:
    """
    Вычисляет TF, IDF и TF-IDF для токенов и лемм каждого документа.

    TF(t, d)      = кол-во вхождений t в d / всего валидных токенов в d
    TF(l, d)      = сумма вхождений токенов леммы l в d / всего токенов в d
    IDF(t)        = log(N / df(t))
    TF-IDF(t, d)  = TF(t, d) * IDF(t)
    """

    def __init__(self) -> None:
        self._stop_words: Set[str] = set(stopwords.words('russian'))
        self._stop_words.update(TokenizerUtils.get_custom_stopwords(TokenizerUtils.STOPWORDS_FILE))
        self._russian_pattern = re.compile(r'^[а-яА-ЯёЁ]+$')

    def process_all(
        self,
        crawl_dir: str,
        tokenized_dir: str,
        output_dir: str,
        n_docs: int,
    ) -> None:
        """
        Запускает полный цикл вычисления TF-IDF по всем документам.

        :param crawl_dir: Каталог с HTML-файлами выкачки (output/).
        :param tokenized_dir: Каталог с результатами токенизации (output_tokenized/).
        :param output_dir: Каталог для сохранения результатов (output_tfidf/).
        :param n_docs: Количество документов.
        """
        all_term_counts: List[Dict[str, int]] = []
        all_lemma_maps: List[Dict[str, Set[str]]] = []

        print("Подсчёт вхождений терминов в документах...")
        for doc_id in range(1, n_docs + 1):
            html_path = os.path.join(crawl_dir, f"{doc_id}.txt")
            vocab = self._load_tokens(tokenized_dir, doc_id)
            counts = self._count_terms(html_path, vocab)
            all_term_counts.append(counts)

            lemma_map = self._load_lemmas(tokenized_dir, doc_id)
            all_lemma_maps.append(lemma_map)

        term_df = self._compute_df(all_term_counts)
        lemma_df = self._compute_lemma_df(all_term_counts, all_lemma_maps)

        print("Сохранение TF-IDF файлов...")
        for doc_id in range(1, n_docs + 1):
            idx = doc_id - 1
            counts = all_term_counts[idx]
            lemma_map = all_lemma_maps[idx]
            total = sum(counts.values()) or 1

            tokens_data = self._build_tokens_tfidf(counts, total, term_df, n_docs)
            lemmas_data = self._build_lemmas_tfidf(counts, total, lemma_map, lemma_df, n_docs)

            self._save(output_dir, doc_id, tokens_data, 'tokens_tfidf.txt')
            self._save(output_dir, doc_id, lemmas_data, 'lemmas_tfidf.txt')

        print(f"TF-IDF сохранён в {output_dir}/")

    def _count_terms(self, html_path: str, vocab: Set[str]) -> Dict[str, int]:
        """
        Читает HTML-файл, извлекает текст и подсчитывает количество вхождений
        каждого слова из vocab (с теми же фильтрами, что и Tokenizer).

        :param html_path: Путь к HTML-файлу.
        :param vocab: Множество допустимых токенов для данного документа.
        :returns: Словарь {токен: кол-во вхождений}.
        """
        with open(html_path, 'r', encoding='utf-8') as f:
            html = f.read()

        soup = BeautifulSoup(html, 'lxml')
        for element in soup(["script", "style", "meta", "noscript", "nav", "header", "footer"]):
            element.decompose()

        text = soup.get_text(separator=' ', strip=True)
        text = re.sub(r'&[a-z]+;', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip().lower()

        counts: Dict[str, int] = {}
        for tok in razdel.tokenize(text):
            token = tok.text.strip(string.punctuation)
            candidates: List[str] = token.split('-') if '-' in token else [token]
            for candidate in candidates:
                if candidate and self._is_valid_token(candidate) and candidate in vocab:
                    counts[candidate] = counts.get(candidate, 0) + 1

        return counts

    def _is_valid_token(self, token: str) -> bool:
        """
        Проверяет, является ли токен допустимым (кириллица, длина > 1, не стоп-слово).

        :param token: Токен для проверки.
        :returns: True если токен допустим.
        """
        if len(token) <= 1:
            return False
        if not self._russian_pattern.match(token):
            return False
        return token not in self._stop_words

    def _load_tokens(self, tokenized_dir: str, doc_id: int) -> Set[str]:
        """
        Загружает множество токенов документа из tokens.txt.

        :param tokenized_dir: Каталог с результатами токенизации.
        :param doc_id: Номер документа.
        :returns: Множество токенов.
        """
        path = os.path.join(tokenized_dir, str(doc_id), TokenizerUtils.TOKENS_FILE)
        tokens: Set[str] = set()
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                token = line.strip()
                if token:
                    tokens.add(token)
        return tokens

    def _load_lemmas(self, tokenized_dir: str, doc_id: int) -> Dict[str, Set[str]]:
        """
        Загружает маппинг лемма → токены из lemmas.txt.

        :param tokenized_dir: Каталог с результатами токенизации.
        :param doc_id: Номер документа.
        :returns: Словарь {лемма: множество токенов}.
        """
        path = os.path.join(tokenized_dir, str(doc_id), TokenizerUtils.LEMMAS_FILE)
        lemma_map: Dict[str, Set[str]] = {}
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    lemma, *tokens = parts
                    lemma_map[lemma] = set(tokens)
        return lemma_map

    def _compute_df(self, all_counts: List[Dict[str, int]]) -> Dict[str, int]:
        """
        Вычисляет document frequency (DF) для каждого токена.

        :param all_counts: Список счётчиков вхождений по каждому документу.
        :returns: Словарь {токен: кол-во документов, содержащих токен}.
        """
        df: Dict[str, int] = {}
        for counts in all_counts:
            for term in counts:
                df[term] = df.get(term, 0) + 1
        return df

    def _compute_lemma_df(
        self,
        all_counts: List[Dict[str, int]],
        all_lemma_maps: List[Dict[str, Set[str]]],
    ) -> Dict[str, int]:
        """
        Вычисляет document frequency для каждой леммы.
        Лемма считается присутствующей в документе, если хотя бы один
        её токен встречается в документе.

        :param all_counts: Список счётчиков вхождений по каждому документу.
        :param all_lemma_maps: Список маппингов лемма→токены по каждому документу.
        :returns: Словарь {лемма: кол-во документов, содержащих лемму}.
        """
        df: Dict[str, int] = {}
        for counts, lemma_map in zip(all_counts, all_lemma_maps):
            for lemma, tokens in lemma_map.items():
                if any(t in counts for t in tokens):
                    df[lemma] = df.get(lemma, 0) + 1
        return df

    def _build_tokens_tfidf(
        self,
        counts: Dict[str, int],
        total: int,
        df: Dict[str, int],
        n_docs: int,
    ) -> List[Tuple[str, float, float]]:
        """
        Формирует список (токен, idf, tf-idf) для одного документа.

        :param counts: Счётчики вхождений токенов в данном документе.
        :param total: Общее кол-во валидных токенов в документе.
        :param df: Глобальный document frequency.
        :param n_docs: Общее кол-во документов.
        :returns: Список кортежей (токен, idf, tf-idf).
        """
        result: List[Tuple[str, float, float]] = []
        for term, count in sorted(counts.items()):
            tf = count / total
            idf = math.log(n_docs / df[term])
            result.append((term, idf, tf * idf))
        return result

    def _build_lemmas_tfidf(
        self,
        counts: Dict[str, int],
        total: int,
        lemma_map: Dict[str, Set[str]],
        lemma_df: Dict[str, int],
        n_docs: int,
    ) -> List[Tuple[str, float, float]]:
        """
        Формирует список (лемма, idf, tf-idf) для одного документа.
        TF леммы = сумма вхождений всех её токенов / всего токенов в документе.

        :param counts: Счётчики вхождений токенов в данном документе.
        :param total: Общее кол-во валидных токенов в документе.
        :param lemma_map: Маппинг лемма→токены для данного документа.
        :param lemma_df: Глобальный document frequency по леммам.
        :param n_docs: Общее кол-во документов.
        :returns: Список кортежей (лемма, idf, tf-idf).
        """
        result: List[Tuple[str, float, float]] = []
        for lemma, tokens in sorted(lemma_map.items()):
            lemma_count = sum(counts.get(t, 0) for t in tokens)
            if lemma_count == 0:
                continue
            tf = lemma_count / total
            doc_freq = lemma_df.get(lemma, 1)
            idf = math.log(n_docs / doc_freq)
            result.append((lemma, idf, tf * idf))
        return result

    def _save(
        self,
        output_dir: str,
        doc_id: int,
        data: List[Tuple[str, float, float]],
        filename: str,
    ) -> None:
        """
        Сохраняет список (термин, idf, tf-idf) в файл.
        Формат строки: <термин> <idf> <tf-idf>

        :param output_dir: Корневой каталог для результатов.
        :param doc_id: Номер документа.
        :param data: Данные для записи.
        :param filename: Имя файла.
        """
        doc_dir = os.path.join(output_dir, str(doc_id))
        os.makedirs(doc_dir, exist_ok=True)
        path = os.path.join(doc_dir, filename)
        with open(path, 'w', encoding='utf-8') as f:
            for term, idf, tfidf in data:
                f.write(f"{term} {idf:.6f} {tfidf:.6f}\n")
