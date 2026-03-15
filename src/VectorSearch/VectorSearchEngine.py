import math
import os
import re
import string
from typing import Dict, List, Set, Tuple

import pymorphy3
import razdel
from nltk.corpus import stopwords
import nltk

from src.TfIdf import TfIdfUtils
from src.Tokenizer import TokenizerUtils

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')


class VectorSearchEngine:
    """
    Поисковый движок на основе векторного поиска с использованием TF-IDF.

    Каждый документ представлен как разреженный вектор TF-IDF по леммам.
    Запрос токенизируется, лемматизируется и также переводится в вектор,
    после чего документы ранжируются по косинусному сходству.
    """

    def __init__(self) -> None:
        self._doc_vectors: Dict[int, Dict[str, float]] = {}
        self._idf: Dict[str, float] = {}
        self._url_index: Dict[int, str] = {}
        self._morph = pymorphy3.MorphAnalyzer()
        self._stop_words: Set[str] = set(stopwords.words('russian'))
        self._stop_words.update(TokenizerUtils.get_custom_stopwords(TokenizerUtils.STOPWORDS_FILE))
        self._russian_pattern = re.compile(r'^[а-яА-ЯёЁ]+$')

    def load(
        self,
        tfidf_dir: str = TfIdfUtils.OUTPUT_DIR,
        n_docs: int = 100,
    ) -> None:
        """
        Загружает TF-IDF векторы всех документов из файлов lemmas_tfidf.txt.
        Одновременно строит глобальный словарь IDF.

        :param tfidf_dir: Каталог с результатами TF-IDF (output_tfidf/).
        :param n_docs: Количество документов.
        """
        for doc_id in range(1, n_docs + 1):
            path = os.path.join(tfidf_dir, str(doc_id), TfIdfUtils.LEMMAS_TFIDF_FILE)
            if not os.path.exists(path):
                continue

            vector: Dict[str, float] = {}
            with open(path, 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) != 3:
                        continue
                    lemma, idf_str, tfidf_str = parts
                    idf = float(idf_str)
                    tfidf = float(tfidf_str)
                    vector[lemma] = tfidf
                    if lemma not in self._idf:
                        self._idf[lemma] = idf

            self._doc_vectors[doc_id] = vector

        print(f"Загружено {len(self._doc_vectors)} документов, {len(self._idf)} уникальных лемм.")

    def load_url_index(self, index_file: str) -> None:
        """
        Загружает маппинг номер документа → URL из файла index.txt.

        :param index_file: Путь к файлу index.txt.
        """
        self._url_index.clear()
        with open(index_file, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split(maxsplit=1)
                if len(parts) == 2:
                    doc_id = int(parts[0])
                    url = parts[1]
                    self._url_index[doc_id] = url

    def search(self, query: str, top_n: int = 10) -> List[Tuple[int, str, float]]:
        """
        Выполняет векторный поиск по запросу.

        Шаги:
        1. Токенизирует и лемматизирует запрос.
        2. Строит TF-IDF вектор запроса на основе глобального IDF.
        3. Вычисляет косинусное сходство с каждым документом.
        4. Возвращает топ-N результатов в виде (doc_id, url, score).

        :param query: Текстовый запрос пользователя.
        :param top_n: Количество возвращаемых результатов.
        :returns: Список кортежей (doc_id, url, score), отсортированных по убыванию score.
        """
        query_lemmas = self._lemmatize_query(query)
        if not query_lemmas:
            return []

        query_vector = self._build_query_vector(query_lemmas)
        if not query_vector:
            return []

        results: List[Tuple[int, str, float]] = []
        query_norm = math.sqrt(sum(v * v for v in query_vector.values()))

        for doc_id, doc_vector in self._doc_vectors.items():
            score = self._cosine_similarity(query_vector, doc_vector, query_norm)
            if score > 0:
                url = self._url_index.get(doc_id, f"doc_{doc_id}")
                results.append((doc_id, url, score))

        results.sort(key=lambda x: x[2], reverse=True)
        return results[:top_n]

    def _lemmatize_query(self, query: str) -> Dict[str, int]:
        """
        Токенизирует запрос, фильтрует стоп-слова и лемматизирует токены.

        :param query: Сырая строка запроса.
        :returns: Словарь {лемма: количество вхождений в запросе}.
        """
        lemma_counts: Dict[str, int] = {}
        query_lower = query.lower()

        for tok in razdel.tokenize(query_lower):
            token = tok.text.strip(string.punctuation)
            candidates = token.split('-') if '-' in token else [token]
            for candidate in candidates:
                if not self._is_valid_token(candidate):
                    continue
                parsed = self._morph.parse(candidate)
                if not parsed:
                    continue
                lemma = parsed[0].normal_form
                lemma_counts[lemma] = lemma_counts.get(lemma, 0) + 1

        return lemma_counts

    def _build_query_vector(self, lemma_counts: Dict[str, int]) -> Dict[str, float]:
        """
        Строит TF-IDF вектор запроса.
        TF(лемма) = кол-во вхождений леммы / общее кол-во лемм в запросе.
        IDF берётся из глобального словаря, построенного по корпусу.
        Леммы, отсутствующие в корпусе, игнорируются.

        :param lemma_counts: Счётчики лемм в запросе.
        :returns: TF-IDF вектор запроса {лемма: tf-idf}.
        """
        total = sum(lemma_counts.values())
        if total == 0:
            return {}

        vector: Dict[str, float] = {}
        for lemma, count in lemma_counts.items():
            idf = self._idf.get(lemma)
            if idf is None:
                continue
            tf = count / total
            vector[lemma] = tf * idf

        return vector

    def _cosine_similarity(
        self,
        query_vector: Dict[str, float],
        doc_vector: Dict[str, float],
        query_norm: float,
    ) -> float:
        """
        Вычисляет косинусное сходство между вектором запроса и вектором документа.

        :param query_vector: TF-IDF вектор запроса.
        :param doc_vector: TF-IDF вектор документа.
        :param query_norm: Норма вектора запроса (вычисляется заранее для эффективности).
        :returns: Косинусное сходство в диапазоне [0, 1].
        """
        if query_norm == 0:
            return 0.0

        dot_product = sum(
            query_vector[lemma] * doc_vector[lemma]
            for lemma in query_vector
            if lemma in doc_vector
        )
        if dot_product == 0:
            return 0.0

        doc_norm = math.sqrt(sum(v * v for v in doc_vector.values()))
        if doc_norm == 0:
            return 0.0

        return dot_product / (query_norm * doc_norm)

    def _is_valid_token(self, token: str) -> bool:
        """
        Проверяет допустимость токена: только кириллица, длина > 1, не стоп-слово.

        :param token: Токен для проверки.
        :returns: True если токен допустим.
        """
        if len(token) <= 1:
            return False
        if not self._russian_pattern.match(token):
            return False
        return token not in self._stop_words
