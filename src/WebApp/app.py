import os
import sys
from urllib.parse import unquote

from src.Crawler import CrawlerUtils
from src.TfIdf import TfIdfUtils
from src.VectorSearch import VectorSearchUtils
from src.VectorSearch.VectorSearchEngine import VectorSearchEngine

# Добавляем src/ в путь импорта, чтобы работали модули VectorSearch, TfIdf, Crawler и т.д.
_SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from flask import Flask, render_template, request

app = Flask(__name__, template_folder="templates")

# Вычисляем пути относительно src/
_TFIDF_DIR = os.path.join(_SRC_DIR, TfIdfUtils.OUTPUT_DIR)
_INDEX_FILE = os.path.join(_SRC_DIR, CrawlerUtils.INDEX_FILE)

_engine: VectorSearchEngine = VectorSearchEngine()
_engine.load(_TFIDF_DIR, CrawlerUtils.MAX_PAGES)
_engine.load_url_index(_INDEX_FILE)


def _page_title(url: str) -> str:
    """
    Извлекает читаемый заголовок страницы из Wikipedia URL.
    Например: https://ru.wikipedia.org/wiki/Кит → «Кит»
    """
    segment = url.rstrip("/").split("/")[-1]
    return unquote(segment).replace("_", " ")


@app.route("/", methods=["GET", "POST"])
def index():
    """
    GET  /  — отображает форму поиска.
    POST /  — выполняет векторный поиск и отображает топ-10 результатов.
    """
    query = ""
    results = []
    searched = False

    if request.method == "POST":
        query = request.form.get("query", "").strip()
        searched = True
        if query:
            raw = _engine.search(query, top_n=VectorSearchUtils.TOP_N)
            results = []
            for rank, (doc_id, url, score) in enumerate(raw, start=1):
                has_url = url.startswith("http")
                results.append(
                    {
                        "rank": rank,
                        "doc_id": doc_id,
                        "title": _page_title(url) if has_url else None,
                        "url": url if has_url else None,
                        "score": round(score, 6),
                    }
                )

    return render_template("index.html", query=query, results=results, searched=searched)


if __name__ == "__main__":
    app.run(debug=True, port=5000)
