import os
from Crawler import CrawlerUtils
from Crawler.WikiCrawler import WikiCrawler
from Tokenizer import TokenizerUtils
from Tokenizer.HtmlProcessor import HtmlProcessor
from InvertedIndex.InvertedIndexProcessor import InvertedIndexProcessor
from TfIdf import TfIdfUtils
from TfIdf.TfIdfProcessor import TfIdfProcessor
from VectorSearch import VectorSearchUtils
from VectorSearch.VectorSearchEngine import VectorSearchEngine

def crawl_required(output_dir = CrawlerUtils.OUTPUT_DIR, index_file = CrawlerUtils.INDEX_FILE, min_files=100):
    if not os.path.exists(output_dir):
        return True

    if not os.path.exists(index_file):
        return True

    txt_files = [
        f for f in os.listdir(output_dir)
        if f.endswith(".txt")
    ]

    if len(txt_files) < min_files:
        return True

    return False

def tfidf_required(output_dir: str = TfIdfUtils.OUTPUT_DIR, tokens_file: str = TfIdfUtils.TOKENS_TFIDF_FILE, lemmas_file: str = TfIdfUtils.LEMMAS_TFIDF_FILE, n_docs: int = 100) -> bool:
    for i in range(1, n_docs + 1):
        if not os.path.exists(os.path.join(output_dir, str(i), tokens_file)) \
                or not os.path.exists(os.path.join(output_dir, str(i), lemmas_file)):
            return True
    return False

def tokenization_required(output_dir = TokenizerUtils.OUTPUT_DIR, tokens_file = TokenizerUtils.TOKENS_FILE, lemmas_file = TokenizerUtils.LEMMAS_FILE, min_files=200):
    for i in range(1, min_files//2+1):
        if not os.path.exists(os.path.join(output_dir, str(i), tokens_file)) or not os.path.exists(os.path.join(output_dir, str(i), lemmas_file)):
            return True

if __name__ == '__main__':
    need_to_run_crawler = False

    if crawl_required(CrawlerUtils.OUTPUT_DIR, CrawlerUtils.INDEX_FILE, CrawlerUtils.MAX_PAGES):
        print("Запускаем краулер...")
        need_to_run_crawler = True
    else:
        print("Файлы уже существуют. Краулер не запущен.")    

    if need_to_run_crawler:
        crawler = WikiCrawler(
            start_url = CrawlerUtils.WIKI_START_URL,
            max_pages = CrawlerUtils.MAX_PAGES,
            output_dir = CrawlerUtils.OUTPUT_DIR,
            index_file = CrawlerUtils.INDEX_FILE
        )

        crawler.crawl()

    need_to_run_tokenization = False
    if tokenization_required(TokenizerUtils.OUTPUT_DIR, TokenizerUtils.TOKENS_FILE, TokenizerUtils.LEMMAS_FILE, CrawlerUtils.MAX_PAGES * 2):
        print("Запускаем токенизацию и лемматизацию...")
        need_to_run_tokenization = True
    else:
        print("Файлы уже существуют. Токенизатор не запущен.")    
    
    if need_to_run_tokenization:
        html_processor = HtmlProcessor(
            processing_dir = CrawlerUtils.OUTPUT_DIR,
            output_dir=TokenizerUtils.OUTPUT_DIR,
            tokens_file = TokenizerUtils.TOKENS_FILE,
            lemmas_file = TokenizerUtils.LEMMAS_FILE
        )    

        txt_files = [
            f for f in os.listdir(CrawlerUtils.OUTPUT_DIR)
            if f.endswith(".txt")
        ]
        for i in range(1, len(txt_files)+1):
            html_processor.process_document(str(i))

    index_filename = "inverted_index.txt"
    indexer = InvertedIndexProcessor()
    if not os.path.exists(index_filename):
        print("Построение инвертированного индекса...")
        indexer.build(TokenizerUtils.OUTPUT_DIR)
        indexer.save(index_filename)
        print(f"Индекс сохранён в {index_filename}")
    else:
        indexer.load(index_filename)
        print(f"Загружен индекс из {index_filename}")

    if tfidf_required(TfIdfUtils.OUTPUT_DIR, TfIdfUtils.TOKENS_TFIDF_FILE, TfIdfUtils.LEMMAS_TFIDF_FILE, CrawlerUtils.MAX_PAGES):
        print("Вычисление TF-IDF...")
        tfidf_processor = TfIdfProcessor()
        tfidf_processor.process_all(
            crawl_dir=CrawlerUtils.OUTPUT_DIR,
            tokenized_dir=TokenizerUtils.OUTPUT_DIR,
            output_dir=TfIdfUtils.OUTPUT_DIR,
            n_docs=CrawlerUtils.MAX_PAGES,
        )
    else:
        print(f"TF-IDF файлы уже существуют в {TfIdfUtils.OUTPUT_DIR}/. Пропускаем.")

    while True:
        query = input("\nВведите булев запрос (AND, OR, NOT, скобки). Пустая строка для выхода:\n")
        if not query.strip():
            break
        try:
            results = indexer.query(query)
            if results:
                print("Найденные документы:", ' '.join(sorted(results, key=int)))
            else:
                print("Ничего не найдено.")
        except Exception as e:
            print("Ошибка при разборе запроса:", e)

    print("\nИнициализация векторного поиска...")
    vector_engine = VectorSearchEngine()
    vector_engine.load(TfIdfUtils.OUTPUT_DIR, CrawlerUtils.MAX_PAGES)
    vector_engine.load_url_index(CrawlerUtils.INDEX_FILE)

    while True:
        query = input("\nВведите запрос для векторного поиска. Пустая строка для выхода:\n")
        if not query.strip():
            break
        results = vector_engine.search(query, top_n=VectorSearchUtils.TOP_N)
        if results:
            print(f"Топ-{len(results)} результатов:")
            for rank, (doc_id, url, score) in enumerate(results, start=1):
                print(f"  {rank}. [doc {doc_id}] (score={score:.6f})")
        else:
            print("Ничего не найдено.")