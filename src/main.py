import os
from Crawler import CrawlerUtils
from Crawler.WikiCrawler import WikiCrawler
from Tokenizer import TokenizerUtils
from Tokenizer.HtmlProcessor import HtmlProcessor

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