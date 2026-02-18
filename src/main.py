import os
from Parser import Utils
from Parser.WikiCrawler import WikiCrawler

def crawl_required(output_dir="output", index_file="index.txt", min_files=100):
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

if __name__ == '__main__':
    crawler = WikiCrawler(
        start_url = Utils.WIKI_START_URL,
        max_pages = Utils.MAX_PAGES,
        output_dir = Utils.OUTPUT_DIR,
        index_file = Utils.INDEX_FILE
    )

    if crawl_required(Utils.OUTPUT_DIR, Utils.INDEX_FILE, Utils.MAX_PAGES):
        print("Запускаем краулер...")
        crawler.crawl()
    else:
        print("Файлы уже существуют. Краулер не запущен.")