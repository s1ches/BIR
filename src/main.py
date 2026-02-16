from Parser import Utils
from Parser.WikiCrawler import WikiCrawler


if __name__ == '__main__':
    crawler = WikiCrawler(
        start_url = Utils.WIKI_START_URL,
        max_pages = Utils.MAX_PAGES,
        output_dir = Utils.OUTPUT_DIR,
        index_file = Utils.INDEX_FILE
    )

    crawler.crawl()