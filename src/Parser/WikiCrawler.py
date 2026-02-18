import os
import time
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
from collections import deque
import Parser.Utils


class WikiCrawler:
    def __init__(self, start_url, max_pages=100, output_dir="output", index_file="index.txt"):
        self.start_url = start_url
        self.max_pages = max_pages
        self.output_dir = output_dir
        self.index_file = index_file

        self.base_domain = Parser.Utils.WIKI_URL
        self.visited = set()
        self.queue = deque([start_url])
        self.page_count = 0

        self.headers = {
            "User-Agent": "Mozilla/5.0 (compatible; Crawler/1.0)"
        }

        self._prepare_output_directory()

    def _prepare_output_directory(self):
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def _is_valid_wiki_article(self, url):
        parsed = urlparse(url)

        if parsed.netloc != "ru.wikipedia.org":
            return False

        if not parsed.path.startswith("/wiki/"):
            return False

        excluded_postfixes = (
            "static",
            "load.php",
            "resources",
            "upload",
            ".php",
            "/wiki/Служебная:",
            "/wiki/Файл:",
            "/wiki/Категория:",
            "/wiki/Портал:",
            "/wiki/Шаблон:",
            "/wiki/Википедия:",
            "/wiki/Специальная:"
        )

        for prefix in excluded_postfixes:
            if prefix in parsed.path:
                return False

        if ":" in parsed.path[6:]:
            return False

        return True

    def _download_page(self, url):
        try:
            response = requests.get(url, headers=self.headers, timeout=10)
            if response.status_code == 200:
                return response.text
        except Exception as e:
            print(f"Ошибка загрузки {url}: {e}")
        return None

    def _save_page(self, content):
        filename = os.path.join(self.output_dir, f"{self.page_count}.txt")
        with open(filename, "w", encoding="utf-8") as f:
            f.write(content)

    def _write_index(self, file, url):
        file.write(f"{self.page_count} {url}\n")

    def _extract_links(self, html):
        soup = BeautifulSoup(html, "html.parser")
        links = []

        for tag in soup.find_all("a", href=True):
            href = tag["href"]
            full_url = urljoin(self.base_domain, href)

            if self._is_valid_wiki_article(full_url):
                links.append(full_url)

        return links

    def crawl(self):
        with open(self.index_file, "w", encoding="utf-8") as index_file:
            while self.queue and self.page_count < self.max_pages:
                current_url = self.queue.popleft()

                if current_url in self.visited:
                    continue

                html = self._download_page(current_url)
                if not html:
                    continue

                self.visited.add(current_url)
                self.page_count += 1

                print(f"[{self.page_count}] Скачиваю: {current_url}")

                self._save_page(html)

                self._write_index(index_file, current_url)

                new_links = self._extract_links(html)
                for link in new_links:
                    if link not in self.visited:
                        self.queue.append(link)

                time.sleep(0.5)

        print(f"Готово. Скачано страниц: {self.page_count}")