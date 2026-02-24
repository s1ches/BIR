LEMMAS_FILE='lemmas.txt'
TOKENS_FILE='tokens.txt'
STOPWORDS_FILE='stopwords.txt'
OUTPUT_DIR='output_tokenized'

def get_custom_stopwords(filename="stopwords.txt"):
    custom_stops = set()
    
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            for line in f:
                word = line.strip().lower()
                if word and not word.startswith('#'):
                    custom_stops.add(word)
        
        print(f"Загружено {len(custom_stops)} стоп-слов из {filename}")
        return custom_stops
    
    except FileNotFoundError:
        print(f"Файл {filename} не найден. Создайте его со списком стоп-слов.")
        return set()