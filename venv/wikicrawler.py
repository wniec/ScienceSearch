import json
import re
import string
from urllib.error import HTTPError
import nltk.corpus
import nltk
import requests
import wikipediaapi as wpa
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from bs4 import BeautifulSoup as bs
import urllib.request
from urllib.parse import quote

dictionary = set(nltk.corpus.words.words())


def get_content(sites: list[wpa.Wikipedia.page], words: dict, dicts: list[dict], word_count: dict):
    wnl = WordNetLemmatizer()
    en_stops = set(wnl.lemmatize(word) for word in stopwords.words('english'))
    for site in sites:
        link = get_link(site)
        try:
            webpage = str(urllib.request.urlopen(link).read())
            dicts.append(dict())
            soup = bs(webpage, features="html.parser")
            site_text = clean(soup.getText())
            word_list = site_text.split()[1:]
            stemmed = [wnl.lemmatize(word).casefold() for word in word_list if (
                    (word in dictionary or word[0].isupper()) and word not in en_stops)]
            for s in stemmed:
                if s not in en_stops:
                    if s in dicts[-1]:
                        dicts[-1][s] += 1
                    else:
                        dicts[-1][s] = 1
                        if s in words:
                            words[s] += 1
                        else:
                            words[s] = 1
                    if s in word_count:
                        word_count[s] += 1
                    else:
                        word_count[s] = 1
        except HTTPError:
            pass


def search(url, category, maxdepth, depth=0):
    page = requests.get(url)
    data = page.text
    soup = bs(data, features="html.parser")
    result = set()
    txt = soup.getText()
    if category.casefold() in txt or category in txt:
        for link in soup.find_all('a'):
            href = link.get('href')
            if href and href.startswith('/wiki/'):
                rest = href[6:]
                if maxdepth >= depth and not re.match(
                        'Category*|Wikipedia*|Special*|Wayback*|List*|File*|.*(identifier)$|Help*', rest):
                    result.add(rest)
                elif maxdepth > depth and re.match('Category*|List*', rest) and not rest.startswith('Category:Commons'):
                    new_url = "https://en.wikipedia.org/wiki/" + rest
                    result.update(search(new_url, category, maxdepth, depth + 1))
    return result


def main(length: int, buffer_size: int = 10):
    nltk.download("punkt")
    nltk.download('words')
    nltk.download('stopwords')
    nltk.download("wordnet")
    nltk.download("omw-1.4")
    print("Downloading categories started")
    categories = ["Physics", "Mathematics", "Computer_science", "Astronomy"]
    sites = set()
    for category in categories:
        start = "https://en.wikipedia.org/wiki/Category:"+category
        sites.update(search(start, category, 2))
    site_list = list(sites)[:length]
    print("total: ", len(site_list), " sites")
    with open("saved/sites.json", "w") as write_file:
        json.dump([title(site) for site in site_list], write_file)
    print("Downloading sites content started")
    words = dict()
    word_count = dict()
    dicts = []
    for i in range(len(site_list) // buffer_size):
        get_content(site_list[i * buffer_size:(i + 1) * buffer_size], words, dicts, word_count)
    print("Downloading sites content ended")
    reduce(dicts, word_count, words, length)
    print("total: ", len(words), " words")
    with open("saved/words.json", "w") as write_file:
        json.dump(words, write_file)
    return words, dicts


def reduce(dicts: list[dict], word_count: dict, words: dict, n: int):
    to_remove = set()
    for word in word_count:
        if word_count[word] < 20:
            for d in dicts:
                if word in d:
                    d.pop(word)
            to_remove.add(word)
    for word in words:
        if words[word] <3:
            for d in dicts:
                if word in d:
                    d.pop(word)
            to_remove.add(word)
    for word in list(to_remove):
        words.pop(word)


def get_link(site_title: string) -> string:
    return "https://en.wikipedia.org/wiki/" + quote(site_title.replace(" ", "_"))


def title(site: string) -> string:
    return site.replace("_", " ")


def clean(text: string) -> string:
    partly = re.sub('\\\\t|\\\\n|\\\\r|\\\\a|\\\\f|\\\\v|\\\\b', " ", text)
    return re.sub('[^a-zA-Z]+', ' ', partly)


if __name__ == "__main__":
    word_matrix, word_index = main(100)
