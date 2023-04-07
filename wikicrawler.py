import json
import string
import nltk
import wikipediaapi as wpa
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


def get_content(sites):
    words = dict()
    dicts = []
    wnl = WordNetLemmatizer()
    en_stops = set(wnl.lemmatize(word) for word in stopwords.words('english'))
    for site in sites:
        dicts.append(dict())
        content = site.text
        word_list = content.split()
        stemmed = [wnl.lemmatize("".join(filter(lambda x: x.isalpha(), word))
                                 ).casefold() for word in word_list]
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
    return words, dicts


def get_category_members(category_members, level=0, max_level=1):
    result = set()
    for c in category_members.values():
        if c.ns == wpa.Namespace.CATEGORY and level < max_level:
            result.update(get_category_members(c.categorymembers, level=level + 1, max_level=max_level))
        elif c.ns != wpa.Namespace.CATEGORY and not c.title.startswith("List"):
            result.add(c)
    return result


def main(length: int):
    nltk.download("punkt")
    nltk.download('stopwords')
    nltk.download("wordnet")
    nltk.download("omw-1.4")
    wiki_wiki = wpa.Wikipedia('en')
    categories = ["Physics", "Mathematics", "Medicine", "Chemistry", "Biology", "Astronomy"]
    sites = set()
    print("Downloading categories started")
    for c in categories[:1]:
        print(c)
        cat_name = "Category:" + c
        cat = wiki_wiki.page(cat_name)
        sites.update(get_category_members(cat.categorymembers))
    site_list = list(sites)[:length]
    with open("jsons/sites.json", "w") as write_file:
        json.dump([site.title for site in site_list], write_file)
    print("Downloading sites content started")
    words, dicts = get_content(site_list)
    print("Downloading sites content ended")
    return words, dicts


def get_link(title: string) -> string:
    return "https://en.wikipedia.org/wiki/"+title.replace(" ", "_")


if __name__ == "__main__":
    word_matrix, word_index = main()
