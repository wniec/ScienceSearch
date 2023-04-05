import wikipediaapi as wpa
import nltk


def main():
    wiki_wiki = wpa.Wikipedia('en')
    categories = ["Physics", "Mathematics", "Medicine", "Chemistry", "Biology", "Astronomy"]
    sites = set()
    for c in categories[:1]:
        cat_name = "Category:" + c
        cat = wiki_wiki.page(cat_name)
        sites.update(get_category_members(cat.categorymembers))
    site_list = list(sites)
    get_content(site_list)
    for site in sites:
        content = site.text
        print(len(content))
    return sites


def get_content(sites):
    words = dict()
    dicts = []
    lemma = nltk.wordnet.WordNetLemmatizer()
    for site in sites:
        dicts.append(dict())
        content = site.text
        word_list = content.split()
        stemmed = [lemma.lemmatize(word) for word in word_list]
        for s in stemmed:
            if s in words:
                words[s] += 1
            else:
                words[s] = 1

            if s in dicts[-1]:
                dicts[-1][s] += 1
            else:
                dicts[-1][s] = 1
    return words, dicts


def get_category_members(category_members, level=0, max_level=1):
    result = set()
    for c in category_members.values():
        if c.ns == wpa.Namespace.CATEGORY and level < max_level:
            result.update(get_category_members(c.categorymembers, level=level + 1, max_level=max_level))
        else:
            result.add(c)
    return result
