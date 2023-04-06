import json
import wikicrawler
import word_matrix as wma
# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    words, dicts = wikicrawler.main()
    wm = wma.WordMatrix()
    wm.adjust_word_matrix(words,dicts)
    wm.save()
    wm.read()
    wm.info()
    text = input("ENTER SENTENCE TO SEARCH ")
    i = wm.compare(text)
    with open('sites.json', 'r') as read_file:
        sites = json.load(read_file)
    print(sites[i])
