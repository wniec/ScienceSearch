import wikicrawler
import word_matrix as wma
if __name__ == '__main__':
    words, dicts = wikicrawler.main(50)
    wm = wma.WordMatrix()
    wm.adjust_word_matrix(words, dicts)
    wm.save()
