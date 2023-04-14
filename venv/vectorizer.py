import string
import numpy as np
from nltk.stem import WordNetLemmatizer


class Vectorizer:
    def __init__(self, word_index):
        self.word_index = word_index

    def vectorize(self, text: string) -> np.array:
        wnl = WordNetLemmatizer()
        result = np.zeros(len(self.word_index), dtype=np.float32)
        for word in text.split():
            word_lemma = wnl.lemmatize("".join(filter(lambda x: x.isalpha(), word))).casefold()
            if word_lemma in self.word_index:
                index = self.word_index[word_lemma]
                result[int(index)] += 1
        return result
