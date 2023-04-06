import numpy as np
import json
from json import JSONEncoder
from vectorizer import Vectorizer


class WordMatrix:
    def __init__(self, matrix: np.array = None, index: dict = None):
        self.inverse_index = None
        self.word_matrix = matrix
        self.word_index = index
        self.vector_func = Vectorizer(self.inverse_index)

    def adjust_word_matrix(self, words, dicts):
        print("Words indexing started")
        n = len(dicts)
        keys = list(words.keys())
        word_index = {i: keys[i] for i in range(len(words))}
        inverse_index = {keys[i]: i for i in range(len(words))}
        word_matrix = np.zeros((n, len(keys)), dtype=np.float64)
        for i in range(len(dicts)):
            print(i)
            for word in dicts[i]:
                index = inverse_index[word]
                word_matrix[i, index] = dicts[i][word]

        print("frequency normalization started")
        for i in range(len(keys)):
            m = words[word_index[i]]
            word_matrix[:, i] *= np.log10(n / m)

        print(len(word_matrix))
        print(len(word_matrix[-1]))
        print(word_matrix[:6, :6])
        print(list(words.keys())[:30])
        self.word_index = word_index
        self.inverse_index = inverse_index
        self.word_matrix = word_matrix
        self.vector_func = Vectorizer(self.inverse_index)

    def save(self):
        numpy_data = {"wordMatrix": self.word_matrix}
        with open("matrix.json", "w") as write_file:
            json.dump(numpy_data, write_file, cls=NpArrayEncoder)
        with open("dictionary.json", "w") as write_file:
            json.dump(self.word_index, write_file)

    def read(self):
        with open("matrix.json", "r") as read_file:
            decoded_array = json.load(read_file)
            self.word_matrix = np.asarray(decoded_array["wordMatrix"])
        with open('dictionary.json', 'r') as read_file:
            self.word_index = json.load(read_file)
        self.inverse_index = {self.word_index[i]: i for i in self.word_index}
        self.vector_func = Vectorizer(self.inverse_index)

    def info(self):
        print(type(self.word_index), "\t", len(self.word_index))
        print(type(self.word_matrix), "\t", len(self.word_matrix))
        print(self.word_matrix[:10, :10])

    def compare(self, text):
        vector = self.vector_func.vectorize(text)
        print(any(vector))
        print(np.sqrt(np.sum(self.word_matrix ** 2, axis=1)))
        result = (vector.T @ self.word_matrix.T) / np.sqrt(np.sum(vector ** 2)) / np.sqrt(
         np.sum(self.word_matrix ** 2, axis=1))
        print(result)
        print(result.argmax())
        return result.argmax()


class NpArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)
