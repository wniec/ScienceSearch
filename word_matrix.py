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
            word_sum = 0
            for word in dicts[i]:
                index = inverse_index[word]
                word_matrix[i, index] = dicts[i][word]
                word_sum += dicts[i][word]
        print("frequency normalization started")
        m = np.log10(np.array([n / words[word_index[i]] for i in range(len(keys))]))
        word_matrix = word_matrix * m
        lengths = 1 / np.sqrt(np.sum(word_matrix ** 2, axis=1))
        word_matrix = (word_matrix.T * lengths).T
        self.word_index = word_index
        self.inverse_index = inverse_index
        self.word_matrix = lower_rank_approximation(word_matrix)
        self.vector_func = Vectorizer(self.inverse_index)

    def save(self):
        numpy_data = {"wordMatrix": self.word_matrix}
        with open("jsons/matrix.json", "w") as write_file:
            json.dump(numpy_data, write_file, cls=NpArrayEncoder)
        with open("jsons/dictionary.json", "w") as write_file:
            json.dump(self.word_index, write_file)

    def read(self):
        with open("jsons/matrix.json", "r") as read_file:
            decoded_array = json.load(read_file)
            self.word_matrix = np.asarray(decoded_array["wordMatrix"])
        with open('jsons/dictionary.json', 'r') as read_file:
            self.word_index = json.load(read_file)
        self.inverse_index = {self.word_index[i]: i for i in self.word_index}
        self.vector_func = Vectorizer(self.inverse_index)

    def info(self):
        print(type(self.word_index), "\t", len(self.word_index))
        print(type(self.word_matrix), "\t", len(self.word_matrix))
        print(self.word_matrix[:10, :10])

    def compare(self, text, top):
        vector = self.vector_func.vectorize(text)
        result = np.abs((vector.T @ self.word_matrix.T)) / np.sqrt(np.sum(
            self.word_matrix ** 2, axis=1))
        ind = np.argpartition(result, -top)[-top:]
        return ind[np.argsort(-result[ind])]


def lower_rank_approximation(matrix):
    U, D, V = np.linalg.svd(matrix)
    r = len(D)
    return U[:, :r] @ np.diag(D) @ V[:r, :]


class NpArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)
