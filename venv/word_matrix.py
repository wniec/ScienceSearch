import numpy as np
import pickle as pkl

from vectorizer import Vectorizer
from scipy.sparse.linalg import svds


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
        word_matrix = np.zeros((n, len(keys)), dtype=np.float32)
        for i in range(len(dicts)):
            word_sum = 0
            for word in dicts[i]:
                index = inverse_index[word]
                word_matrix[i, index] = dicts[i][word]
                word_sum += dicts[i][word]
        print("frequency normalization started")
        m = np.log10(np.array([n / words[word_index[i]] for i in range(len(keys))]))
        word_matrix = word_matrix * m
        lengths = np.nan_to_num(1 / np.sqrt(np.sum(word_matrix ** 2, axis=1)), False, nan=0.0, posinf=0.0, neginf=0.0)
        word_matrix = (word_matrix.T * lengths).T
        self.word_index = word_index
        self.inverse_index = inverse_index
        # self.word_matrix = lower_rank_approximation(word_matrix, len(dicts)//3)
        self.vector_func = Vectorizer(self.inverse_index)

    def save(self):
        np.save("saved/matrix.npy", self.word_matrix)
        with open("saved/dictionary.pkl", "wb") as write_file:
            pkl.dump(self.word_index, write_file)

    def read(self):
        self.word_matrix = np.load("saved/matrix.npy", allow_pickle=True)
        with open('saved/dictionary.pkl', 'rb') as read_file:
            self.word_index = pkl.load(read_file)
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

    def lower_rank(self, k):
        U, D, V = np.linalg.svd(self.word_matrix)
        self.word_matrix = U[:, :k] @ np.diag(D) @ V[:k, :]
        self.vector_func = Vectorizer(self.inverse_index)


def lower_rank_approximation(matrix, k: int):
    U, D, V = np.linalg.svd(matrix)
    return U[:, :k] @ np.diag(D) @ V[:k, :]
