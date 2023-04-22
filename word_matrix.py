import queue
import numpy as np
import pickle as pkl
from vectorizer import Vectorizer
from scipy.sparse.linalg import svds
from scipy.sparse import csr_matrix


class WordMatrix:
    def __init__(self, matrix: np.array = None, index: dict = None):
        self.lengths = None
        self.V = None
        self.D = None
        self.U = None
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
        self.word_matrix = np.zeros((n, len(keys)), dtype=np.float32)
        for i in range(len(dicts)):
            word_sum = 0
            for word in dicts[i]:
                index = inverse_index[word]
                self.word_matrix[i, index] = dicts[i][word]
                word_sum += dicts[i][word]
        print("frequency normalization started")
        m = np.log10(np.array([n / words[word_index[i]] for i in range(len(keys))]))
        self.word_matrix = self.word_matrix * m
        lengths = np.nan_to_num(1 / np.sqrt(
            np.array([sum(self.word_matrix[i, :] ** 2) for i in range(
                len(dicts))])), False, nan=0.0, posinf=0.0, neginf=0.0)
        self.word_matrix = self.word_matrix.T
        self.word_matrix *= lengths
        self.word_matrix = self.word_matrix.T
        self.lengths = np.sqrt([sum(self.word_matrix[i, :] ** 2) for i in range(len(self.word_matrix))])
        self.word_matrix = csr_matrix(self.word_matrix)
        self.word_index = word_index
        self.lower_rank()

    def save(self):
        np.save("saved/U.npy", self.U)
        np.save("saved/V.npy", self.V)
        np.save("saved/D.npy", self.D)
        np.save("saved/lengths.npy", self.lengths)
        with open("saved/dictionary.pkl", "wb") as write_file:
            pkl.dump(self.word_index, write_file)

    def read(self):
        self.U = np.load("saved/U.npy", allow_pickle=True, mmap_mode='r')
        self.V = np.load("saved/V.npy", allow_pickle=True, mmap_mode='r')
        self.D = np.load("saved/D.npy", allow_pickle=True, mmap_mode='r')
        self.lengths = np.load("saved/lengths.npy", allow_pickle=True, mmap_mode='r')
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
        result = np.abs((((vector.T @ self.V.T) @ np.diag(self.D).T) @ self.U.T)) / self.lengths
        best = queue.PriorityQueue()
        for i in range(top):
            best.put((result[i], i))
        for i in range(top, len(result)):
            best.put((result[i], i))
            best.get()
        return [best.get()[1] for _ in range(top)][::-1]

    def lower_rank(self):
        print("decomposition started")
        self.U, self.D, self.V = svds(self.word_matrix, k=300)
        print("decomposition ended")
        print("saving decomposition")
        print("calculating reult")
        self.word_matrix = None
        self.vector_func = Vectorizer(self.inverse_index)
