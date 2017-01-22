"""
Copyright 2017, University of Freiburg.

Elmar Haussmann <haussmann@cs.uni-freiburg.de>
Patrick Brosi <brosi@cs.uni-freiburg.de>
"""

import re
import sys
import numpy
import pdb
from scipy.sparse import csr_matrix


def generate_vocab(filename):
    """
    Read from the provided file name and return vocabularies mapping from
    string to ID for words and classes/labels.

    You should call this ONLY on your training data.
    """

    next_class_id = 0
    next_feature_id = 0

    # Map from label/class to label id.
    class_vocab = dict()

    # Map from word to word id.
    word_vocab = dict()

    with open(filename, "r", encoding="utf8") as f:
        for line in f:
            cols = line.strip().split('\t')
            label, text = cols[0], cols[1]
            if label not in class_vocab:
                class_vocab[label] = next_class_id
                next_class_id += 1
            words = re.sub("\W+", " ", text.lower()).split()
            for w in words:
                if w not in word_vocab:
                    word_vocab[w] = next_feature_id
                    next_feature_id += 1

    return word_vocab, class_vocab


def read_labeled_data(filename, class_vocab, word_vocab):
    """
    Read the file and return a sparse document-term matrix as well as a list
    of labels of each document. You need to provide a class and word
    vocabulary. Words not in the vocabulary are ignored. Documents labeled
    with classes not in the class vocabulary are also ignored.

    The returned document-term matrix X has size n x m, where n is the number
    of documents and m the number of features (i.e. word ids). The value at
    i, j denotes the number of times word id j is present in document i.

    The returned labels vector y has size n (one label for each document). The
    value at index j denotes the label (class id) of document j.
    """

    labels = []
    row, col, value = [], [], []
    num_examples = 0
    num_cols = len(word_vocab)

    with open(filename, "r", encoding="utf8") as f:
        for i, line in enumerate(f):
            cols = line.strip().split('\t')
            label, text = cols[0], cols[1]
            if label in class_vocab:
                num_examples += 1
                labels.append(class_vocab[label])
                words = re.sub("\W+", " ", text.lower()).split()
                for w in words:
                    if w in word_vocab:
                        w_id = word_vocab[w]
                        row.append(i)
                        col.append(w_id)
                        # Duplicate values at the same position
                        # ij are summed
                        value.append(1.0)

    X = csr_matrix((value, (row, col)), shape=(num_examples, num_cols))
    y = numpy.array(labels)
    return X, y


class NaiveBayes(object):

    def __init__(self,class_vocab, word_vocab):
        """
        Init a naive bayes classifier supporting num_classes of classes
        and num_features of words.
        """

        # stored probabilities of each class.
        self.p_c = list()
	
        # stored probabilities of each word in each class
        self.p_wc = None
        self.e = 0.1
        self.log_p_wc = None
        self.classes = list()
        self.class_vocab = class_vocab
        self.word_vocab = word_vocab
        
    def train(self, X, y, word_vocab):
        """
        Train on the sparse document-term matrix X and associated labels y.

        In the test case below, p_wc is a class-term-matrix and has a row
        for each class and a column for each term. So the value at ij is
        the p_wc for the j-th term in the i-th class.

        p_c is an array of global probabilities for each class.

        >>> wv, cv = generate_vocab("example.txt")
        >>> X, y = read_labeled_data("example.txt", cv, wv)
        >>> nb = NaiveBayes()
        >>> nb.train(X, y, wv)
        >>> numpy.round(nb.p_wc, 3)
        array([[ 0.664,  0.336],
               [ 0.335,  0.665]])
        >>> numpy.round(nb.p_c, 3)
        array([ 0.5,  0.5])
        """

        total = numpy.unique(y)
        self.classes = total
        matrices = []
        for i, label in enumerate(total):
            self.p_c.append((y == label).sum() / len(y))
            matrices.append(X[y == label])

        for i, matrix in enumerate(matrices):
            column_summed_matrix = csr_matrix.sum(matrix, axis=0)
            # column_summed_matrix = numpy.sum(matrix.todense(), axis=0)

            n_vocab = len(word_vocab.keys())
            nC = column_summed_matrix.sum() + self.e * n_vocab

            column_summed_matrix += self.e

            if i == 0:
                p_row = numpy.divide(column_summed_matrix, nC)

            else:
                p_row_onwards = numpy.divide(column_summed_matrix, nC)
                p_row = numpy.concatenate((p_row, p_row_onwards))

        self.p_wc = p_row
        self.log_p_wc = numpy.log(self.p_wc)

        # TODO!

    def predict(self, X):
        """
        Predict a label for each example in the document-term matrix,
        based on the learned probabities stored in this class.

        Return a list of predicted label ids.

        >>> wv, cv = generate_vocab("example.txt")
        >>> X, y = read_labeled_data("example.txt", cv, wv)
        >>> nb = NaiveBayes()
        >>> nb.train(X, y, wv)
        >>> X_test, y_test = read_labeled_data("example_test.txt", cv, wv)
        >>> nb.predict(X_test)
        matrix([[0, 1]], dtype=int32)
        >>> nb.predict(X)
        matrix([[0, 0, 1, 0, 1, 1]], dtype=int32)
        """

        self.log_p_wc = csr_matrix(self.log_p_wc)
        x_transpose = X.transpose(copy=False)
        result_matrix = self.log_p_wc.dot(x_transpose)

        predictions = numpy.argmax(result_matrix.todense(), axis=0)
        return predictions  # TODO!

    def evaluate(self, X, y):
        """
        Predict the labels of X and print evaluation statistics.
        """
        # ret = {}
        # result = []
        # for i,xclass in enumerate(y):
        #     if(y[i] == result[i]):
        #         ret[self.classes[y[i]] += 1
        # result = numpy.array(self.predict(X))
        # correct = len([r for r in result if r in y])
        # precision = correct / len(y)
        # recall = correct / len(result) * 100
        self.print_pc_values(X,y)


    def print_pc_values(self, X, y):
        print ("")
        sorted_row_idx = numpy.argsort(self.p_wc, axis=1)[:, self.p_wc.shape[1] - 30::]
        for i, prob in enumerate(self.p_c):
            class_name = self.get_class_from_value(i)
            print("Probability for class ", class_name, ": ", numpy.round(prob, 3))
            print("Top 30 words with highest p_wc values in this class: ")
            matrix_row = sorted_row_idx[i]
            for count in range(matrix_row.shape[1]):
                index = matrix_row.item(count)
                print("Word ", count + 1, ": ", self.get_word_from_index(index))



    def get_class_from_value(self, value):
        class_name ={v: value for value, v in self.class_vocab.items()}[value]
        return class_name

    def get_word_from_index(self, value):
        word ={v: value for value, v in self.word_vocab.items()}[value]
        return word

    def get_top_N_elements(self, row, n):
        return numpy.argsort(row)[::-1][:n]



def main():
    if len(sys.argv) != 3:
        print("Usage: python3 naive_bayes.py <train-input> <test-input>")
        exit(1)

    word_vocab, class_vocab = generate_vocab(sys.argv[1])
    X_train, y_train = read_labeled_data(sys.argv[1], class_vocab, word_vocab)
    X_test, y_test = read_labeled_data(sys.argv[2], class_vocab, word_vocab)
    n = NaiveBayes(class_vocab, word_vocab)
    n.train(X_train,y_train, word_vocab)
    n.predict(X_test)
    n.evaluate(X_test, y_test)
    # do training on training dataset

    # run the evaluation on the test dataset

    # print the 30 words with the highest p_wc values per class


if __name__ == '__main__':
    main()
