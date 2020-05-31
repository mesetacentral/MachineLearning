import time

import numpy as np
from scipy import stats as s


class DataSet:
    """
    A class used to represent a data set

    Attributes:
        :param X: (np.ndarray) array of array of float
        :param y: (np.array) array of int

    Methods:
        subset_from_ratio(self, ratio_samples): Builds a new DataSet off self of length :param ratio_samples*len(self)
        divide(self, feature, value): Splits self in two DataSet using :param feature and :param value
        mode(self): Returns mode of self.y
        mean(self): Returns mean of self.y
        counts(self): Returns the number of occurrences in self.y
        print(self): Prints self.X and self.y
    """

    def __init__(self, X, y):
        """
        Parameters:
            :param X: (np.ndarray) array of array of float
            :param y: (np.array) array of int
        """

        self.X = X
        self.y = y

    def subset_from_ratio(self, ratio_samples):
        """
        Builds a new DataSet off self


        A number of indexes are randomly chosen with replacement. The new DataSet is built with the indexes' elements
        of self. The number of generated indexes is ratio_samples * len(self.y).

        Attributes:
            :param ratio_samples: (float) ratio between self's length and new subset's length. Could be greater than 1
        """

        idxs = [np.random.randint(0, len(self.y)) for _ in range(int(ratio_samples * len(self.y)))]
        X_array = np.asarray(self.X[idxs])
        y_array = np.asarray(self.y[idxs])

        return DataSet(X_array, y_array)

    def divide(self, feature, value):
        """
        Splits self in two DataSet using :param feature and :param value

        Two boolean lists are built by comparing a given :param feature for all samples in self.X to :param value.
        These are then used to choose which samples are used in the DataSets.

        Attributes:
            :param feature:
            :param value:
        """

        idx1 = self.X[:, feature] >= value
        idx2 = self.X[:, feature] < value

        subset1 = DataSet(self.X[idx1], self.y[idx1])
        subset2 = DataSet(self.X[idx2], self.y[idx2])

        return subset1, subset2

    def mode(self):
        """Returns mode of self.y"""
        return int(s.mode(self.y.tolist())[0])

    def mean(self):
        """Returns mean of self.y"""
        return np.sum(self.y)/self.y.shape

    def counts(self):
        """Returns the number of occurrences in self.y"""

        return np.bincount(self.y)

    def print(self):
        """Prints self.X and self.y"""
        print(self.X)
        print(self.y)


# MAIN for checking purposes
# X = np.array([[2.1, 5.4], [3.1, 4.5], [2.3, 6.7], [1.8, 4.7], [2.5, 5.1], [1.5, 5.9]])
# y = np.array([1, 1, 0, 2, 1, 2])
# dataset = DataSet(X, y)
# dataset.print()
# print(dataset.X)
# print(dataset.y)
# print(dataset.mode())
# print(dataset.counts())
# subset = dataset.subset_from_ratio(0.75)
# subset.print()
# subset_1, subset_2 = dataset.divide(0, 1.4)
# subset_1.print()
# subset_2.print()
