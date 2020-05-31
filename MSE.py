import numpy as np


class MSE:
    """
    MSE coefficient calculator

    Attributes:
        :param subset: (DataSet) data to compute the coefficient from
    """

    def __init__(self, subset):
        """
        Parameters:
            :param subset: (DataSet) data set to compute the coefficient from
        """

        self.subset = subset

    def compute(self):
        """Computes MSE coefficient"""
        try:
            assert self.subset.y.shape[0] > 1  # variance can not be computed with just one value
        except AssertionError:
            return 0
        return np.var(self.subset.y, dtype=np.float64)
