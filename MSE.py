import numpy as np


class MSE:
    """
    MSE coefficient calculator

    Methods:
        compute: calculates the coefficient for a given data set
    """

    def __init__(self):
        pass

    def compute(self, subset):
        """
        Computes MSE coefficient
        Parameters:
            :param subset: (DataSet) data to calculate the coefficient from
        """
        try:
            assert subset.y.shape[0] > 1  # variance can not be computed with just one value
        except AssertionError:
            return 0
        return np.var(subset.y, dtype=np.float64)
