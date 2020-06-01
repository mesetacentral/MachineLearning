from Gini import Gini
from MSE import MSE
import logging

logger = logging.getLogger('Logger')


class Criterion:
    """
    This class chooses and computes the chosen coefficient for two data sets

    Attributes:
        :param coefficient: (str) coefficient's name
    """

    def __init__(self, coefficient):
        """
        Parameters:
            :param coefficient: (str) coefficient's name
        """

        if coefficient == 'Gini':
            self.coefficient = Gini()
        elif coefficient == 'MSE':
            self.coefficient = MSE()
        else:
            logger.error('Invalid coefficient %s', self.coefficient)
            raise NotImplementedError

    def compute(self, subset1, subset2):
        """
        Returns weighted sum of :param subset1 and :param subset2 chosen coefficients

        Parameters:
            :param subset1: (DataSet) one of the data sets to compute the coefficient from
            :param subset2: (DataSet) one of the data sets to compute the coefficient from
        """

        size1 = subset1.X.shape[0]
        size2 = subset2.X.shape[0]

        return (self.coefficient.compute(subset1) * size1 + self.coefficient.compute(subset2) * size2)/(size1 + size2)
