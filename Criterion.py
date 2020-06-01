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
        self.coefficient = coefficient

    def compute(self, subset1, subset2):
        """
        Returns weighted sum of :param subset1 and :param subset2 chosen coefficients

        Parameters:
            :param subset1: (DataSet) one of the data sets to compute the coefficient from
            :param subset2: (DataSet) one of the data sets to compute the coefficient from
        """
        # TODO: Could be better to check if coefficient is ok in init, and build the coefficient calculator there

        size1 = subset1.X.shape[0]
        size2 = subset2.X.shape[0]

        if self.coefficient == 'gini':
            return (Gini(subset1).compute() * size1 + Gini(subset2).compute() * size2)/(size1 + size2)
        elif self.coefficient == 'MSE':
            return (MSE(subset1).compute() * size1 + MSE(subset2).compute() * size2)/(size1 + size2)
        else:
            logger.error('Invalid coefficient %s', self.coefficient)
            raise NotImplementedError
