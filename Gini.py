class Gini:
    """
    Gini coefficient calculator

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
        """Computes Gini coefficient"""
        squared = [(count/len(self.subset.y))**2 for count in self.subset.counts()]
        return 1 - sum(squared)
