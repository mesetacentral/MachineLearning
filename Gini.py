class Gini:
    """
    Gini coefficient calculator

    Methods:
        compute: calculates the coefficient for a given data set
    """

    def __init__(self):
        pass

    def compute(self, subset):
        """
        Computes Gini coefficient

        Parameters:
            :param subset: (DataSet) data to calculate the coefficient from
        """
        squared = [(count/len(subset.y))**2 for count in subset.counts()]
        return 1 - sum(squared)
