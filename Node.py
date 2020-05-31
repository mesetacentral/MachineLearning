class Node:
    """
    A class used to represent a generic node containing a :param dataset.

    Attributes:
        :param dataset: (DataSet) node's data set
        :param depth: (int) node's depth

    Methods:
        print(self): prints the node's dataset
    """

    def __init__(self, dataset, depth):
        """
        Parameters:
            :param dataset: (DataSet) node's data set
            :param depth: (int) node's depth
        """

        self.dataset = dataset
        self.depth = depth

    def print(self):
        """Prints the node's dataset"""
        self.dataset.print()


