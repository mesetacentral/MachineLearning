import numpy as np

from LeafNode import LeafNode


class SplitNode(LeafNode):
    """
    A class used to represent a split node.

    Attributes:
        :param dataset: (DataSet) node's data set
        :param depth: (int) node's depth
        left_child: (LeafNode) one of the childs
        right_child: (LeafNode) one of the childs

    Methods:
        print(self): prints the node's dataset
    """
    def __init__(self, dataset, depth):
        """
        Parameters:
            :param dataset: (DataSet) node's data set
            :param depth: (int) node's depth
        """
        LeafNode.__init__(self, dataset, depth)
        self.feature = np.Inf
        self.value = np.Inf
        self.left_child = None
        self.right_child = None

    def print(self):
        """Prints the node's dataset"""
        LeafNode.print(self)
