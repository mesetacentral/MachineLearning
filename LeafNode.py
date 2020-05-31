from Node import Node


class LeafNode(Node):
    """
    A class used to represent a final node.

    Attributes:
        :param dataset: (DataSet) node's data set
        :param depth: (int) node's depth
        type: (int) node data set's mode

    Methods:
        print(self): prints the node's dataset
    """

    def __init__(self, dataset, depth):
        """
        Parameters:
            :param dataset: (DataSet) node's data set
            :param depth: (int) node's depth
        """

        Node.__init__(self, dataset, depth)
        self.type = self.dataset.mode()

    def print(self):
        """Prints the node's data set"""
        Node.print(self)
