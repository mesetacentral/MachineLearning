import logging
from SplitNode import SplitNode
from Criterion import Criterion


logger = logging.getLogger('Logger')


class DecisionTree:
    """
    A class used to represent a DecisionTree.

    Attributes:
        :param dataset: (DataSet) Contains all data
        criterion: (Criterion) It computes the chosen :param coefficient
        :param values:
            ([float]) List of values to use
            (None) A list needs to be built
        :param max_depth: (int) Maximum depth for any node
        first_node: (SplitNode) Root node for this tree

    Methods:
        predict(self, X_test): Returns the predicted types for each sample in :param X_test
        build_values(self, values, feature): Builds the list of values for any given feature.
    """

    def __init__(self, dataset, coefficient, values, max_depth):
        self.dataset = dataset
        self.criterion = Criterion(coefficient)
        self.values = values
        self.max_depth = max_depth

        self.first_node = SplitNode(self.dataset, 0)

    def predict(self, X_test):
        """
        Returns the predicted types for each sample in :param X_test


        The prediction process starts with the first node. Any sample in the test data has as many features
        as any in the training data, so sample[node.feature] is compared to node.value. Then node is updated
        based on the last comparison: node = node.left_child or node = node.right_child.
        Either way, the comparison and update are repeated self.max_depth + 1 times, or until node is LeafNode.
        When that happens, we'll know what type sample has been predicted into: node.dataset includes all samples types,
        so the predicted value for a given node is the mode of those types, as it is the most likely type.
        This process is repeated for all samples in X_test and that gives us a predictions list.

        Parameters:
            :param X_test: test data
        """

        predictions = []
        for x in X_test:  # for every sample in X_test
            node = self.first_node
            try:  # tries to go to the deepest node
                for _ in range(self.max_depth + 1):
                    if x[node.feature] > node.value:
                        node = node.left_child
                    else:
                        node = node.right_child
            except AttributeError:  # but if node has no childs
                pass
            predictions.append(node.type)  # we get the prediction for the sample

        logger.debug('Predict list: %s', predictions)
        return predictions

    def build_values(self, values, feature):
        """
        Builds the list of values for any given feature.

        If values is None, the list has to be built. Otherwise, it returns values.

        Parameters:
            :param values: ([float]) or (None)
            :param feature: (int) index of any sample in self.dataset.X
        """

        if values is None:
            assert feature <= self.dataset.X.shape[1]
            return list(set([self.dataset.X[i][feature] for i in range(self.dataset.X.shape[0])]))
        else:
            return values
