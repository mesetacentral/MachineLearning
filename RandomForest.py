import time
import logging
from scipy import stats as s
import numpy as np

from DecisionTree import DecisionTree
from DataSet import DataSet
from LeafNode import LeafNode
from SplitNode import SplitNode

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
# logging.basicConfig(filename='logging.log', filemode='w', level=logging.INFO,
# format='%(process)d-%(levelname)s-%(message)s')  # logging to file
logger = logging.getLogger('Logger')


class RandomForest:
    """
    RandomForest algorithm predicts what type a sample is after being trained to recognize it.

    Attributes:
        _max_depth: (int) maximum depth for any tree
        _min_split_size: (int) a node should not split if it has less than 5 samples
        :param ratio_samples: (float) ratio between the length of the whole data and the one given to a tree
        _num_trees: (int) number of trees RandomForest will build
        _num_features_node: (int) maximum number of features checked in any node
        :param coefficient: (str) type of coefficient calculated
        :param percentile: (int) [optional, default = 90] used to calculate top :param percentile most important
                features
        :param values: ([float]) [optional, default = None] a list of values to use instead of building one for
                every feature.
        _min_std_deviation: (float) [optional, default = 0] some features could be discarded before
                building any tree by deleting those whose values for every sample differ very little, which is to say,
                the std deviation of the data is small (smaller than min_std_deviation).
                Be careful as a value too big would discard too many.
        _non_important_features: ([int]) list of features with standard deviation smaller than :param min_std_deviation
        _most_important_features: ([int]) list of most used features
        _trees: ([DecisionTree]) list of trees built by RandomForest

    Methods:
        train(self, X_train, y_train): Builds all trees and the nodes associated to them.
        predict(self, X_test, y_test): Calculates what the final prediction for X_test is, and the accuracy to y_test.
    """

    def __init__(self, max_depth, min_split_size, ratio_samples, num_trees,
                 num_features_node, coefficient, percentile=90, values=None, min_std_deviation=0):
        """
        Parameters:
            :param max_depth: (int) maximum depth for any node
            :param min_split_size: (int) a node should not split if it has less than 5 samples
            :param ratio_samples: (float) ratio between the length of the whole data and the one given to a tree
            :param num_trees: (int) number of trees RandomForest will build
            :param num_features_node: (int) maximum number of features checked in any node
            :param coefficient: (str) type of coefficient calculated
            :param percentile: (int) [optional, default = 90] used to calculate top :param percentile most important
                features
            :param values: ([float]) [optional, default = None] a list of values to use instead of building one for
                every feature.
            :param min_std_deviation: (float) [optional, default = 0] some features could be discarded before
                building any tree by deleting those whose values for every sample differ very little, which is to say,
                the std deviation of the data is small (smaller than min_std_deviation).
                Be careful as a value too big would discard too many.
        """

        self._max_depth = max_depth
        self._min_split_size = min_split_size
        self.ratio_samples = ratio_samples
        self._num_trees = num_trees
        self._num_features_node = num_features_node
        self.coefficient = coefficient
        self._percentile = percentile
        self.values = values
        self._min_std_deviation = min_std_deviation

        self._non_important_features = []
        self._most_important_features = []  # not used for any performance improvements, it is just calculated
        self._trees = []

    def train(self, X_train, y_train):
        """
        Builds all trees and the nodes associated to them


        First it tries to delete those features which will most certainly not be useful: those whose standard deviation
        is less than self.min_std_deviation.
        Then initializes a number (self.num_trees) of trees with different random subsets of :parameter X_train.
        Once the trees are built, they are trained one by one, independently.

        When initializing a tree, the first node is created: its data is the whole data set, and node.depth = 0.
        The first step is to split this node. The algorithm will go over some of the features the node's data set has,
        and for each of them, goes over the values and splits the data into two data sets using each feature and value.
        Then the chosen coefficient (self.coefficient) is calculated for both data sets. If the weighted sum of these
        coefficients is lower than the last best sum, this feature, value, and the data sets are stored until there's
        a better way to split the data set.
        Once it has been through all the features and all its values, node.feature = feature, node.value = value
        and two new nodes are created using the data sets and depth = node.depth + 1.
        These two new nodes will be node's childs: node.left_child = node1, node.right_child = node2.
        If conditions are met, any of the two recently created nodes will be splitted again.

        The training process ends when no more nodes may be splitted.

        Parameters:
            :param X_train: (np.ndarray) training data
            :param y_train: (np.array) training data types
        """

        start_time1 = time.time()
        assert type(X_train) is np.ndarray

        X_train = self._delete_not_important_features(X_train)
        self._most_important_features = [0] * X_train.shape[1]

        assert not self._trees  # self._trees == []
        self._create_trees(X_train, y_train)

        for tree_index in range(self._num_trees):
            start_time2 = time.time()
            self._train_tree(tree_index)
            logger.debug('Tree {} trained in {:.6f} seconds'.format(tree_index + 1, time.time() - start_time2))
        self._get_most_important_features()
        logger.debug('most_important_features: %s', self._most_important_features)
        logger.info('length most_important_features %s', len(self._most_important_features))
        logger.info('RandomForest training: {:.6f} seconds'.format(time.time() - start_time1))

    def predict(self, X_test, y_test):
        """
        Calculates what the final prediction for X_test is.


        First it makes all trees predict :param X_test. Then goes over all trees' prediction for each sample,
        calculates the mode and appends that the final list.
        It also calculates the accuracy using :param y_test.

        Parameters:
            :param X_test: (np.ndarray) data to predict types from
            :param y_test: (np.array) correct types
        """

        start_time = time.time()
        X_test = self._delete_not_important_features(X_test)

        prediction_lists = np.asarray([tree.predict(X_test) for tree in self._trees])
        predicted = [self._mode(prediction_lists[:, sample]) for sample in range(len(X_test))]

        logger.info('RandomForest predicts: {:.6f} seconds'.format(time.time() - start_time))
        logger.debug('Predicted: %s', predicted)
        logger.debug('Correct:   %s', y_test.tolist())
        logger.info('Ratio: {:.6f}'.format(self._predicted_to_correct_ratio(predicted, y_test)))

    def _create_trees(self, X_train, y_train):
        dataset = DataSet(X_train, y_train)
        self._trees = [DecisionTree(dataset.subset_from_ratio(self.ratio_samples),
                                    self.coefficient, self.values, self._max_depth) for _ in range(self._num_trees)]

        logger.info("RandomForest built")

    def _delete_not_important_features(self, X):
        if not self._non_important_features and self._min_std_deviation > 0:
            for feature in range(X.shape[1]):
                aux = X[:, feature]
                if np.sqrt(np.var(aux)) < self._min_std_deviation:
                    self._non_important_features.append(feature)
            logger.info('length non_important_features: %s', len(self._non_important_features))
            logger.debug('non_important_features = %s', self._non_important_features)
        return np.delete(X, self._non_important_features, 1)

    def _train_tree(self, tree_index):
        self._make_childs(self._trees[tree_index].first_node, tree_index)

    def _make_childs(self, split_node, tree_index):
        tree = self._trees[tree_index]

        best_score = np.Inf
        best_feature = np.Inf
        best_value = np.Inf
        best_subset1 = None
        best_subset2 = None

        features = range(tree.dataset.X.shape[1])
        features = np.random.permutation(features)
        features = features[:self._num_features_node]

        for feature, value in ((feature, value) for feature in features
                               for value in tree.build_values(self.values, feature)):

            subset1, subset2 = split_node.dataset.divide(feature, value)   # A new better split is attempted
            score = tree.criterion.compute(subset1, subset2)

            if score < best_score:  # Checks if the new split is better
                best_subset1, best_subset2 = subset1, subset2
                best_feature = feature
                best_value = value
                best_score = score

        assert best_score != np.Inf, logging.error('score = np.Inf. features = %s', features)
        logger.debug('New division. feature: %s, value: %s, score: %s', best_feature, best_value, best_score)
        # TODO: Maybe its not necessary to log every new division
        self._most_important_features[feature] += 1
        self._add_nodes(tree_index, split_node, best_subset1, best_subset2, best_feature, best_value)

    def _add_nodes(self, tree_index, split_node, subset1, subset2, feature, value):
        split_node.feature = feature
        split_node.value = value

        node1 = self._create_node(subset1, split_node.depth, tree_index)
        node2 = self._create_node(subset2, split_node.depth, tree_index)

        split_node.left_child = node1
        split_node.right_child = node2

    def _create_node(self, subset, depth, tree_index):
        new_node = LeafNode(subset, depth + 1)

        if new_node.depth < self._max_depth:
            counts = new_node.dataset.counts()
            counts = np.delete(counts, np.argmax(counts))
            if any(count > self._min_split_size for count in counts):
                new_node = SplitNode(new_node.dataset, new_node.depth)
                self._make_childs(new_node, tree_index)
                return new_node
        return new_node

    def _get_most_important_features(self):
        min_value = int(np.percentile(self._most_important_features, self._percentile))
        self._most_important_features = [idx for idx in range(len(self._most_important_features))
                                         if self._most_important_features[idx] > min_value]
        # TODO: Could it be faster if a dictionary were used to store the most important features? Sort the dictionary
        # and get the last keys

    def _mode(self, values):
        return int(s.mode(values)[0])

    def _predicted_to_correct_ratio(self, predicted, y_test):
        correct = (np.asarray(predicted).astype(int) == np.asarray(y_test))
        return correct.sum() / len(correct)
