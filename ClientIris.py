import numpy as np
from sklearn import datasets

from RandomForest import RandomForest

iris = datasets.load_iris()

X = iris.data
y = iris.target

ratio_train_test = 0.85

num_samples, num_features = X.shape
idx = np.random.permutation(range(num_samples))
num_samples_train = int(num_samples*ratio_train_test)
idx_train = idx[:num_samples_train]
idx_test = idx[num_samples_train:]
X_train, y_train = X[idx_train], y[idx_train]
X_test, y_test = X[idx_test], y[idx_test]

# HYPER PARAMETERS
max_depth = 10
min_split_size = 5
ratio_samples = 0.5
num_trees = 10
num_features_node = int(np.sqrt(num_features))
coefficient = 'gini'
percentile = 90
values = None
min_std_deviation = 0

rf = RandomForest(max_depth, min_split_size, ratio_samples, num_trees, num_features_node, coefficient, percentile,
                  values, min_std_deviation)
rf.train(X_train, y_train)
rf.predict(X_test, y_test)
