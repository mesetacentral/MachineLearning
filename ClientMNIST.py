from SaveMNIST import load
import numpy as np
from RandomForest import RandomForest


X_train, y_train, X_test, y_test = load()

# HYPER PARAMETERS
max_depth = 12
min_split_size = 5
ratio_samples = 0.35
num_trees = 5
num_features_node = 28  # int(np.sqrt(X_train.shape[1]))
coefficient = 'Gini'  # 'MSE' works but is not fully implemented
percentile = 90  # does not affect accuracy nor performance
values = [1]  # this works because almost all values are 0 or 255, there are very few inbetweeners
min_std_deviation = 90
# Joel Guevara and his team found that values = [1] works as fine as [32, 64, 96, 128, 160, 192, 224]


rf = RandomForest(max_depth, min_split_size, ratio_samples, num_trees,
                  num_features_node, coefficient, percentile, values, min_std_deviation)
rf.train(X_train, y_train)
rf.predict(X_test, y_test)
