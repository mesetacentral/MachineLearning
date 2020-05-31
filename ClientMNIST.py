from mnist import load

from RandomForest import RandomForest

X_train, y_train, X_test, y_test = load()

# HYPER PARAMETERS
max_depth = 15
min_split_size = 5
ratio_samples = 0.3
num_trees = 15
num_features_node = 28  # int(np.sqrt(X_train.shape[1]))
coefficient = 'gini'  # 'MSE' works but is not fully implemented
percentile = 90  # does not affect accuracy nor performance
values = [1]  # this works because almost all values are 0 or 255, there are very few inbetweeners
# Joel Guevara and his team found that [1] works as fine as [32, 64, 96, 128, 160, 192, 224] but is much faster:
# takes 70s to train and predict MNIST with [32, 64, 96, 128, 160, 192, 224] but only 20s with [1]
# running on Intel® Core™ i7-7700HQ CPU @ 2.80GHz × 8 64bits.
min_std_deviation = 5


rf = RandomForest(max_depth, min_split_size, ratio_samples, num_trees,
                  num_features_node, coefficient, percentile, values, min_std_deviation)
rf.train(X_train, y_train)
rf.predict(X_test, y_test)
