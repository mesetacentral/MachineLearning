Ignacio Cobas
_________________________

EXECUTION INSTRUCTIONS:

To execute the program automatically:
./runmnist.sh

If it doesn't work, try:
sudo ./runmnist.sh

To download and store MNIST as pickle:
python3 SaveMNISTpy

To execute the program: 
python3 ClientMNIST.py

Other data sets may be tried out too. The following don't need to be pre downloaded, client does all of the work.
python3 ClientIris.py
python3 ClientSonar.py

Hyper parameters may be changed in ClientMNIST.py (or any other Client) but that could lower the accuracy or
increase execution time.
Description of any hyper parameter can be found in RandomForest.py documentation.

Default hyper parameters for MNIST:
max_depth = 15
min_split_size = 5
ratio_samples = 0.2
num_trees = 12
num_features_node = 28
coefficient = 'gini'
percentile = 90
values = [1]
min_std_deviation = 80

With these, the execution should take 12s and the accuracy should be of 92%.

To debug the program, simply change logger level to logging.DEBUG in SaveMNIST.py
_________________________
