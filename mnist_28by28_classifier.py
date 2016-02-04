from __future__ import print_function
import time
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split
from sklearn.datasets import load_digits
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import LabelBinarizer
from ann import ANN


# load lecun mnist dataset
X = []
y = []
#with open('data/mnist_train_data.txt', 'r') as fd, open('data/mnist_train_label.txt', 'r') as fl:
with open('data/mnist_test_data.txt', 'r') as fd, open('data/mnist_test_label.txt', 'r') as fl:
	for line in fd:
		img = line.split()
		pixels = [int(pixel) for pixel in img]
		X.append(pixels)
	for line in fl:
		pixel = int(line)
		y.append(pixel)
X = np.array(X, np.float)
y = np.array(y, np.float)

#X_val = X[55000:]
#y_val = y[55000:]

#X = X[:55000]
#y = y[:55000]

# shuffle
def shuffle_in_unison_inplace(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]

X, y = shuffle_in_unison_inplace(X, y)

print("first A", X[1])
print("first B", y[1])
print("X.shape", X.shape)
print("y.shape", y.shape)

print("X.shape", X.shape)
print("y.shape", y.shape)

# normalize input into [0, 1]
X -= X.min()
X /= X.max()

# split data into training and testing 75% of examples are used for training and 25% are used for testing
X_train, y_train = X, y
X_test, y_test = X, y ##############################3
#########################################################

# binarize the labels from a number into a vector with a 1 at that index
# ex: label 4 -> binarized [0 0 0 0 1 0 0 0 0 0]
# ex: label 7 -> binarized [0 0 0 0 0 0 0 1 0 0]
labels_train = LabelBinarizer().fit_transform(y_train)
#labels_test = LabelBinarizer().fit_transform(y_test)

# convert from numpy to normal python list for our simple implementation
X_train_l = X_train.tolist()
labels_train_l = labels_train.tolist()

# free memory
X = None
y = None


def step_cb(nn, step):
	print("ping")
	nn.serialize(nn, str(step) + ".pickle")

# load or create an ANN
nn = ANN([1,1])
serialized_name = '28_1000000.pickle'

if os.path.exists(serialized_name):
	# load a saved ANN
	nn = nn.deserialize(serialized_name)
else:
	# create the ANN with:
	# 1 input layer of size 64 (the images are 8x8 gray pixels)
	# 1 hidden layer of size 100
	# 1 output layer of size 10 (the labels of digits are 0 to 9)
	nn = ANN([784, 300, 10])

	# see how long training takes
	startTime = time.time()

	# train it
	nn.train2(30, X_train_l, labels_train_l, 100000, step_cb)

	elapsedTime = time.time() - startTime
	print("Training took {0} seconds".format(elapsedTime))

	# serialize and save the ANN
	nn.serialize(nn, serialized_name)

# compute the predictions

predictions = []
for i in range(X_test.shape[0]):
	o = nn.predict(X_test[i])
	# the inverse of the binarization would be taking the maximum argument index
	# ex: [.1 .1 .1 .1 .9 .1 .1 .1 .1 .1] -> 4
	# ex: [.1 .1 .1 .1 .1 .1 .1 .9 .1 .1] -> 7
	predictions.append(np.argmax(o))

# compute a confusion matrix
print("confusion matrix")
print(confusion_matrix(y_test, predictions))

# show a classification report
print("classification report")
print(classification_report(y_test, predictions))
