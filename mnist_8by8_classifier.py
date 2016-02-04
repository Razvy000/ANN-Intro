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


# http://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html#sklearn.datasets.load_digits

def show_training_digits(digits):
	images_and_labels = list(zip(digits.images, digits.target))
	for index, (image, label) in enumerate(images_and_labels[:4]):
		plt.subplot(2, 4, index + 1)
		plt.axis('off')
		plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
		plt.title('Training: %i' % label)


def show_prediction_digits(digits, images, imgshape, predicted):
	n_samples = len(digits.images)
	images_and_predictions = list(zip(images, predicted))
	for index, (image, prediction) in enumerate(images_and_predictions[:4]):
		plt.subplot(2, 4, index + 5)
		plt.axis('off')
		# transform vector to image for imshow
		img = np.array(image, np.float)
		img = img.reshape(imgshape)
		plt.imshow(img, cmap=plt.cm.gray_r, interpolation='nearest')
		plt.title('Prediction: %i' % prediction)


# import the simplified mnist dataset from scikit learn
digits = load_digits()

# get the input vectors (X is a vector of vectors of type int)
X = digits.data

# get the output vector ( y is a vector of type int)
y = digits.target

print("X.shape", X.shape)
print("y.shape", y.shape)

# visualize some training examples
show_training_digits(digits)
plt.show()

# normalize input into [0, 1]
X -= X.min()
X /= X.max()

# split data into training and testing 75% of examples are used for training and 25% are used for testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

# split again for validation
X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.1)

# binarize the labels from a number into a vector with a 1 at that index
# ex: label 4 -> binarized [0 0 0 0 1 0 0 0 0 0]
# ex: label 7 -> binarized [0 0 0 0 0 0 0 1 0 0]
labels_train = LabelBinarizer().fit_transform(y_train)
labels_test = LabelBinarizer().fit_transform(y_test)
labels_valid = LabelBinarizer().fit_transform(y_valid)

# convert from numpy to normal python list for our simple implementation
X_train_l = X_train.tolist()
X_valid_l = X_valid.tolist()

labels_train_l = labels_train.tolist()
labels_valid_l = labels_valid.tolist()

steps = [] #[1, 2, 3, 4]
train_error = [] #[50, 40, 20, 20]
validation_err = [] #[70, 65, 63, 60]




def evaluate(X_t, y_t, X_v, y_v):
	def step_cb(nn, step):
		training_error = nn.get_avg_error(X_t, y_t)
		testing_error = nn.get_avg_error(X_v, y_v)

		steps.append(step)
		train_error.append(training_error)
		validation_err.append(testing_error)
		print("Terr {0}   Verr {1}".format(training_error, testing_error))
		nn.serialize(nn, str(step) + ".pickle")

	return step_cb

# load or create an ANN
nn = ANN([1,1])
serialized_name = 'models/nn_mnist_8by8_5epochs.pickle'

if os.path.exists(serialized_name):
	# load a saved ANN
	nn = nn.deserialize(serialized_name)
else:
	# create the ANN with:
	# 1 input layer of size 64 (the images are 8x8 gray pixels)
	# 1 hidden layer of size 100
	# 1 output layer of size 10 (the labels of digits are 0 to 9)
	nn = ANN([64, 100, 10])

	# see how long training takes
	startTime = time.time()

	# train it
	nn.train2(5, X_train_l, labels_train_l, 1000, evaluate(X_train_l, labels_train_l, X_valid_l, labels_valid_l))

	elapsedTime = time.time() - startTime
	print("Training took {0} seconds".format(elapsedTime))

	# serialize and save the ANN
	nn.serialize(nn, serialized_name)

	# plot error over time
	#plt.plot(step, train_error, 'b--', step, validation_err, 'gs', t, t**3, 'g^')
	#plt.plot(step, train_error, 'b--', step, validation_err, 'g')
	plt.plot(steps, train_error, 'b--', label="Training Error")
	plt.plot(steps, validation_err, 'g', label="Validation Error")
	plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=2, mode="expand", borderaxespad=0.)
	plt.title('Training Error vs Validation Error')
	plt.show()

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

# visualize
show_training_digits(digits)
show_prediction_digits(digits, X_test, (8, 8,), predictions)
plt.show()

# 94%-97% precision 94-97% recall




print("trained on")
for i in range(10):
	out = nn.predict(X_train[i])
	out = np.array(out)
	softmax = np.exp(out) / np.sum(np.exp(out), axis= 0)
	predictNo = np.argmax(softmax)
	print(y_train[i], (100 * softmax).astype(int))


print("tested on")
for i in range(10):
	out = nn.predict(X_test[i])
	out = np.array(out)
	# what happens when u scale it?
	# out = 100 * out
	softmax = np.exp(out) / np.sum(np.exp(out), axis= 0)
	predictNo = np.argmax(softmax)
	print(y_test[i], (100 * softmax).astype(int))