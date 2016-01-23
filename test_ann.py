import unittest
from ann import ANN


class TestANN(unittest.TestCase):

	def test_xor_trainig(self):
		print("test_xor_trainig...")
		nn = ANN([2, 2, 1])
		inputs = [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]
		targets = [[0.0], [1.0], [1.0], [0.0]]
		predicts = []

		# train
		nn.train(40000, inputs, targets)

		for i in range(len(targets)):
			predicts.append(nn.predict(inputs[i]))

		# the prediction for 0,0 and 1,1 should be less than prediction for 0,1 and 1,0
		self.assertTrue(predicts[0] < predicts[1], 'xor relation1 not learned')
		self.assertTrue(predicts[0] < predicts[2], 'xor relation2 not learned')
		self.assertTrue(predicts[3] < predicts[1], 'xor relation3 not learned')
		self.assertTrue(predicts[3] < predicts[2], 'xor relation4 not learned')

	def test_mnist_8by8_training(self):
		print("test_mnist_8by8_training")
		import time
		import numpy as np
		import matplotlib.pyplot as plt
		from sklearn.cross_validation import train_test_split
		from sklearn.datasets import load_digits
		from sklearn.metrics import confusion_matrix, classification_report
		from sklearn.preprocessing import LabelBinarizer
		from sklearn.metrics import precision_score, recall_score

		# import the simplified mnist dataset from scikit learn
		digits = load_digits()

		# get the input vectors (X is a vector of vectors of type int)
		X = digits.data

		# get the output vector ( y is a vector of type int)
		y = digits.target

		# normalize input into [0, 1]
		X -= X.min()
		X /= X.max()

		# split data into training and testing 75% of examples are used for training and 25% are used for testing
		X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=123)

		# binarize the labels from a number into a vector with a 1 at that index
		labels_train = LabelBinarizer().fit_transform(y_train)
		labels_test = LabelBinarizer().fit_transform(y_test)

		# convert from numpy to normal python list for our simple implementation
		X_train_l = X_train.tolist()
		labels_train_l = labels_train.tolist()

		# create the artificial neuron network with:
		# 1 input layer of size 64 (the images are 8x8 gray pixels)
		# 1 hidden layer of size 100
		# 1 output layer of size 10 (the labels of digits are 0 to 9)
		nn = ANN([64, 100, 10])

		# see how long training takes
		startTime = time.time()

		# train it
		nn.train(10, X_train_l, labels_train_l)

		elapsedTime = time.time() - startTime
		print("time took " + str(elapsedTime))
		self.assertTrue(elapsedTime < 300, 'Training took more than 300 seconds')

		# compute the predictions
		predictions = []
		for i in range(X_test.shape[0]):
			o = nn.predict(X_test[i])
			predictions.append(np.argmax(o))

		# compute a confusion matrix
		# print(confusion_matrix(y_test, predictions))
		# print(classification_report(y_test, predictions))

		precision = precision_score(y_test, predictions, average='macro')
		print("precision", precision)
		recall = recall_score(y_test, predictions, average='macro')
		print("recall", recall)

		self.assertTrue(precision > 0.93, 'Precision must be bigger than 93%')
		self.assertTrue(recall > 0.93, 'Recall must be bigger than 93%')

if __name__ == '__main__':
	unittest.main()
