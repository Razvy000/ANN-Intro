import unittest
from ann import ANN


class TestANN(unittest.TestCase):

	def test_xor_trainig(self):
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

if __name__ == '__main__':
	unittest.main()
