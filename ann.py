from __future__ import print_function
import math
import random
import pickle

class ANN:

	def __init__(self, layer_sizes):

		self.layers = []
		self.learn_rate = 0.1

		for l in range(len(layer_sizes)):
			prevLayerSize = (0 if l == 0 else layer_sizes[l - 1])
			currLayerSize = layer_sizes[l]
			layer = Layer(l, currLayerSize, prevLayerSize)
			self.layers.append(layer)

	def train(self, n_epochs, inputs, targets):
		print("Training...")
		n_examples = 0
		for epoch in range(0, n_epochs):
			# if epoch % 10 == 0:
			#    print("Epoch:", epoch)
			print("Epoch:", epoch)
			for i in range(0, len(inputs)):
				n_examples += 1

				self.set_input(inputs[i])

				self.propagate_input()

				err = self.update_output_error(targets[i])

				# if n_examples % 100 == 0:
				#    print("Err:", err)

				self.backprogate_error()

				self.update_weights()

	def predict(self, input):
		self.set_input(input)
		self.propagate_input()
		return self.get_output()

	# Put data inside the network's first layer
	def set_input(self, input_vector):
		input_layer = self.layers[0]

		for i in range(0, input_layer.n_neurons):
			input_layer.output[i + use_bias] = input_vector[i]

	# Get data form the network's last layer
	def get_output(self):
		output_layer = self.layers[-1]
		res = [0] * output_layer.n_neurons
		for i in range(0, len(res)):
			res[i] = output_layer.output[i + use_bias]

		return res

	# Forward signal propagation
	def propagate_input(self):
		for l in range(len(self.layers) - 1):  # exclude last
			lower = self.layers[l]  # lower is source
			upper = self.layers[l + 1]  # upper is destination
			for j in range(0, upper.n_neurons):
				sum_in = 0
				for i in range(0, lower.n_neurons + use_bias):
					sum_in += upper.weight[i][j] * lower.output[i]
				upper.input[j] = sum_in
				upper.output[j + use_bias] = squash(sum_in)

	# The eror on the last layer is the difference between the expected target and the actual output
	def update_output_error(self, target_vector):
		output_layer = self.layers[-1]
		sum_error = 0
		for i in range(0, output_layer.n_neurons):
			neuron_output = output_layer.output[i + use_bias]
			neuron_error = target_vector[i] - neuron_output
			# slower
			# sigmoid(output_layer.input[i]) === output_layer.output[i + use_bias]
			output_layer.error[i] = deriv_squash(output_layer.input[i]) * neuron_error
			# faster
			# output_layer.error[i] = (1 - output_layer.output[i + use_bias] )*output_layer.output[i + use_bias] * neuron_error
			sum_error += neuron_error * neuron_error / 2.0
		return sum_error

	# Backward error propagation
	def backprogate_error(self):
		for l in range(len(self.layers) - 1, 0, -1):  # second param range is exclusive
			upper = self.layers[l]  # upper is source
			lower = self.layers[l - 1]  # lower is destination

			for i in range(0, lower.n_neurons):

				error = 0

				for j in range(0, upper.n_neurons):
					error += upper.weight[i + use_bias][j] * upper.error[j]

				# slower
				lower.error[i] = deriv_squash(lower.input[i]) * error
				# faster
				# lower.error[i] = (1 - lower.output[i + use_bias]) * lower.output[i + use_bias] * error

	def update_weights(self):
		for l in range(1, len(self.layers)):
			for j in range(0, self.layers[l].n_neurons):
				for i in range(0, self.layers[l - 1].n_neurons + use_bias):
					out = self.layers[l - 1].output[i]
					err = self.layers[l].error[j]

					self.layers[l].weight[i][j] += self.learn_rate * out * err
		return

	def printW(self):
		print("Weights:")
		for l in range(0, len(self.layers)):
			for j in range(0, len(self.layers[l].weight)):
				print(self.layers[l].weight[j])
			print()

	def serialize(self, nn, name):
		with open(name, 'wb') as f:
			pickle.dump(nn, f)


	def deserialize(self, name):
		with open(name, 'rb') as f:
			nn = pickle.load(f)
			return nn


class Layer:

	def __init__(self, id, layer_size, prev_layer_size):

		self.id = id
		self.n_neurons = layer_size

		self.bias_val = 1

		self.input = [0] * self.n_neurons
		self.output = [0] * (self.n_neurons + use_bias)
		self.output[0] = self.bias_val
		self.error = [0] * self.n_neurons

		self.weight = make_matrix(prev_layer_size + use_bias, self.n_neurons)
		for i in range(len(self.weight)):
			for j in range(len(self.weight[i])):
				self.weight[i][j] = between(-0.2, 0.2)

		print('Layer', self.id, 'Size', str(len(self.weight)) + ' x ' + str(len(self.weight[0]) if len(self.weight) > 0 else 0))


def sigmoid(x):
	return 1.0 / (1 + math.exp(-x))

def deriv_sigmoid(x):
	return (1 - sigmoid(x)) * sigmoid(x)


def hyperbolic_tangent(x):
	return math.tanh(x)

def deriv_hyperbolic_tangent(x):
	th = math.tanh(x)
	return 1 - th * th


# squash [-infinity, infinity] into [0, 1] or [-1, 1]
use_bias = 1 # clear way in code to remember about the bias weight that shifts the index of the weights by 1
squash = sigmoid
deriv_squash = deriv_sigmoid


# Create an N rows by M columns matrix
def make_matrix(N, M):
	return [[0 for i in range(M)] for i in range(N)]


def between(min, max):
	return random.random() * (max - min) + min


if __name__ == '__main__':

	# learn XOR
	nn = ANN([2, 2, 1])
	inputs = [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]
	targets = [[0.0], [1.0], [1.0], [0.0]]

	# predict before training
	print("Predict before training")
	for i in range(len(targets)):
		print(nn.predict(inputs[i]))
	print()

	# train
	nn.train(20000, inputs, targets)

	# weights
	# nn.printW()

	# predict after training
	print("Predict after training")
	for i in range(len(targets)):
		print(nn.predict(inputs[i]))
	