# Decide and Fix Network Topology				# of layers, types of neurons, etc.
# Randomize all weights (-1 to +1)
# Shuffle (randomize) the training examples			Do this once before training starts
# Set Eta = 0.2, Mu = 0.2
import numpy as np


class MLP:
    def __init__(self, x, y):
        self.input = x
        self.weights1 = np.random.rand(self.input.shape[1], 4)
        self.weights2 = np.random.rand(4, 1)
        self.y = y
        self.output = np.zeros(y.shape)

    def feedforward(self):
        self.layer1 = sigmoid(np.dot(self.input, self.weights1))
        self.output = sigmoid(np.dot(self.layer1, self.weights2))

