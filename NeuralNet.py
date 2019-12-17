import numpy
import math
import random

def NeuralNet(measure1, measure2, weight1, weight2, bias):
    z = measure1 * weight1 + measure2 * weight2 + bias
    return sigmoid(z)

def sigmoid(x):
    return 1/(1 + numpy.exp(-x))

for x in range(10):
    weight1 = random.uniform(-1.0, 1.0)
    weight2 = random.uniform(-1.0, 1.0)
    bias = numpy.random.rand()

    print(weight1)

learnRate = 0.2
errorThresh = 0.2
