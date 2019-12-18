import numpy as np
import math
import random

dataset = [0,0, 0, 0, 0 ,     0, 1, 1,
           0, 0, 0, 0, 1,     0, 1, 0,
           0, 0, 0, 1, 0,     0, 1, 1,
           0, 0, 0, 1, 1,     0, 1, 0,
           0, 0, 1, 0, 0,     0, 0, 1,
           0, 0, 1, 0, 1,     0, 0, 0,
           0, 0, 1, 1, 0,     0, 0, 1,  #t
           0, 0, 1, 1, 1,     0, 0, 0,
           0, 1, 0, 0, 0,     0, 1, 1,
           0, 1, 0, 0, 1,     0, 1, 0,
           0, 1, 0, 1, 0,     0, 1, 1,  #t
           0, 1, 0, 1, 1,     0, 1, 0,
           0, 1, 1, 0, 0,     0, 0, 1,
           0, 1, 1, 0, 1,     0, 0, 0,
           0, 1, 1, 1, 0,     0, 0, 1,
           0, 1, 1, 1, 1,     0, 0, 0,  #t
           1, 0, 0, 0, 0,     1, 1, 1,
           1, 0, 0, 0, 1,     1, 1, 0,
           1, 0, 0, 1, 0,     1, 1, 1,  #t
           1, 0, 0, 1, 1,     1, 1, 0,
           1, 0, 1, 0, 0,     1, 0, 1,
           1, 0, 1, 0, 1,     1, 0, 0,
           1, 0, 1, 1, 0,     1, 0, 1,
           1, 0, 1, 1, 1,     1, 0, 0,
           1, 1, 0, 0, 0,     1, 1, 1,
           1, 1, 0, 0, 1,     1, 1, 0,
           1, 1, 0, 1, 0,     1, 1, 1,  #t
           1, 1, 0, 1, 1,     1, 1, 0,
           1, 1, 1, 0, 0,     1, 0, 1,
           1, 1, 1, 0, 1,     1, 0, 0,
           1, 1, 1, 1, 0,     1, 0, 1,  #t
           1, 1, 1, 1, 1,     1, 0, 0
]

def NeuralNet(inp, target, weightL, weightH):
    inp = np.array([0,1,0,1,1])
    target = np.array([0,1,1])

    netH = np.dot(inp, weightL)
    outH = sigmoid(netH)

    netO = np.dot(outH, weightH)
    outO = sigmoid(netO)

    errList = np.array([0, 0, 0])
    i = 0
    for x in outO:
        errList[i] = np.subtract(target[i], x)
        i+=1

    print("INPUT\n", inp)
    print("TARGET\n", target)
    print("NETH\n", netH)
    print("OUTH\n", outH)
    print("NETO\n", netO)
    print("OUTO\n", outO)#
    print("ERRORLIST\n", errList)

    return outO, outH, errList

def sigmoid(x):
    return 1/(1 + np.exp(-x))

# for x in range(10):
#     # weight1 = random.uniform(-1.0, 1.0)
#     # weight2 = random.uniform(-1.0, 1.0)
#     weightL = np.random.uniform(-1, 1, (5, 4))
#     weightH = np.random.uniform(-1, 1, (4, 3))
#
#     print(weightL)

learnRate = 0.2
errorThresh = 0.2
input = 0
target = 0

weightL = np.random.uniform(-1, 1, (5, 4))
weightH = np.random.uniform(-1, 1, (4, 3))
NeuralNet(input, target, weightL, weightH)

#make 2 arrays, 1 5x4 and 4x3
#store in here the weights