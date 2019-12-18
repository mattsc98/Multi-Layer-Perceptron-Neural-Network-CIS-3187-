import numpy as np
import math
import random

learnRate = 0.2  # η
errorThresh = 0.2  # μ

inp = np.array([0,1,0,1,1])
target = np.array([0,1,1])

weightInp = []
weightHid = []

dataset = [["00000",     "011"],
           ["00001",     "010"],
           ["00010",     "011"],
           ["00011",     "010"],
           ["00100",     "001"],
           ["00101",     "000"],
           ["00110",     "001"],  #t
           ["00111",     "000"],
           ["01000",     "011"],
           ["01001",     "010"],
           ["01010",     "011"],  #t
           ["01011",     "010"],
           ["01100",     "001"],
           ["01101",     "000"],
           ["01110",     "001"],
           ["01111",     "000"],  #t
           ["10000",     "111"],
           ["10001",     "110"],
           ["10010",     "111"],  #t
           ["10011",     "110"],
           ["10100",     "101"],
           ["10101",     "100"],
           ["10110",     "101"],
           ["10111",     "100"],
           ["11000",     "111"],
           ["11001",     "110"],
           ["11010",     "111"],  #t
           ["11011",     "110"],
           ["11100",     "101"],
           ["11101",     "100"],
           ["11110",     "101"],  #t
           ["11111",     "100"]
]

testing = [["00110",    "001"],  #t
           ["01010",    "011"],  #t
           ["01111",    "000"],  #t
           ["10010",    "111"],  #t
           ["11010",    "111"],  #t
           ["11110",    "101"],  #t
]

training = [["00000",     "011"],
            ["00001",     "010"],
            ["00010",     "011"],
            ["00011",     "010"],
            ["00100",     "001"],
            ["00101",     "000"],
            ["00111",     "000"],
            ["01000",     "011"],
            ["01001",     "010"],
            ["01011",     "010"],
            ["01100",     "001"],
            ["01101",     "000"],
            ["01110",     "001"],
            ["10000",     "111"],
            ["10001",     "110"],
            ["10011",     "110"],
            ["10100",     "101"],
            ["10101",     "100"],
            ["10110",     "101"],
            ["10111",     "100"],
            ["11000",     "111"],
            ["11001",     "110"],
            ["11011",     "110"],
            ["11100",     "101"],
            ["11101",     "100"],
            ["11111",     "100"]
]

# def setInputs():
#     input = ([])
#     for x in range(28):
#         input = training[x]
#         print("- " + input)

def Feedfoward(inp, target, weightInp, weightHid):

    netH = np.dot(inp, weightInp)
    outH = sigmoid(netH)

    netO = np.dot(outH, weightHid)
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
    backProb(outO, outH, target)
    return outO, outH, errList

def sigmoid(x):
    return 1/(1 + np.exp(-x))

def backProb(outputO, outputH, target):

    outputLayerDelta = (outputO)*(1-outputO)*(target-outputO)
    for i in range(4):
        for j in range(3):
            weightHid[i][j] += learnRate*outputLayerDelta[j]*outputH[i]

    for i in range(5):
        for j in range(4):
            hiddenLayerDelta = (outputH) * (1 - outputH) * addition(outputLayerDelta, weightHid[j])
            weightInp[i][j] += learnRate*hiddenLayerDelta*inp[i]

def addition(outDelta, outputOLayer):
    result = 0
    for i, delta in enumerate(outDelta):
        result += delta * outputOLayer[i]
    return result

# for x in range(10):
#     # weight1 = random.uniform(-1.0, 1.0)
#     # weight2 = random.uniform(-1.0, 1.0)
#     weightL = np.random.uniform(-1, 1, (5, 4))
#     weightH = np.random.uniform(-1, 1, (4, 3))
#
#     print(weightL)

def main():

    weightInp = np.random.uniform(-1, 1, (5, 4))
    weightHid = np.random.uniform(-1, 1, (4, 3))

    Feedfoward(inp, target, weightInp, weightHid)

# setInputs()
#make 2 arrays, 1 5x4 and 4x3
#store in here the weights


main()
