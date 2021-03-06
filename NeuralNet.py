import numpy as np
import matplotlib.pyplot as plt

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

# ___________________________________________________________________________________________ #

learnRate = 0.2  #η
errorThresh = 0.2  #μ

#inp = np.array([0,1,0,1,1])
#tar = np.array([0, 1, 1])

# weightHid = np.zeros((4, 3), np.float)
# weightInp = np.zeros((5, 4), np.float)
weightInp = np.random.uniform(-1, 1, (5, 4))
weightHid = np.random.uniform(-1, 1, (4, 3))

badfacts = 0


def Feedfoward(inp, target, weightInp, weightHid, badfacts):

    netH = np.dot(inp, weightInp)
    outH = sigmoid(netH)

    netO = np.dot(outH, weightHid)
    outO = sigmoid(netO)

    errList = np.array([0.0, 0.0, 0.0])

    badOut = 0

    for i in range(3):
        errList[i] = np.subtract(target[i], outO[i])

        if(abs(errList[i]) > errorThresh):
            badOut += 1

    if(badOut > 0):
        badfacts += 1

    print("\nInput:\n", inp)
    print("Target:\n", target)
    #print("NETH\n", netH)
    #print("OUTH\n", outH)
    #print("NETO\n", netO)
    print("Output: \n", outO)
    print("Error List: \n", errList)
    #print("TESTLIST\n", testList)
    # if (badOut > 0):
    #     backProb(outO, outH, target, inp)
    #Feedforward Result:
    return outO, outH, badfacts, badOut

def sigmoid(x):
    return 1/(1 + np.exp(-x))

def backProb(outputO, outputH, target, inp):

    outputLayerDelta = (outputO) * (1-outputO)*(target-outputO)
    for i in range(4):
        for j in range(3):
            weightHid[i][j] += learnRate*outputLayerDelta[j]*outputH[i]

    for i in range(5):
        for j in range(4):
            hiddenLayerDelta = (outputH) * (1 - outputH) * addition(outputLayerDelta, weightHid[j])
            # print(learnRate)
            # print(hiddenLayerDelta[j])
            # print(inp[i])
            weightInp[i][j] += learnRate*hiddenLayerDelta[j]*inp[i]

    # print("\n\nOUTPU  TDELTA\n", outputLayerDelta)
    # print("WEIGHTHIDDEN\n", weightHid)
    # print("HIDDENDELTA\n", hiddenLayerDelta)
    # print("WEIGHTINP\n", weightInp)


def addition(outDelta, outputOLayer):
    result = 0
    for i, delta in enumerate(outDelta):
        result += delta * outputOLayer[i]
    return result

def plotGraph(x):
    plt.plot(x)

    plt.xlabel('Epochs')
    plt.ylabel('Bad Facts')
    plt.show()

#method to prepare dataset to be entered into NN
def setInputs(data, input, target):
    for x in range(len(data)):
        temp = list(data[x][0])
        temp_list = []
        for i in range(5):
            temp_list.append(int(temp[i]))
        input.append(temp_list)
        #print("inp - " + str(input[x]))

        temp2 = list(data[x][1])
        temp_list2 = []
        for i in range(3):
            temp_list2.append(int(temp2[i]))
        target.append(temp_list2)
        #print("tar - " + str(target[x]))

    # print("Input: ")
    # print(input)
    # print("Target: ")
    # print(target)
    return input, target

def main():
    #weightInp = np.random.uniform(-1, 1, (5, 4))
    #weightHid = np.random.uniform(-1, 1, (4, 3))

    train_input = []
    train_target = []

    training_input, training_target = setInputs(training, train_input, train_target)

    test_input = []
    test_target = []

    testing_input, testing_target = setInputs(training, test_input, test_target)
    zeros = 500 #max number of estimated epochs, increase if array out of bounds shows up
    badfactsGraph = np.zeros(zeros) #create empty list and fill with 0s
    print("-----TRAINING-----")
    badfacts = 1
    epoch = 0
    while (badfacts != 0):
    #for epoch in range(200):
        badfacts = 0
        for i in range(25):
            outO, outH, badfacts, badOut = Feedfoward(training_input[i], training_target[i], weightInp, weightHid, badfacts)
            if (badOut > 0):
                backProb(outO, outH, training_target[i], training_input[i])
            else: continue
        badfactsGraph[epoch] = (badfacts/26)*100
        epoch += 1
        #plotGraph(badfacts, epoch)
    print("\n==============================\n")
    print("Bad Facts Graph %:\n")
    print(badfactsGraph)
    plotGraph(badfactsGraph)


    print("\n\n\n-----TESTING-----")
    badfacts = 0
    for i in range(5):
        outO, outH, badfacts, badOut = Feedfoward(testing_input[i], testing_target[i], weightInp, weightHid, badfacts)


main()

