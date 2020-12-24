import numpy as np
from random import random
from sklearn.preprocessing import MinMaxScaler

a = 1
maxIteration = 100
learningRate = 0.5
hiddenlayers = [4,5,4]   #########these can be changed#########

def dataProcessing(filetrain):
    lines = filetrain.readlines()
    x = lines[0].split()
    numOfFeature = len(x) - 1

    dataSet = []
    classSet = []

    for line in lines:
        data = []
        x = line.split()
        for feature in range(numOfFeature):
            data.append(float(x[feature]))
        dataSet.append(data)
        classSet.append(int(x[numOfFeature]))

    numUniqueClass = len(np.unique(classSet))
    datasetAfterScaling = MinMaxScaler().fit_transform(dataSet)

    classSetinZeroOne = []
    for cls in classSet:
        set = []
        i = 1
        while i <= numUniqueClass:
            if i == cls:
                set.append(1)
            else:
                set.append(0)
            i += 1
        classSetinZeroOne.append(set)
    return numOfFeature, numUniqueClass, datasetAfterScaling, classSetinZeroOne

def setWeights(kr):
    weights = []
    for i in range(len(kr) - 1):
        weights.append(np.random.rand(kr[i], kr[i + 1]))
    return weights

def setderivatives(kr):
    derivatives = []
    for i in range(len(kr) - 1):
        derivatives.append(np.zeros((kr[i], kr[i + 1])))
    return derivatives

def setActivations(kr):
    activations = []
    for i in range(len(kr)):
        activations.append(np.zeros(kr[i]))
    return activations

def sigmoidFunc(x):
    return 1.0 / (1 + np.exp(-x))

def backwardComputation(error, weightsBetweenLayers, derivatives, activations):
    for i in reversed(range(len(derivatives))):
        activation = activations[i + 1]  ####activation for previous layer
        delta = error * (a * activation * (1.0 - activation))  #### derivative of logistic function
        deltaReshaped = delta.reshape(delta.shape[0], -1)
        currentActivation = activations[i].reshape(activations[i].shape[0], -1)
        derivatives[i] = np.dot(currentActivation, deltaReshaped.T)
        error = np.dot(delta, weightsBetweenLayers[i].T)  # error for next

    return derivatives

def trainMLP(traindataSet, trainclassSet, weightsBetweenLayers, derivatives, activations):
    for i in range(maxIteration):
        for j, data in enumerate(traindataSet):
            actualClass = trainclassSet[j]


            ##### the forward computation part ######
            k = 0
            currentActivation = data
            activations[k] = currentActivation   # save the activations for backwardComputation

            for w in weightsBetweenLayers:
                k += 1
                product = np.dot(currentActivation, w)
                currentActivation = sigmoidFunc(product)
                activations[k] = currentActivation  # save the activations for backwardComputation
            predictedClass = currentActivation


            #error = (actualClass - predictedClass) ** 2
            error = predictedClass - actualClass

            derivatives = backwardComputation(error, weightsBetweenLayers, derivatives, activations)

            ##### Updating the weights ######
            for l in range(len(weightsBetweenLayers)):
                derivative = derivatives[l]
                weightsBetweenLayers[l] -= derivative * learningRate

    return weightsBetweenLayers

def test(trainedWeights, testdataSet):
    output = testdataSet
    for w in trainedWeights:
        product = np.dot(output, w)
        output = sigmoidFunc(product)
    return output

def accuracy(predictedClass, testclassSet):
    incorrect = 0
    predictedClass = predictedClass.tolist()
    for i in range(len(predictedClass)):
        predicted = predictedClass[i].index(max(predictedClass[i]))
        actual = testclassSet[i].index(max(testclassSet[i]))
        #print("{}. Predicted Class --> {} ----- {} <-- Actual Class".format(i+1, predicted, actual))
        if predicted != actual:
            incorrect += 1
            print("{}. Predicted Class --> {} ----- {} <-- Actual Class".format(i + 1, predicted, actual))

    accuracy = float(((len(predictedClass) - incorrect) / len(predictedClass)) * 100)
    print("Accuracy :", accuracy, "%")
    return accuracy


if __name__ == "__main__":
    filetrain = open("trainNN.txt")
    numOfFeatureinTrain, numOfClassinTrain, traindataSet, trainclassSetinZeroOne = dataProcessing(filetrain)


    kr = []
    kr.append(numOfFeatureinTrain)
    for i in hiddenlayers:
        kr.append(i)
    kr.append(numOfClassinTrain)

    weightsBetweenLayers = setWeights(kr)  #### initial weights between two layers
    derivatives = setderivatives(kr)       #### derivatives between two layers
    activations = setActivations(kr)       #### activations per layer

    trainedWeights = trainMLP(traindataSet, trainclassSetinZeroOne, weightsBetweenLayers, derivatives, activations)

    filetest = open("testNN.txt")
    numOfFeatureinTest, numOfClassinTest, testdataSet, testclassSetinZeroOne = dataProcessing(filetest)

    predictedclassSet = test(trainedWeights, testdataSet)    #### running the test file
    #print(predictedclassSet)

    accuracy = accuracy(predictedclassSet, testclassSetinZeroOne)     ##### calculating accuracy

    #reportfile = open("report.txt", "a")
    #reportfile.write("no. of hidden layers = " + str(len(hiddenlayers)) + " ----- no. of nodes/hidden layer = " + str(
    #    hiddenlayers) + " ----- accuracy = " + str(accuracy) + "%\n")
    #reportfile.close()