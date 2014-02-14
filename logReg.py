import numpy
from numpy import *

labelMat = {}
dataMat = {}

# load data set
def loadDataSet(fileName):
    data = loadtxt(fileName, delimiter = '\t')
    global labelMat
    labelMat = data[:, 0 : 1]
    global dataMat
    dataMat = data[:, 1: ]

# sigmoid function computation
def sigmoid(matrix):
    result = 1.0 / (1 + exp(-1.0 * matrix))
    #print len(result), len(result[0])
    return 1.0 / (1 + exp(-1.0 * matrix))

# core gradient ascent algorithm
def gradAscent(dataMatrix, labels, maxCycles):
    #dataMatrix = mat(dataMatIn)
    labelMatrix = labels
    numOfRows, numOfCols = shape(dataMatrix)
    #print numOfRows, numOfCols
    
    # define the step size
    alpha = 0.001
    
    # initialize theta to be an all 1's array
    theta = ones((numOfCols, 1))
    
    for i in range(maxCycles):
        g = sigmoid(dataMatrix.dot(theta))
        #print len(g), len(g[0])
        error = (labelMatrix - g)
        #print len(error), len(error[0])
        theta = theta + alpha * dataMatrix.transpose().dot(error)
    
    return theta

# classify test samples
def classifyVector(sampleVector, theta):
    prob = sigmoid(sampleVector.dot(theta))
    if prob > 0.5: return 1
    else: return -1

# predict classification
def predict(dataMatrix, labels, theta):
    labelMatrix = labels
    numOfSamples, numOfFeatures = shape(dataMatrix)
    #print numOfSamples, numOfFeatures
    predictResult = []
    for i in range(numOfSamples):
        dataVector = dataMatrix[i]
        prob = classifyVector(dataVector, theta)
        predictResult.append(prob)
    return predictResult

# calculate error rate
def getErrorRate(predictResult, fileName):
    loadDataSet(fileName)
    totalNum = len(predictResult)
    errorNum = 0

    for i in range(totalNum):
        if predictResult[i] != labelMat[i]:
            errorNum = errorNum + 1
    return (1.0 * errorNum) / (1.0 * totalNum)

# calculate different norms of a matrix
def getVectorNorm(dataVector, norm):
    return numpy.linalg.norm(dataVector, ord = norm)

# normalize vectors
def normalizeVector(dataVector, norm):
    vectorLen = len(dataVector)
    for i in range (vectorLen):
        dataVector[i] = dataVector[i] / norm
    return dataVector

# normalize the whole matrix
def normalizeMatrix(dataMatrix, norm):
    numOfRows, numOfCols = shape(dataMatrix)
    for i in range (numOfRows):
        dataMatrix[i] = normalizeVector(dataMatrix[i], getVectorNorm(dataMatrix[i], norm))
    return dataMatrix

def main():
    maxCycles = 200
    norm = 2
    errorRate = {}
    for i in range(maxCycles):
        # train the classifier
        fileName = 'bclass-train'
        loadDataSet(fileName)
        normalizeMatrix(dataMat, norm)
        theta = gradAscent(dataMat, labelMat, i)
    
        # predict test samples
        fileName = 'bclass-test'
        loadDataSet(fileName)
        normalizeMatrix(dataMat, norm)
        predictResult = predict(dataMat, labelMat, theta)
    
        # get error rate
        errorRate[i] = getErrorRate(predictResult, fileName)

    print errorRate[100]


# start from main()
if __name__ == "__main__":
    main()
