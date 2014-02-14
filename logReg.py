from numpy import *

labelMat = {}
dataMat = {}

# load data set
def loadDataSet(fileName, fLength):
    data = loadtxt(fileName, delimiter = '\t')
    global labelMat
    labelMat = data[:, 0 : 1]
    global dataMat
    dataMat = data[:, 1: ]

# find the number of training/test samples
def fileLength(fileName):
    with open(fileName) as fName:
        for i, l in enumerate(fName):
            pass
    fName.close()
    return i + 1

# sigmoid function computation
def sigmoid(matrix):
    result = 1.0 / (1 + exp(-1.0 * matrix))
    #print len(result), len(result[0])
    return 1.0 / (1 + exp(-1.0 * matrix))

# core gradient ascent algorithm
def gradAscent(dataMatrix, labels):
    #dataMatrix = mat(dataMatIn)
    labelMatrix = labels
    numOfRows, numOfCols = shape(dataMatrix)
    #print numOfRows, numOfCols
    
    # define the step size and max number of iterations
    alpha = 0.001
    maxCycles = 200
    
    # initialize theta to be an all 1's array
    theta = ones((numOfCols, 1))
    
    for i in range(maxCycles):
        g = sigmoid(dataMatrix.dot(theta))
        #print len(g), len(g[0])
        error = (labelMatrix - g)
        #print len(error), len(error[0])
        theta = theta + alpha * dataMatrix.transpose().dot(error)
    
    return theta

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

def getErrorRate(predictResult):
    fileName = 'bclass-test'
    fLength = fileLength(fileName)
    loadDataSet(fileName, fLength)
    totalNum = len(predictResult)
    errorNum = 0

    for i in range(totalNum):
        if predictResult[i] != labelMat[i]:
            errorNum = errorNum + 1
    return (1.0 * errorNum) / (1.0 * totalNum)

def main():
    
    # train
    fileName = 'bclass-train'
    fLength = fileLength(fileName)
    loadDataSet(fileName, fLength)
    theta = gradAscent(dataMat, labelMat)
    
    # predict
    fileName = 'bclass-test'
    fLength = fileLength(fileName)
    loadDataSet(fileName, fLength)
    predictResult = predict(dataMat, labelMat, theta)

    errorRate = getErrorRate(predictResult)
    print errorRate


# start from main()
if __name__ == "__main__":
    main()
