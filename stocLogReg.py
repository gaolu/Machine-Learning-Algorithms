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

# core stochastic gradient ascent algorithm
def stocGradAscent(dataMatrix, labels, numIter):
    numOfRows, numOfCols = shape(dataMatrix)
    theta = ones(numOfCols)
    labelMatrix = labels
    
    for j in range(numIter):
        dataIndex = range(numOfRows)
        for i in range(numOfRows):
            alpha = 4 / (1.0 + j + i) + 0.01
            randIndex = int(random.uniform(0, len(dataIndex)))
            g = sigmoid(sum(dataMatrix[randIndex] * theta))
            error = labelMatrix[randIndex] - g
            theta = theta + alpha * error * dataMatrix[randIndex]
            del(dataIndex[randIndex])
    return theta

def main():
    # train the classifier
    fileName = 'bclass-train'
    fLength = fileLength(fileName)
    loadDataSet(fileName, fLength)
    numIter = 150
    theta = stocGradAscent(dataMat, labelMat, numIter)
    
    # predict sample classifications
    fileName = 'bclass-test'
    fLength = fileLength(fileName)
    loadDataSet(fileName, fLength)
    predictResult = predict(dataMat, labelMat, theta)
    
    # calculate error rate
    errorRate = getErrorRate(predictResult)
    print errorRate

# start from main()
if __name__ == "__main__":
    main()
