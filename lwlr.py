from numpy import loadtxt, zeros, ones, shape, tile
from numpy import *
import numpy

# load data set
def loadDataSet(fileName):
    data = loadtxt(fileName, delimiter = '\t')
    labelMat = data[:, 0:1]
    dataMat = data[:, 1: ]
    return labelMat, dataMat

# core Locally Weighted Logistic Regression algorithm
def locWeightLogReg(labelMatrix, dataMatrix, testMatrix, tau, rowIndex):
    lam = 1e-3
    numOfRows, numOfCols = shape(dataMatrix)
    theta = zeros((numOfCols, 1))
    
    testVector = testMatrix[rowIndex,]
    
    # calculate weights
    repMat = tile(testVector, (numOfRows,1))
    square = power((dataMatrix - repMat), 2)
    sumProd = square.sum(axis=1)
    w_original = power(e,(-sumProd / (2 * tau * tau)))
    w = numpy.matrix(w_original).transpose()
    
    delta = ones((numOfCols, 1))
    iteration = 0
    
    # Newton's method
    while linalg.norm(delta) > 1e-6:
        h = 1 / (1 + power(e, dot(-dataMatrix, theta)))
        delta = transpose(dataMatrix) * (multiply(w, (labelMatrix - h))) - lam * theta
        diagonal = diagflat(multiply(multiply(w, h), (1-h)).transpose())
        H = (-transpose(dataMatrix).dot(diagonal)).dot(dataMatrix) - lam * eye(numOfCols)
        theta = theta - linalg.solve(H,delta)
        iteration = iteration + 1
        if iteration > 5000:
            break

    # do the classification
    if transpose(testVector).dot(theta) > 0:
        return 1
    else:
        return -1

# classify for each data entry
def mult_lwlr(labelMatrix, dataMatrix, testMatrix, tau):
    numOfRows, numOfCols = shape(testMatrix)
    result = []
    for i in range(numOfRows):
        result.append(locWeightLogReg(labelMatrix, dataMatrix, testMatrix, tau, i))
    return result

# calcualte error rate
def getErrorRate(predictResult, testLabel):
    totalNum = len(predictResult)
    errorNum = 0
    
    for i in range(totalNum):
        if abs(predictResult[i] - testLabel[i]) > 1e-6:
            errorNum = errorNum + 1
    return (1.0 * errorNum) / (1.0 * totalNum)

# main function
def main():
    tau = 0.1
    
    # load training data
    fileName = 'bclass-train'
    labelMatrix, dataMatrix = loadDataSet(fileName)
    
    # load testing data
    fileName = 'bclass-test'
    testLabel, testMatrix = loadDataSet(fileName)
    
    # do the prediction
    predictResult = mult_lwlr(labelMatrix, dataMatrix, testMatrix, tau)
    
    # calculate error rate
    errorRate = getErrorRate(predictResult, testLabel)
    print errorRate

if __name__ == "__main__":
    main()
