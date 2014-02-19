from numpy import loadtxt, zeros, ones, shape, tile
from numpy import *
import numpy
import math

def loadDataSet(fileName):
    data = loadtxt(fileName, delimiter = '\t')
    labelMat = data[:, 0:1]
    dataMat = data[:, 1: ]
    return labelMat, dataMat

# core Locally Weighted Logistic Regression algorithm
def locWeightLogReg(labelMatrix, dataMatrix, testMatrix, tau, rowIndex):
    #print rowIndex
    lam = 1e-3
    numOfRows, numOfCols = shape(dataMatrix)
    theta = zeros((numOfCols, 1))
    
    testVector = testMatrix[rowIndex,]
    #print testVector
    repMat = tile(testVector, (numOfRows,1))
    #print (dataMatrix - repMat)
    square = power((dataMatrix - repMat), 2)
    #print square
    sumProd = square.sum(axis=1)
    #print -sumProd / (2 * tau * tau)
    #print sumProd / (2 * tau * tau)
    w_original = power(e,(-sumProd / (2 * tau * tau)))
    
    #print w_original
    
    # compute weights
    w = numpy.matrix(w_original).transpose()
    #print w
    delta = ones((numOfCols, 1))
    #print delta
    iteration = 0

    while linalg.norm(delta) > 1e-6:
        h = 1 / (1 + power(e, dot(-dataMatrix, theta)))
        #print theta
        delta = transpose(dataMatrix) * (multiply(w, (labelMatrix - h))) - lam * theta
        #print delta
        a = -transpose(dataMatrix)
        b = diag(numpy.mat(multiply(w, multiply(h, (1 - h)))))
        c = a * b
        d = c.dot(dataMatrix)
        #print d.shape
        f = lam * eye(numpy.mat(numOfCols))
        #print f.shape
        #H = (-transpose(dataMatrix) * diag(numpy.mat(multiply(w, multiply(h, (1 - h)))))).dot(dataMatrix) - lam * eye(numpy.mat(numOfCols))
        diagonal = diagflat(multiply(multiply(w, h), (1-h)).transpose())
        #print diagonal
        #print -transpose(dataMatrix).dot(diagonal)
        H = (-transpose(dataMatrix).dot(diagonal)).dot(dataMatrix) - lam * eye(numOfCols)
        #print H
        #print lam * eye(numOfCols)
        theta = theta - linalg.solve(H,delta)
        #print theta
        #print theta.shape
        iteration = iteration + 1
        #print iteration
        if iteration > 5000:
            break

#print transpose(testVector).dot(theta)
    if transpose(testVector).dot(theta) > 0:
        return 1
    else:
        return -1

def mult_lwlr(labelMatrix, dataMatrix, testMatrix, tau):
    numOfRows, numOfCols = shape(testMatrix)
    result = []
    #print numOfRows, numOfCols
    for i in range(numOfRows):
        #print i + 1
        result.append(locWeightLogReg(labelMatrix, dataMatrix, testMatrix, tau, i))

    return result

def getErrorRate(predictResult, testLabel):
    #print len(predictResult), len(testLabel)
    totalNum = len(predictResult)
    errorNum = 0
    
    for i in range(totalNum):
        if abs(predictResult[i] - testLabel[i]) > 1e-6:
            errorNum = errorNum + 1
    return (1.0 * errorNum) / (1.0 * totalNum)

def main():
    tau = 1
    fileName = 'bclass-train'
    labelMatrix, dataMatrix = loadDataSet(fileName)
    fileName = 'bclass-test'
    testLabel, testMatrix = loadDataSet(fileName)
    predictResult = mult_lwlr(labelMatrix, dataMatrix, testMatrix, tau)
    #print predictResult
    errorRate = getErrorRate(predictResult, testLabel)
    print errorRate

if __name__ == "__main__":
    main()
