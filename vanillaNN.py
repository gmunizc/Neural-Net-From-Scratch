import numpy as np

class VNN(object):
    """Just a simple barebone implementation of a vanilla neural network."""
    def __init__(self,input,output, hidden):
        self.input = input
        self.output = output
        self.hidden = hidden
        self.initWeights()
        self.initBias()

    def initWeights(self):
        weight1 = np.random.random((input.shape[1],hidden))
        weight2 = np.random.random((hidden,output))

    def initBias(self):
        bias1 = np.random.random((input[0],hidden))
        bias2 = np.random.random((input[0],output))

    def helloNN(self):
        print("Input: {0}".format(self.input))
        print("Output: {0}".format(self.output))
        print("Hidden: {0}".format(self.hidden))

    def sigmoid(z):
        return 1.0/(1 + np.exp(-z))

inputNumber = np.array([[1, 2, 3],[9, 8, 7]])
outputNumber = 2
hiddenNumber = 10
vnn = VNN(inputNumber,outputNumber, hiddenNumber)
vnn.helloNN()
