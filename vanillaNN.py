import numpy as np

class VNN(object):
    """Just a simple barebone implementation of a vanilla neural network."""
    def __init__(self,input,output, hidden):
        self.input = input
        self.output = output
        self.hidden = hidden
        self.weight = []
        self.bias = []
        self.initWeights()
        self.initBias()

    def initWeights(self):
        self.weight.append(np.random.random((self.input.shape[1],self.hidden)))
        self.weight.append(np.random.random((self.hidden,self.output)))

    def initBias(self):
        self.bias.append(np.random.random((1,self.hidden)))
        self.bias.append(np.random.random((1,self.output)))

    def helloNN(self):
        print("Input: {0}".format(self.input))
        print("Output: {0}".format(self.output))
        print("Hidden: {0}".format(self.hidden))
        print("Weight: {0}".format(self.weight))
        print("bias: {0}".format(self.bias))

    def sigmoid(self,z):
        return 1.0/(1 + np.exp(-z))

    def feedforward(self, a):
        for w, b in zip(self.weight, self.bias):
            a = self.sigmoid(np.dot(a,w) + b)
        return a;

    def evaluate(self):
        self.eval = self.feedforward(self.input)
        print(self.eval)

inputNumber = np.array([[1, 2, 3],[9, 8, 7]])
outputNumber = 2
hiddenNumber = 10
vnn = VNN(inputNumber,outputNumber, hiddenNumber)
vnn.helloNN()
vnn.evaluate()
