import numpy as np

class VNN(object):
    """Just a simple barebone implementation of a vanilla neural network."""
    def __init__(self,layers):
        self.inputLayer = layers[0]
        self.outputLayer = layers[-1]
        self.hiddenLayers = [layers[i] for i in range(1, len(layers)-1)]
        self.weights = []
        self.bias = []
        self.initWeights()
        self.initBias()

    def initWeights(self):
        self.weights.append(np.random.random((self.hiddenLayers[0],self.inputLayer)))
	for hL in range(1,len(self.hiddenLayers)):
		self.weights.append(np.random.random((self.hiddenLayers[hL],self.hiddenLayers[hL-1])))
        self.weights.append(np.random.random((self.outputLayer,self.hiddenLayers[-1])))

    def initBias(self):
	for layer in self.hiddenLayers:
	        self.bias.append(np.random.random((layer,1)))
        self.bias.append(np.random.random((self.outputLayer,1)))

    def helloNN(self):
        print("Input: {0}".format(self.inputLayer))
        print("Output: {0}".format(self.outputLayer))
        print("Hidden: {0}".format(self.hiddenLayers))
        print("Weight: {0}".format(self.weights))
        print("bias: {0}".format(self.bias))

    def sigmoid(self,z):
        return 1.0/(1 + np.exp(-z))

    def feedforward(self, a):
        for w, b in zip(self.weights, self.bias):
            a = self.sigmoid(np.dot(w,a) + b)
        return a;

    def evaluate(self):
        self.eval = self.feedforward(self.inputLayer)
        print(self.eval)

inputNumber = 784	#MNIST input
hiddenNumber = 30	#Chosen architecture
outputNumber = 10	#10 possible digits
architecture = [inputNumber, hiddenNumber, outputNumber]
vnn = VNN(architecture)
#vnn.helloNN()
vnn.evaluate()
