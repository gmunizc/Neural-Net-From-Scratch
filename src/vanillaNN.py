import numpy as np
import random

class MSECost(object):
    @staticmethod
    def fn(a,y):
        return 0.5*np.linalg.norm(a-y)**2

    @staticmethod
    def delta(z, a, y):
        return (a-y) * sigmoid_prime(z)


class Network(object):
    """Just a simple barebone implementation of a vanilla neural network."""
    def __init__(self,layers,cost):
        self.numLayers = len(layers)
        self.inputLayer = layers[0]
        self.outputLayer = layers[-1]
        self.hiddenLayers = [layers[i] for i in range(1, len(layers)-1)]
        self.weights = []
        self.bias = []
        self.cost = cost
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

    def feedforward(self, a):
        for w, b in zip(self.weights, self.bias):
            a = sigmoid(np.dot(w,a) + b)
        return a;

    def train(self,trainingSet,batchSize, alpha, epochs):
        SGD(trainingSet, batchSize, alpha, epochs)

    def SGD(self, trainingSet, batchSize, alpha, epochs):
        for i in range(0, epochs):
            random.shuffle(trainingSet)
            miniBatches = [trainingSet[i:i+batchSize] for i in range(0,len(trainingSet),batchSize)]
            for miniBatch in miniBatches:
                updateMiniBatch(miniBatch, alpha)

    def updateMiniBatch(self, miniBatch, alpha):
        for sample in miniBatch:
            gradsB, gradsW = backpropagation(sample)
            self.bias = [b - (alpha/len(miniBatch))*gradB for b, gradB in zip(self.bias,gradsB)]
            self.weights = [w - (alpha/len(miniBatch))*gradW for w, gradW in zip(self.weights,gradsW)]

    def backpropagation(self, sample):
        x, y = sample
        activations = []
        zs = []
        gradsB = [np.zeros(b.shape) for b in self.bias]
        gradsW = [np.zeros(w.shape) for w in self.weights]
        for w, b in zip(self.weights, self.bias):
            z = np.dot(w,a) + b
            zs.append(z)
            a = sigmoid(z)
            activations.append(a)

        delta = self.cost.delta(z,a,y)*sigmoid_prime(z)
        gradsB[-1] = delta
        gradsW[-1] = np.dot(activations[-2 ],delta)

        for l in range(2,len(self.bias)):
            delta = np.dot(self.weights[-l+1],delta) * sigmoid_prime(zs[-l-1])
            gradsB[-l] = delta
            gradsW[-l] = np.dot(activations[-l-1],delta)
        return (gradsB,gradsW)

    def evaluate(self, sample):
        x, y = sample
        output = self.feedforward(x)
        self.eval = max(output)
        print(self.eval)
        self.error = (y - a)
        print(self.error)

    def helloNN(self):
        print("Input: {0}".format(self.inputLayer))
        print("Output: {0}".format(self.outputLayer))
        print("Hidden: {0}".format(self.hiddenLayers))
        print("Weight: {0}".format(self.weights))
        print("bias: {0}".format(self.bias))

def sigmoid(z):
    return 1.0/(1 + np.exp(-z))

def sigmoid_prime(z):
    return sigmoid(z)*(1-sigmoid(z))
