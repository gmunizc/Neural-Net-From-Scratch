import vanillaNN as vanilla
import mnist_loader

def main():

    inputNumber = 784	#MNIST input
    hiddenNumber = 30	#Chosen architecture
    outputNumber = 10	#10 possible digits
    architecture = [inputNumber, hiddenNumber, outputNumber]
    vnn = vanilla.Network(architecture, vanilla.MSECost)

    training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
    alpha = 3.0
    epochs = 30
    batchSize = 10
    vnn.train(trainingSet, batchSize, alpha, epochs)

    vnn.evaluate(trainingSet[0])

main()
