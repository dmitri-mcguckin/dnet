import numpy

from ftfutils import log, Mode # TODO: remove this when done

DEBUG = False

class NeuralNetwork:
    def __init__(self, labels, data, output, layers=1, width=None, beta=1, momentum=0.75, bias=1, weight=None):
        self.labels = labels # The labels (aka targets)
        self.data = data # The input data
        self.output = output # The number of output nodes
        self.layers = layers # The number of layers in the network
        if(width is None): self.width = len(self.data) # The width of each layer
        else: self.width = width #   - Defaults to width of input
        self.beta = beta # Hyperparameter for... something...
        self.momentum = momentum # Hyper parameter for gradient descent
        self.bias = bias # Hyper parameter for bias
        if(weight is None): self.weights = numpy.random.rand(self.layers + 1, self.width) - 0.5
        else: self.weights = numpy.tile(weight, (self.layers + 1, self.width))

    # Sigmoid activation function
    def activate(self, input): return numpy.around(1 / (1 + numpy.exp(-1 * input)), 3)

    # Softmax for output layer
    def softmax(self, input): return input

    def feed_forward(self, input, layer=0):
        if(DEBUG): log(Mode.DEBUG, "W: " + str(self.weights))
        # If no args, kick off the feed forward
        # if(input is None): return self.feed_forward(self.data[0], 0)
        # If we've recured down all layers, process output layer
        if(layer == self.layers):
            if(DEBUG): log(Mode.DEBUG, "Reached layer #" + str(layer) + " which is the last hidden layer, with data: " + str(input))
            output = []
            for node in range(0, self.output):
                h = self.weights[layer][0] * self.bias
                for data in range(0, self.width - 1):
                    h += self.weights[layer][data + 1] * self.data[layer][data]
                output.append(self.softmax(self.activate(h)))
            return output

        if(DEBUG): log(Mode.DEBUG,  "N: " + str(self.width) + " | W: " + str(self.weights[layer]) + " | D: " + str(input))

        activations = []
        for node in range(0, self.width):
            # Start with the product of the bias and weight
            h = self.weights[layer][0] * self.bias
            # Sum the product of weights and inputs
            for data in range(0, self.width - 1):
                h += self.weights[layer][data + 1] * self.data[layer][data]
            activations.append(self.activate(h))

        return self.feed_forward(activations, layer + 1)
