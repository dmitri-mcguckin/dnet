import numpy

from ftfutils import log, Mode # TODO: remove this when done

DEBUG = False

class NeuralNetwork:
    def __init__(self, labels, data, output_size, layers=1, width=None, learning_rate=1, momentum=0.75, bias=1, weight=None, accuracy=3, rsoftmax=False):
        self.labels = labels # The labels (aka targets)
        self.data = data # The input data

        self.output_size = output_size # The number of output nodes
        self.activations = [] # The activation table of the last set of input data
        self.layers = layers # The number of layers in the network

        if(width is None): self.width = len(self.data) # The width of each layer
        else: self.width = width #   - Defaults to width of input

        self.learning_rate = learning_rate # Hyperparameter for learning rate (gradient descent)
        self.momentum = momentum # Hyper parameter for minima (gradient descent)
        self.bias = bias # Hyper parameter for bias

        if(weight is None): self.weights = numpy.random.rand(self.layers + 1, self.width + 1) - 0.5
        else: self.weights = numpy.tile(weight, (self.layers + 1, self.width + 1))

        self.accuracy = accuracy # Accuracy of all float values
        self.rsoftmax = rsoftmax

    # Sigmoid activation function
    def activate(self, input): return 1 / (1 + numpy.exp(-1 * input))

    # Softmax for output layer
    def softmax(self, input):
        e_o_k = 0
        softmaxes = []
        for o_k in input: e_o_k += numpy.exp(o_k)
        for o_i in input: softmaxes.append(numpy.around(numpy.exp(o_i) / e_o_k, self.accuracy))
        return softmaxes

    # Single-row feed forward function
    def feed_forward(self, input, layer=0):
        # If layer is 0, we're starting a new epoch, wipe the activation table
        if(layer == 0): self.activations = []

        if(DEBUG): log(Mode.DEBUG, "L" + str(layer) + " Weight Table: " + str(self.weights[layer]))

        # If we've recured down all layers, process output layer and return
        if(layer == self.layers):
            output_layer = []
            for _ in range(0, self.output_size):
                h = self.weights[layer][0] * self.bias # Start with adding the bias to the node calculation
                h += numpy.matmul(self.weights[layer][1:], input) # Sum the product of weights and inputs
                output_layer.append(self.activate(h))
            self.activations.append(output_layer)
            if(self.rsoftmax): return self.softmax(output_layer)
            else: return numpy.around(output_layer, self.accuracy)

        # Recur down the layers normally
        activations = []
        for node in range(0, self.width):
            # Start with the product of the bias and weight
            h = self.weights[layer][0] * self.bias # Start with adding the bias to the node calculation
            h += self.weights[layer][2:] * input[layer] # Sum the product of weights and inputs
            activations.append(self.activate(h))
        self.activations.append(activations)

        return self.feed_forward(activations, layer + 1)

    def back_propogate(self, layer, target=None, ptaus=None):
        if layer < 0: return

        if(DEBUG): log(Mode.DEBUG, "Driving back propogation for L" + str(layer))

        # Calculate the taus dependant on hidden or output layer
        taus = []
        deltas = []
        if(target is not None and ptaus is None): # Output layer
            for i in range(0, self.width + 1):
                for h in self.activations[layer]:
                    taus.append(h * (1 - h) * (target - h))

            deltas.append(self.learning_rate * taus[0] * self.bias)
            for i in range(0, self.width):
                deltas.append(self.learning_rate * taus[i] * self.activations[layer - 1][i])
            self.weights[layer] += deltas
        elif(target is None and ptaus is not None): # Hidden layer
            sum = self.weights[layer] * ptaus
            for i in range(0, self.width):
                h = self.activations[layer][i]
                taus.append(h * (1 - h) * sum[0])

            deltas.append(self.learning_rate * taus[0] * self.bias)
            for i in range(0, self.width):
                deltas.append(self.learning_rate * taus[i] * self.bias)
            input = []
            input.append(self.bias)
            input = numpy.append(input, self.data[layer])
            deltas[1:] *= input
            self.weights[layer] += deltas

        else:
            log(Mode.ERROR, "Invalid set of arguments for backprop at L" + str(layer))
            return

        return self.back_propogate(layer - 1, ptaus=taus)
