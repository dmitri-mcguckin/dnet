import numpy
from enum import Enum
from ftfutils import log, Mode

DEBUG = False

class Type(Enum):
    linear = 0
    logistic = 1
    softmax = 2

class NeuralNetwork:
    def __init__(self, targets, data,
                 outputs=1, # Size of output layer
                 layers=1, # Number of hidden layers (not including output)
                 width=None, # Width of each layer
                 learning_rate=1, # Step size for GD
                 momentum=0.75, # Momentum for GD
                 bias=1, # Bias for each layer
                 weight=None, # Starting weight of the NN
                 accuracy=3, # Accuracy of the output
                 ret_type=Type.linear, # Return type of the NN
                 tolerance=0.01): # Error tolerance
        self.targets = numpy.around(targets, accuracy) # The target values for training
        self.data = [] # The training/test data

        # Add the bias into the first column of the data matrix
        for row in data: self.data.append(numpy.around(numpy.insert(row, 0, bias).tolist(), accuracy))

        self.outputs = outputs
        self.layers = layers

        # Takes user value if provided otherwise defaults to
        #   - the width of the data input
        if(width is None): self.width = len(self.data[0])
        else: self.width = width

        self.learning_rate = learning_rate
        self.momentum = momentum
        self.bias = bias

        # Takes user value if provided otherwise defaults to
        #   - random values between -0.5 and 0.5
        if(weight is None): self.weights = numpy.around(numpy.random.rand(self.layers + 1, self.width) - 0.5, accuracy)
        else: self.weights = numpy.around(numpy.tile(weight, (self.layers + 1, self.width)), accuracy)

        self.accuracy = accuracy
        self.ret_type = ret_type
        self.tolerance = tolerance

    # Sigmoid activation function
    def activate(self, input): return 1 / (1 + numpy.exp(-1 * input))

    # Softmax for output layer
    def softmax(self, input):
        e_o_k = 0
        softmaxes = []
        for o_k in input: e_o_k += numpy.exp(o_k)
        for o_i in input: softmaxes.append(numpy.around(numpy.exp(o_i) / e_o_k, self.accuracy))
        return softmaxes

    def train(self):
        MST = self.tolerance * self.tolerance # Mean square tolerance
        self.activations = []
        for i in range(0, len(self.data)):
            output = self.feed_forward(self.data[i])[0]
            error = self.targets[i] - output
            error *= error

            log(Mode.INFO, "Target: " + str(self.targets[i]) + ",\n\tOutput: " + str(output) + ",\n\tMSE: " + str(error) + "\n")

            if(error > MST): self.back_propogate()
            else: break;

    # Single-data-row feed forward function
    def feed_forward(self, input):
        self.activations.append(input.tolist())
        in_layer = input
        # Recur through the hidden layers
        for l in range(0, self.layers):
            out_layer = []
            for n in range(0, self.width - 1):
                out_layer.append(numpy.around(self.activate(numpy.dot(in_layer, self.weights[l])), self.accuracy))
            in_layer = out_layer
            in_layer = numpy.insert(in_layer, 0, self.bias)
            self.activations.append(in_layer.tolist())

        # Generate the output layer
        output = []
        for o in range(0, self.outputs):
            output.append(self.activate(numpy.dot(in_layer, self.weights[self.layers])))
        if(self.ret_type == Type.softmax): output = self.softmax(output)
        self.activations.append(output)
        return output

    def back_propogate(self):
        tau = []
        for o in self.activations[-1]: tau.append(o * (1 - o) * (self.targets[0] - o))

        # Process the output layer
        delta = []
        for t in tau:
            for h in self.activations[1]:
                delta.append(self.learning_rate * t * h)
        self.weights[1] += numpy.around(delta, self.accuracy + 1)

        # Process the hidden layers
        old_tau = tau
        sum = 0
        for h in self.activations[1]: sum += h * old_tau[0]
        for h in self.activations[1]:
            tau = h * (1 - h) * sum

        for l in range(0, self.layers):
            delta = []
            sum = 0
            for h in self.activations[l]: sum += h * self.learning_rate * tau
            delta.append(sum)
            self.weights[l - 1] += delta

    def __str__(self):
        ret = "\nWeight Table (" + str(len(self.weights)) + "x" + str(len(self.weights[0])) + "):"
        for row in self.weights: ret += "\n\t" + str(row)

        ret += "\n\nData Table (" + str(len(self.data)) + "x" + str(len(self.data[0])) + "):"
        for i in range(0, self.width - 1): ret += "\nR" + str(i + 1) + ":\t" + str(self.targets[i]) +"\t?= " + str(self.data[i])
        ret += "\n"
        return ret
