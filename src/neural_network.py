import numpy, random

class NeuralNetwork:
    def __init__(self, labels, input, hidden_layers, beta=1, momentum=0.75):
        self.labels = labels
        self.input = input

        # self.input = numpy.shape(input)[1]
        # self.labels = numpy.shape(labels)[1]

        self.hidden_layers = hidden_layers
        self.beta = len(input)
        self.momentum = momentum

        self.weights = numpy.random.rand(len(input), beta)

        # self.weights_1 = (numpy.random.rand(self.input + 1, self.hidden_layers) - 0.5) * 2 / numpy.sqrt(self.input)
        # self.weights_2 = (numpy.random.rand(self.hidden_layers + 1, self.labels) - 0.5) * 2 / numpy.sqrt(self.hidden_layers)

    def forward(self, input): pass

    def __str__(self):
        return "in=" + str(self.input) \
                + ", out=" + str(self.labels) \
                + ", hlc=" + str(self.hidden_layers) \
                + ", mntm=" + str(self.momentum) \
                + ", weights=" + str(self.weights)
