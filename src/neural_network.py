import numpy, random

class NeuralNetwork:
    def __init__(self, labels, input, hidden_layers, momentum=0.75):
        self.input = numpy.shape(input)[1]
        self.labels = numpy.shape(labels)[1]

        self.data = numpy.shape(input)[0]
        self.hidden_layers = hidden_layers
        self.momentum = momentum

        self.weights = []

        self.weights_1 = (numpy.random.rand(self.input + 1, self.hidden_layers) - 0.5) * 2 / numpy.sqrt(self.input)
        self.weights_2 = (numpy.random.rand(self.hidden_layers + 1, self.labels) - 0.5) * 2 / numpy.sqrt(self.hidden_layers)

    def forward(self, input): pass

    def __str__(self):
        return "in=" + str(self.input) \
                + ", out=" + str(self.labels) \
                + ", data=" + str(self.data) \
                + ", hlc=" + str(self.hidden_layers) \
                + ", mntm=" + str(self.momentum) \
                + ", l1w=" + str(self.weights_1) \
                + ", l2w=" + str(self.weights_2)
