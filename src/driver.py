#!/usr/bin/env python3
import csv
from neural_network import NeuralNetwork
from ftfutils import log, Mode

def main():
    # Runtime stuff
    labels = []
    training_data = []

    # Read the data file
    file = open("data/mnist_demo.csv")
    raw_csv = csv.reader(file, delimiter=',')
    for i, row in enumerate(raw_csv):
        labels.append(row[:1])
        training_data.append(row[1:])
    file.close()

    dnet = NeuralNetwork(labels, training_data, 3, 0.9)

    log(Mode.DEBUG, "Loaded NN: " + str(dnet))

if __name__ == "__main__": main()
