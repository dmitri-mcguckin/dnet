#!/usr/bin/env python3
import csv, numpy
from neural_network import NeuralNetwork
from ftfutils import log, Mode

def preprocess(input):
    t = []
    print(input)
    for i in input:
        print(i)
        t.append(float(i[0] / 255))
    return t

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

    # dnet = NeuralNetwork(preprocess(labels), preprocess(training_data), 1, len(training_data), 0.9)

    log(Mode.INFO, str(preprocess([255, 120, 69])))
    # log(Mode.DEBUG, "Loaded NN: " + str(dnet))

if __name__ == "__main__": main()
