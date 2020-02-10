#!/usr/bin/env python3
import csv, numpy, os.path, sys
from neural_network import NeuralNetwork
from ftfutils import log, Mode

# Read in the CSV file, then separate the data between targets and data
#   - It is assumed that the first collumn of each row is the target in the
#       training data
def retrieve_from_csv(filename):
    # Check if file exists
    if(not os.path.isfile(filename)): raise FileNotFoundError("File does not exist: " + filename)

    labels = []
    data = []

    # Read from the file and turn it into a CSV object
    file = open(filename)
    raw_csv_data = csv.reader(file, delimiter=',')

    # Separate the data accordingly then close the buffer
    for row in raw_csv_data:
        labels.append(float(row[:1][0]))

        s_row = []
        for i in row[1:]:
            s_row.append(float(i))
        data.append(s_row)

    file.close()

    return labels, data

def preprocess(input):
    t = []
    for i in input: t.append(float(i / 255))
    return t

def main():
    try: labels, data = retrieve_from_csv("./mnist_data/mnist_demo.csv")
    except FileNotFoundError as e:
        log(Mode.ERROR, str(e))
        sys.exit(-1)

    # Preprocess the data
    labels = preprocess(labels)
    _pd = []
    for row in data: _pd.append(preprocess(row))
    data = _pd

    # Validate the data
    if(len(labels) != len(data)):
        log(Mode.Error, "Mismatched label-data sizes: (labels " + str(len(labels)) + ", data: " + str(len(data)) + ")")
        sys.exit(-1)

    acc_error = 0.001
    layer_size = 10
    log(Mode.INFO, "label-data: (" + str(len(labels)) + ", " + str(len(data)) + ") | nodes-per-row: " + str(len(data[0])))
    training_model = NeuralNetwork(labels, data, layer_size, learning_rate=0.2, momentum=0.9, weight=0.1, rsoftmax=True)

    for i, row in enumerate(data):
        log(Mode.INFO, "Training on data: " + str(row))
        target = labels[i]

        output = training_model.feed_forward(row)[0]
        error = target - output
        log(Mode.INFO, "(o: " + str(output) + ", t: " + str(target) + ", d: " + str(error) + ")")
        training_model.back_propogate(layer_size, target)
        # else: break;

    log(Mode.WARN, "Training ended!")

if __name__ == "__main__": main()
