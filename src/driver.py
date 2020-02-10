#!/usr/bin/env python3
import csv, numpy, os.path, sys
from neural_network import NeuralNetwork, Type
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
    try: targets, training_data = retrieve_from_csv("./mnist_data/mnist_train.csv")
    except FileNotFoundError as e:
        log(Mode.ERROR, str(e))
        sys.exit(-1)

    # Preprocess the data
    targets = preprocess(targets)
    _pd = []
    for row in training_data: _pd.append(preprocess(row))
    training_data = _pd

    # Create the neural network
    training_model = NeuralNetwork(targets,
                                   training_data,
                                   outputs=10,
                                   layers=1,
                                   learning_rate=0.1,
                                   momentum=0.9,
                                   accuracy=20,
                                   ret_type=Type.softmax)
    log(Mode.INFO, "Created neural network: " + str(training_model))

    # Run the training model
    log(Mode.WARN, "Training started!")
    training_model.train()
    log(Mode.WARN, "Training ended!")

if __name__ == "__main__": main()
