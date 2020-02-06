#!/usr/bin/env python3
import csv
from ftfutils import log, Mode

def main():
    training_file = open("data/mnist_demo.csv")
    raw_csv = csv.reader(training_file, delimiter=',')
    labels = []
    training_data = []

    for i, row in enumerate(raw_csv):
        labels.append(row[:1])
        training_data.append(row[1:])

    log(Mode.DEBUG, str(labels[0]) + "<" + str(len(labels[0])) + ">" + ": " + str(training_data[0]) + "<" + str(len(training_data[0])) + ">")

if __name__ == "__main__": main()
