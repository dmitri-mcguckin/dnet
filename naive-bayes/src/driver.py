import csv

def load_file(filename):
    labels = []
    data = []
    file = open(filename)
    raw_data = csv.reader(file, delimiter=',')
    for raw_row in raw_data:
        row = []
        for d in raw_row: row.append(float(d))
        labels.append(row[0])
        data.append(row[1:])
    file.close()
    return labels, data

def main():
    labels, data = load_file("data/training-set.data")
    print("Loaded", str(len(data)), "examples from the training set!\n\tWith", str(len(data[0])), "features per example!")
    print("Loaded", str(len(labels)), "labels!")

if __name__ == "__main__": main()
