import sys, csv

def load_file(filename):
    labels = []
    data = []
    file = open(filename)
    raw_data = csv.reader(file, delimiter=',')
    for raw_row in raw_data:
        row = []
        for d in raw_row: row.append(float(d))
        labels.append(int(row[-1]))
        data.append(row[:-1])
    file.close()
    return labels, data

def main():
    if(len(sys.argv) != 2):
        print("usage: run <file name>")
        sys.exit(-1)
    filename = sys.argv[1]
    print("Loading file:", filename)
    labels, data = load_file(filename)
    print("Loaded", str(len(data)), "examples from the training set!\n\tWith", str(len(data[0])), "features per example!")
    print("Loaded", str(len(labels)), "labels!")

    print(labels)

if __name__ == "__main__": main()
