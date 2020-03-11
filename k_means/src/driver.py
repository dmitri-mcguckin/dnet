import sys, matplotlib.pyplot as plt, numpy as np

def plot(data):
    data = np.array(data)
    x, y = data.T
    plt.scatter(x,y)

def load_file(filename):
    res = []
    print("Loading file: " + str(filename))

    file = open(filename)
    for line in file:
        pieces = line.strip().split(" ")
        coords = (float(pieces[0]), float(pieces[-1]))
        res.append(coords)
    file.close()

    return res

def main(args):
    if(len(args) == 0):
        print("usage: run <filename>")
        sys.exit(-1)

    data = load_file(args[0])
    plot(data)
    plt.show()

if __name__ == "__main__": main(sys.argv[1:])
