import sys, csv, numpy

import matplotlib.pyplot as plt
import scipy.stats as st
import math

def error(num, den):
    p = num / den
    if(p >= 1): p -= 1
    else: p = 1 - p
    return p

def get_features(data, index):
    res = []
    for d in data: res.append(d[index])
    return res

def load_file(filename):
    labels = []
    data = []
    file = open(filename)
    raw_data = csv.reader(file, delimiter=',')
    for raw_row in raw_data:
        row = []
        for d in raw_row: row.append(d)
        labels.append(int(row[-1]))
        r = []
        for d in row[:-1]: r.append(float(d))
        data.append(r)
    file.close()
    return labels, data

def prob_density()

def plot_model(title, mean, std):
    sigma = math.sqrt(std)
    x = numpy.linspace(mean - 3 * sigma, mean + 3 * sigma, 100)
    # plt.set_title(title)
    plt.xlabel("probabilities")
    plt.plot(x, st.norm.pdf(x, mean, sigma))
    plt.show()

def main():
    # Check for debug mode
    if(len(sys.argv) == 2 and sys.argv[1] == '-d'): DEBUG = True
    else: DEBUG = False

    # Loading the data file
    labels, data = load_file("data/training-set.data")
    print("Loaded", str(len(data)), "examples from the training set! With", str(len(data[0])), "features per example and ", str(len(labels)), " labels!")
    label_mods = {}
    is_spam_mods = []
    not_spam_mods = []

    is_spam_probs = {}
    not_spam_probs = {}

    # Statistics funny buisness
    import matplotlib.pyplot as plt
    import scipy.stats as st
    import math

    means = []
    stds = []

    for i in range(0, len(data[0])):
        means.append(numpy.mean(get_features(data, i)))
        stds.append(numpy.std(get_features(data, i)))
        plot_model("Feture #" + str(i), means[i], stds[i])

    sys.exit(0)

    # Determine the label set
    for l in labels:
        if(label_mods.get(l) is None): label_mods[l] = 1
        else:label_mods[l] += 1

    # Determine the feature size and set
    for i in range(0, len(data[0])):
        is_spam_mods.append({})
        not_spam_mods.append({})
        for j in range(0, len(data)):
            point = data[j][i]
            label = labels[j]

            if(label == 1):
                if(is_spam_mods[i].get(point) is None): is_spam_mods[i][point] = 1
                else: is_spam_mods[i][point] += 1
            else:
                if(not_spam_mods[i].get(point) is None): not_spam_mods[i][point] = 1
                else: not_spam_mods[i][point] += 1

    if(DEBUG):
        print("\nMODELS:\n\tLABELS:", str(label_mods))
        print("\n\nYES:", str(is_spam_mods))
        print("\n\nNO:", str(not_spam_mods), "\n")

    epsilon = 0.0000000000000001

    # Create is spam probabilities
    for feature in is_spam_mods:
        for name, count in feature.items(): is_spam_probs[name] = ((count + 1) / (label_mods[1]) + len(feature))

    # Create not spam probabilities
    for feature in not_spam_mods:
        for name, count in feature.items(): not_spam_probs[name] = ((count + 1) / (label_mods[0]) + len(feature))

    if(DEBUG):
        print("\nPROBABILITIES:")
        print("\n\nIS SPAM PROBS:")
        for n, p in is_spam_probs.items(): print("\tP(", n, ") =", str(p))
        print("\n\nNOT SPAM PROBS:")
        for n, p in not_spam_probs.items(): print("\tP(", n, ") =", str(p))

    # Load test data
    test_labels, test_data = load_file("data/test-set.data")
    print("Loaded", str(len(test_data)), "examples from the test set! With", str(len(test_data[0])), "features per example and ", str(len(test_labels)), " labels!")

    # Heuristic things
    is_spams = 0
    not_spams = 0
    expected_spam = 0
    expected_not_spam = 0

    # Calculate new predictions on the test set
    for i, datum in enumerate(test_data):
        yes_prob = 1
        for d in datum:
            p = is_spam_probs.get(d)
            if(p is None): p = epsilon
            yes_prob *= p

        no_prob = 1
        for d in datum:
            p = not_spam_probs.get(d)
            if(p is None): p = epsilon
            no_prob *= p

        yes_prob = numpy.log(yes_prob)
        no_prob = numpy.log(no_prob)

        if(yes_prob > no_prob): is_spams += 1
        else: not_spams += 1

        if(test_labels[i] == 1): expected_spam += 1
        else: expected_not_spam += 1

    accuracy = int(100 * (1 - (error(is_spams, expected_spam))))

    print("\nTest Report:")
    print("\tAccuracy:", str(accuracy) + "%")
    print("\tIdentified:\t(spam: " + str(is_spams) + ",\tnon-spam: " + str(not_spams) + ") | checksum: " + str(is_spams + not_spams))
    print("\tExpected:\t(spam: " + str(expected_spam) + ",\tnon-spam: " + str(expected_not_spam) + ") | checksums: " + str(expected_spam + expected_not_spam))

if __name__ == "__main__": main()
