import sys, csv, numpy

EPSILON = 0.0001 # Convert zero-value variances to this
TRAINING_FILE = "data/training-set.csv"
TEST_FILE = "data/test-set.csv"

# Calculate the percentage of error, order agnostic
def error(num, den):
    p = num / den
    if(p >= 1): p -= 1
    else: p = 1 - p
    return p

# Generate the parameters for normal models
#   for both the affirmative and negative class values
def get_models(data, labels):
    aff_mu = []
    aff_sigma = []
    neg_mu = []
    neg_sigma = []

    for f in range(0, len(data[0])):
        aff_f = []
        neg_f = []
        for i in range(0, len(data)):
            if(labels[i] == 0): aff_f.append(data[i][f])
            else: neg_f.append(data[i][f])
        aff_mu.append(round(numpy.mean(aff_f), 2))
        aff_sigma.append(round(numpy.std(aff_f), 2))
        neg_mu.append(round(numpy.mean(neg_f), 2))
        neg_sigma.append(round(numpy.std(neg_f), 2))

    return aff_mu, aff_sigma, neg_mu, neg_sigma

# Load the data file
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

# Dear god, never again
# Calculate the probability density
def prob_density(x, mean, deviation):
    if(deviation == 0): deviation = EPSILON
    return numpy.exp((numpy.power(x - mean, 2) / (-2 * numpy.power(deviation, 2)))) / (numpy.sqrt(2 * numpy.pi) * deviation)

def main():
    # Check for debug mode
    DEBUG = bool(len(sys.argv) == 2 and sys.argv[1] == '-d')

    # Loading the data file
    labels, data = load_file(TRAINING_FILE)
    print("Loaded", str(len(data)), "examples from file: " + TRAINING_FILE + ", With", str(len(data[0])), "features per example and ", str(len(labels)), " labels!")
    label_mods = {}

    # Determine the label set
    for l in labels:
        if(label_mods.get(l) is None): label_mods[l] = 1
        else:label_mods[l] += 1

    # Statistics funny business
    s_mu, s_sigma, ns_mu, ns_sigma = get_models(data, labels)

    if(DEBUG):
        print("\n\nspam means: " + str(s_mu) + "\n\nspam deviations: " + str(s_sigma))
        print("\n\nnon-spam means: " + str(ns_mu) + "\n\nnon-spam deviations: " + str(ns_sigma))

    # Load test data
    test_labels, test_data = load_file("data/test-set.csv")
    print("Loaded", str(len(test_data)), "examples from file: " + TEST_FILE + ", With", str(len(test_data[0])), "features per example and ", str(len(test_labels)), " labels!")

    # Heuristic things
    is_spam = 0
    not_spam = 0
    expected_spam = 0
    expected_not_spam = 0

    # Calculate new predictions on the test set
    for i, datum in enumerate(test_data):
        yes_prob = 0
        no_prob = 0
        for j, d in enumerate(datum):
            yp = prob_density(d, s_mu[j], s_sigma[j])
            np = prob_density(d, ns_mu[j], ns_sigma[j])

            if(yp != 0): yes_prob += numpy.log(yp)
            if(np != 0): no_prob += numpy.log(np)

        if(yes_prob > no_prob): is_spam += 1
        else: not_spam += 1

        if(test_labels[i] == 1): expected_spam += 1
        else: expected_not_spam += 1

    accuracy = int(100 * (1 - (error(is_spam, expected_spam))))

    print("\nTest Report:")
    print("\tAccuracy:", str(accuracy) + "%")
    print("\tIdentified:\t(spam: " + str(is_spam) + ",\tnon-spam: " + str(not_spam) + ") | checksum: " + str(is_spam + not_spam))
    print("\tExpected:\t(spam: " + str(expected_spam) + ",\tnon-spam: " + str(expected_not_spam) + ") | checksum: " + str(expected_spam + expected_not_spam))