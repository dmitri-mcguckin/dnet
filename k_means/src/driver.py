import sys, matplotlib.pyplot as plt, numpy as np, random as r, time
from matplotlib import colors

COLOR_MAP = []
for c in colors.TABLEAU_COLORS.values(): COLOR_MAP.append(c)

class Point:
    def __init__(self, x, y, k_class=-1):
        self.x = x
        self.y = y
        self.k_class = k_class
        self.radius = -1

    # Euclidean distance between two points
    def distance(self, p):
        return np.sqrt(np.square(p.x - self.x) + np.square(p.y - self.y))

    def __str__(self):
        return "(" + str(self.x) + ", " + str(self.y) + ")"

# Plot the data
def plot(ax, k, data, size=2, marker=None):
    for i in range(k):
        k_data = list(filter(lambda x: x.k_class == i, data))
        arr = np.array([[p.x, p.y] for p in k_data])
        if(len(arr) > 0):
            x, y = arr.T
            ax.scatter(x, y,c=COLOR_MAP[i], s=size, marker=marker)

def init_centroids(k):
    res = []
    for i in range(0, k):
        res.append(Point(r.uniform(-4, 4), r.uniform(-4, 4), i))
    return res

def init_data(filename):
    res = []
    print("Loading file: " + str(filename))

    file = open(filename)
    for line in file:
        pieces = line.strip().split(" ")
        res.append(Point(float(pieces[0]), float(pieces[-1])))
    file.close()
    return res

def classify_data(centroids, data):
    did_change = False

    for d in data:
        # Start off assuming the first centroid is the closest
        k_class = centroids[0]
        k_dist = d.distance(k_class)

        # Check the rest
        for c in centroids:
            c_dist = d.distance(c)
            if(c_dist < k_dist):
                k_dist = c_dist
                k_class = c

        # Assign the new classification
        new_class = centroids.index(k_class)
        if(new_class != d.k_class): did_change = True
        d.k_class = new_class

        # Update the centroid's radius of influence

        if(k_dist > k_class.radius): k_class.radius = k_dist
    return did_change

def update_centroids(centroids, data):
    for i, c in enumerate(centroids):
        x = 0
        y = 0
        c_set = list(filter(lambda x: x.k_class == c.k_class, data))

        if(len(c_set) > 0):
            for p in c_set:
                x += p.x
                y += p.y
            x /= len(c_set)
            y /= len(c_set)
            # print("C" + str(i) + ": (" + str(c.x) + ", " + str(c.y) + ")\t->\t(" + str(x) + ", " + str(y) + ")")
            c.x = x
            c.y = y

def main(args):
    if(len(args) < 2):
        print("usage: run <filename> <k-clusters> <optional: time-per-epoch (seconds)>")
        sys.exit(-1)

    # Plot initialization
    fig, ax = plt.subplots(1, 1, constrained_layout=True)
    fig.suptitle(args[0], fontsize=12)

    # Variables
    did_change = True
    epoch_count = 0
    data = init_data(args[0])
    k = int(args[1])
    centroids = init_centroids(k)
    if(len(args) >= 3): playback_time = float(args[2])
    else: playback_time =  1.5

    while(did_change):
        ax.set_title("Epoch #" + str(epoch_count + 1))
        did_change = classify_data(centroids, data) # Run the classification

        plot(ax, k, data)
        plot(ax, k, centroids, marker="+", size=80)

        plt.pause(playback_time)

        update_centroids(centroids, data) # Update the centroids
        if(epoch_count == 0): plt.gcf().savefig('imgs/start.png', dpi=100)
        elif(not did_change): plt.gcf().savefig('imgs/end.png', dpi=100)
        epoch_count += 1
        ax.cla()

    print("Completed k-means in", epoch_count, "epochs!\n\tHanging untill user exit...")
    ax.set_title("[Done]: Epoch #" + str(epoch_count + 1))
    plt.show() # Show the window until the user exits

if __name__ == "__main__": main(sys.argv[1:])
