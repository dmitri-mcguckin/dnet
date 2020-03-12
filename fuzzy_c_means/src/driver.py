import sys, matplotlib.pyplot as plt, numpy as np, random as r, time
from matplotlib import colors

COLOR_MAP = []

def color_average(c1, c2):
    c1 = [int(c1[1:3], 16), int(c1[3:5], 16), int(c1[5:7], 16)]
    c2 = [int(c2[1:3], 16), int(c2[3:5], 16), int(c2[5:7], 16)]
    rgb = []
    for i in range(3):
        rgb.append(int(np.sqrt((np.square(c1[i]) + np.square(c2[i])) / 2)))
    return '#' + str(hex(rgb[0]))[2:].rjust(2, '0') \
               + str(hex(rgb[1]))[2:].rjust(2, '0') \
               + str(hex(rgb[2]))[2:].rjust(2, '0')

class Point:
    def __init__(self, x, y, k):
        self.x = x
        self.y = y
        self.k_grades = []
        sum = 1
        for i in range(k):
            g = round(r.uniform(0, sum), 2)
            sum -= g
            self.k_grades.append(g)

    # Euclidean distance between two points
    def distance(self, p):
        return np.sqrt(np.square(p.x - self.x) + np.square(p.y - self.y))

    def k_class(self):
        return self.k_grades.index(max(self.k_grades))

    def __str__(self):
        return "(" + str(self.x) + ", " + str(self.y) + "): " + str(self.k_grades)

# Plot the data
def plot(ax, k, data, size=2, marker=None, color_override=None):
    for i in range(k):
        k_data = list(filter(lambda x: x.k_class() == i, data))
        arr = np.array([[p.x, p.y] for p in k_data])
        if(len(arr) > 0):
            x, y = arr.T
            if(color_override is not None): color = color_override
            else: color = COLOR_MAP[i]
            ax.scatter(x, y,c=color, s=size, marker=marker)

def init_colors(k):
    res = []
    if(k <= 10): table = colors.TABLEAU_COLORS
    else: table = colors.XKCD_COLORS
    for c in table.values(): res.append(c)
    return res

def init_centroids(k):
    res = []
    for i in range(0, k):
        res.append(Point(r.uniform(-4, 4), r.uniform(-4, 4), i))
    return res

def init_data(filename, k):
    res = []
    print("Loading file: " + str(filename))

    file = open(filename)
    for line in file:
        pieces = line.strip().split(" ")
        res.append(Point(float(pieces[0]), float(pieces[-1]), k))
    file.close()
    return res

def classify_data(m, centroids, data):
    did_change = False
    exp = float(2 / (m - 1))

    for d in data:
        # Get all the centroid distances
        for a, i in enumerate(centroids):
            k_dists = []
            for j in centroids:
                k_dists.append(np.pow(d.distance(i) / d.distance(j), exp))
            d.k_grades[a]


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
            c.x = x
            c.y = y

def main(args):
    global COLOR_MAP
    if(len(args) < 2):
        print("usage: run <filename> <k-clusters> <optional: time-per-epoch (seconds)> <optional: --no-hang>")
        sys.exit(-1)

    # Plot initialization
    fig, ax = plt.subplots(1, 1, constrained_layout=True)
    fig.suptitle("K = " + str(args[1]), fontsize=12)
    ax.set_facecolor('black')

    # Variables
    prgm_id = int(time.time())
    did_change = True
    itter_count = 0
    k = int(args[1])
    data = init_data(args[0], k)
    COLOR_MAP = init_colors(k)
    centroids = init_centroids(k)
    m = 2
    if(len(args) >= 3): playback_time = float(args[2])
    else: playback_time =  1.5
    if(len(args) >= 4 and args[3] == "--no-hang"): no_hang = True
    else: no_hang = False

    while(did_change):
        ax.set_title("Iteration #" + str(itter_count + 1))
        did_change = classify_data(m, centroids, data) # Run the classification

        plot(ax, k, data)
        plot(ax, k, centroids, marker="+", size=80, color_override="white")

        plt.pause(playback_time)

        update_centroids(centroids, data) # Update the centroids
        if(itter_count == 0): plt.gcf().savefig('imgs/' + str(prgm_id) + '_begin.png', dpi=100)
        elif(not did_change): plt.gcf().savefig('imgs/' + str(prgm_id) + '_end.png', dpi=100)
        itter_count += 1
        ax.cla()

    print("Completed k-means in", itter_count, "iterations!")
    if(not no_hang): # Show the window until the user exits
        print("Hanging untill user exit...")
        plt.show()

if __name__ == "__main__": main(sys.argv[1:])
