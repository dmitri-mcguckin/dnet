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
    def __init__(self, x, y, k, cen_i=None):
        self.x = x
        self.y = y
        self.k = k
        self.i = cen_i
        self.r = -1
        self.k_grades = []

        # If the point is a centroid, initialize the grades as a mask
        #   Else, initialize all of the grades randomly
        if(self.i is not None):
            for _ in range(k): self.k_grades.append(0)
            self.k_grades[self.i] = 1.0
        else:
            sum = 1
            for _ in range(k):
                if(k == 1): g = 1.0
                else: g = round(r.uniform(0, sum), 2)
                sum -= g
                self.k_grades.append(g)

    def distance(self, p): return np.sqrt(np.square(p.x - self.x) + np.square(p.y - self.y)) # Euclidean distance between two points
    def k_class(self): return self.k_grades.index(max(self.k_grades)) # Get the dominant class
    def __add__(self, p): return Point(self.x + p.x, self.y + p.y, self.k, self.i) # Add two points togeather
    def __mul__(self, a): return Point(self.x * a, self.y * a, self.k, self.i) # Multiply a point by a scalar
    def __truediv__(self, a): return Point(self.x / a, self.y / a, self.k, self.i) # Divide a point by a scalar
    def __str__(self): return "(" + str(self.x) + ", " + str(self.y) + ")" # Point to string

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
        if(data[i].i is not None):
            c = plt.Circle((data[i].x, data[i].y), data[i].r, color=COLOR_MAP[i], alpha=0.1)
            ax.add_artist(c)

def init_colors(k):
    res = []
    if(k <= 10): table = colors.TABLEAU_COLORS
    else: table = colors.XKCD_COLORS
    for c in table.values(): res.append(c)
    return res

def init_centroids(k):
    res = []
    for i in range(0, k):
        res.append(Point(r.uniform(-4, 4), r.uniform(-4, 4), k, i))
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
    exp = (2 / (m - 1))
    for d in data:
        distances = []
        old_grades = d.k_grades.copy()
        for c in centroids: distances.append(d.distance(c))
        for i, c in enumerate(centroids):
            if(sum(distances) == 0 or distances[i] == 0): continue
            d.k_grades[i] = (1 / (distances[i] / sum(distances))**exp)
        if(d.k_grades != old_grades): did_change = True
        c = centroids[d.k_class()]
        if(d.distance(c) > c.r): c.r = d.distance(c)
    return did_change

def update_centroids(centroids, data):
    res = []
    for i, c in enumerate(centroids):
        numer = Point(0, 0, len(centroids), i)
        denom = 0

        for d in data:
            w = d.k_grades[i]
            numer = numer + (d * w)
            denom += w
        res.append(numer / denom)
    return res

def main(args):
    global COLOR_MAP
    if(len(args) < 3):
        print("usage: run <filename> <k-clusters> <fuzzifier> <optional: time-per-epoch (seconds)> <optional: --no-hang> <max iterations>")
        sys.exit(-1)

    # Variables
    prgm_id = int(time.time())
    did_change = True
    itter_count = 0
    k = int(args[1])
    data = init_data(args[0], k)
    COLOR_MAP = init_colors(k)
    centroids = init_centroids(k)
    m = int(args[2])
    if(len(args) >= 4): playback_time = float(args[3])
    else: playback_time =  1.5
    if(len(args) >= 5 and args[4] == "--no-hang"): no_hang = True
    else: no_hang = False
    if(len(args) >= 6): max_iters = int(args[5])
    else: max_iters = None

    # Plot initialization
    fig, ax = plt.subplots(1, 1, constrained_layout=True)
    fig.suptitle("K = " + str(args[1]) + ", M = " + str(args[2]), fontsize=12)
    ax.set_facecolor('black')

    while((max_iters is None or itter_count < max_iters) and did_change):
        ax.set_title("Iteration #" + str(itter_count + 1))
        did_change = classify_data(m, centroids, data) # Run the classification

        plot(ax, k, data)
        plot(ax, k, centroids, marker="+", size=80, color_override="white")

        plt.pause(playback_time)

        centroids = update_centroids(centroids, data) # Update the centroids
        if(itter_count == 0): plt.gcf().savefig('imgs/' + str(prgm_id) + '_begin.png', dpi=100)
        elif(not did_change): plt.gcf().savefig('imgs/' + str(prgm_id) + '_end.png', dpi=100)
        itter_count += 1
        ax.cla()

    print("Completed fuzzy-c-means in", itter_count, "iterations!")
    if(not no_hang): # Show the window until the user exits
        print("Hanging untill user exit...")
        plt.show()

if __name__ == "__main__": main(sys.argv[1:])
