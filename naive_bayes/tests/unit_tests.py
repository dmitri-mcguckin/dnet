import unittest, sys
sys.path.insert(1, "../src")
from driver import prob_density

class UnitTests(unittest.TestCase):
    def test_probability_density(self):
        input = [[5.2, 4.8, 1.8],
            [6.3, 7.1, 2.0],
            [5.2, 4.7, 2.5],
            [6.3, 4.2, 3.7]]
        expecteds = [.22, .18, .16, .09]

        for c, i in enumerate(input): assert(round(prob_density(i[0], i[1], i[2]), 2) == expecteds[c])

if __name__ == "__main__": unittest.main()
