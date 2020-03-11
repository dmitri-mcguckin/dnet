#!/usr/bin/env python3

import unittest, sys, numpy
sys.path.insert(1, "../src")
from neural_network import NeuralNetwork

class UnitTests(unittest.TestCase):
    def setUp(self):
        self.net = NeuralNetwork([], [], 0, )

    def test_softmax_is_correct(self):
        expected = [0.501, 0.499]
        input = [0.461, 0.455]
        output = self.net.softmax(input)
        self.assertEqual(output, expected)

    def test_activate_is_correct(self):
        expecteds = [0.12, 0.18, 0.27, 0.38, 0.50, 0.62, 0.73, 0.82, 0.88]
        inputs = [-2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2]

        for i in range(0, len(expecteds)):
            output = numpy.around(self.net.activate(inputs[i]), 2)
            self.assertEqual(output, expecteds[i])

if __name__ == "__main__": unittest.main()
