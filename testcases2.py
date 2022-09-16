import unittest
import assignment2

class TestMethods(unittest.TestCase):
    # Example
    # The scores you can obtain in each downhill segment
    downhillScores = [(0, 6, -500), (1, 4, 100), (1, 2, 300),
                      (6, 3, -100), (6, 1, 200), (3, 4, 400), (3, 1, 400),
                      (5, 6, 700), (5, 1, 1000), (4, 2, 100)]

    def test_1(self):
        # The starting and finishing points
        start = 6
        finish = 2

        self.assertEqual(assignment2.optimalRoute(self.downhillScores, start, finish), [6, 3, 1, 2])

    def test_2(self):
        # The starting and finishing points
        start = 0
        finish = 2

        self.assertEqual(assignment2.optimalRoute(self.downhillScores, start, finish), [0, 6, 3, 1, 2])

    def test_3(self):
        downhillScores = [(0, 1, 100), (1, 2, 200), (2, 3, -400), (3, 4, -500), (4, 5, -700), (5, 6, 400)]

        # The starting and finishing points
        start = 0
        finish = 6

        self.assertEqual(assignment2.optimalRoute(downhillScores, start, finish), [0, 1, 2, 3, 4, 5, 6])

    def test_4(self):
        downhillScores = [(0, 1, 100), (1, 2, 200), (2, 3, -400), (3, 4, -500), (4, 5, -700), (5, 6, 400)]

        # The starting and finishing points
        start = 6
        finish = 6

        self.assertEqual(assignment2.optimalRoute(downhillScores, start, finish), [6])

    def test_5(self):
        downhillScores = [(0, 1, 100), (1, 2, 200), (2, 3, -400), (3, 4, -500), (4, 5, -700), (5, 6, 400)]

        # The starting and finishing points
        start = 6
        finish = 0

        self.assertEqual(assignment2.optimalRoute(downhillScores, start, finish), None)



if __name__ == '__main__':
    unittest.main()