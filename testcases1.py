import unittest
from assignment2 import RoadGraph   #change appropriately
#The basic set of roads from the assignment 1.3 examples
roads_basic = [(0, 1, 4), (1, 2, 2), (2, 3, 3), (3, 4, 1), (1, 5, 2),
             (5, 6, 5), (6, 3, 2), (6, 4, 3), (1, 7, 4), (7, 8, 2),
             (8, 7, 2), (7, 3, 2), (8, 0, 11), (4, 3, 1), (4, 8, 10)]
cafes_basic = [(5, 10), (6, 1), (7, 5), (0, 3), (8, 4)]

#The basic set of roads from the assignment 1.1 examples
roads_basic_2 = [(0, 1, 4), (0, 3, 2), (0, 2, 3), (2, 3, 2), (3, 0, 3)]
cafes_basic_2 = [(0, 5), (3, 2), (1, 3)]

#A disconnected set of roads.
roads_disconnected = [(0, 1, 3), (1, 0, 3), (2, 3, 4), (3, 2, 4)]
cafes_disconnected = [(0, 5), (2, 5)]

#A circular graph
roads_circular = [(0, 1, 1), (1, 2, 1), (2, 3, 1), (3, 4, 1), (4, 0, 1)]
cafes_circular = [(4, 2)]

#A graph with a single unreachable cafe
roads_unreachable = [(0, 1, 5), (1, 2, 3), (1, 3, 2), (2, 1, 4), (1, 0, 7)]
cafes_unreachable = [(3, 0)]

class TestRoadmapBasic(unittest.TestCase):  # The tests given on the assessment 2 sheet

    def setUp(self) -> None:
        self.RoadGraph_1 = RoadGraph(roads_basic, cafes_basic)
        self.RoadGraph_2 = RoadGraph(roads_basic_2, cafes_basic_2)

    def test_basics_1(self):
        self.assertEqual([1, 7], self.RoadGraph_1.routing(1, 7))

    def test_basics_2(self):
        self.assertEqual([7, 8], self.RoadGraph_1.routing(7, 8))

    def test_basics_3(self):
        self.assertEqual([1, 5, 6, 3], self.RoadGraph_1.routing(1, 3))

    def test_basics_4(self):
        correct_solutions = [[1, 5, 6, 4], [1, 5, 6, 3, 4]]
        self.assertIn(self.RoadGraph_1.routing(1, 4), correct_solutions)

    def test_basics_5(self):
        self.assertEqual([3, 4, 8, 7, 3, 4], self.RoadGraph_1.routing(3, 4))

    def test_basics_6(self):
        self.assertEqual([1, 7, 8, 0, 1], self.RoadGraph_1.routing(1, 1))

    def test_basics_7(self):
        self.assertEqual([3, 0, 2], self.RoadGraph_2.routing(3, 2))

    def test_basics_8(self):
        self.assertEqual([0, 1], self.RoadGraph_2.routing(0, 1))

    def test_basics_9(self):
        self.assertEqual([2, 3, 0, 1], self.RoadGraph_2.routing(2, 1))

    def test_basics_10(self):
        self.assertEqual([0], self.RoadGraph_2.routing(0, 0))

    def test_basics_11(self):
        self.assertEqual([0], self.RoadGraph_1.routing(0, 0))


class TestRoadmapIsNone(unittest.TestCase):  # testing scenarios where None should be returned

    def setUp(self) -> None:
        self.NoCafeGraph = RoadGraph(roads_basic, [])
        self.NoRoadGraph = RoadGraph([], [])
        self.RoadGraph_2 = RoadGraph(roads_basic_2, cafes_basic_2)

    def test_no_cafes(self):
        self.assertIsNone(self.NoCafeGraph.routing(1,7))

    def test_no_roads(self):
        self.assertIsNone(self.NoRoadGraph.routing(1, 7))

    def test_no_routes(self):
        self.assertIsNone(self.RoadGraph_2.routing(1, 0))


class TestRoadmapDisconnected(unittest.TestCase): #testing the disconnected graph

    def setUp(self) -> None:
        self.disconnected_graph = RoadGraph(roads_disconnected, cafes_disconnected)

    def test_disconnected_1(self):
        self.assertIsNone(self.disconnected_graph.routing(0,2))

    def test_disconnected_2(self):
        self.assertIsNone(self.disconnected_graph.routing(3,1))

    def test_disconnected_3(self):
        self.assertEqual([0, 1], self.disconnected_graph.routing(0,1))

    def test_disconnected_4(self):
        self.assertEqual([3, 2], self.disconnected_graph.routing(3,2))


class TestRoadmapCircular(unittest.TestCase): #testing the disconnected graph

    def setUp(self) -> None:
        self.circular_graph = RoadGraph(roads_circular, cafes_circular)

    def test_circular_1(self):
        self.assertEqual([0, 1, 2, 3, 4, 0], self.circular_graph.routing(0, 0))

    def test_circular_2(self):
        self.assertEqual([0, 1, 2, 3, 4, 0, 1, 2, 3], self.circular_graph.routing(0, 3))


class TestRoadmapUnreachable(unittest.TestCase): #testing the unreachable cafe graph

    def setUp(self) -> None:
        self.unreachable_graph = RoadGraph(roads_unreachable, cafes_unreachable)

    def test_unreachable_1(self):
        self.assertIsNone(self.unreachable_graph.routing(0, 2))

    def test_unreachable_2(self):
        self.assertIsNone(self.unreachable_graph.routing(2, 0))

    def test_unreachable_3(self):
        self.assertIsNone(self.unreachable_graph.routing(3, 0))


if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(TestRoadmapBasic)
    unittest.TextTestRunner(verbosity=0).run(suite)
    suite = unittest.TestLoader().loadTestsFromTestCase(TestRoadmapCircular)
    unittest.TextTestRunner(verbosity=0).run(suite)
    suite = unittest.TestLoader().loadTestsFromTestCase(TestRoadmapDisconnected)
    unittest.TextTestRunner(verbosity=0).run(suite)
    suite = unittest.TestLoader().loadTestsFromTestCase(TestRoadmapIsNone)
    unittest.TextTestRunner(verbosity=0).run(suite)
    suite = unittest.TestLoader().loadTestsFromTestCase(TestRoadmapUnreachable)
    unittest.TextTestRunner(verbosity=0).run(suite)