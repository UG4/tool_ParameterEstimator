import unittest
import os
import sys
sys.path.insert(0, os.path.abspath('../..'))

import numpy as np
from UGParameterEstimator import FreeSurfaceTimeDependentEvaluation

class FreeSurfaceTimeDependentEvaluationTests(unittest.TestCase):

    def setUp(self):
        with open("0_measurement.csv","w") as f:
            f.write("step,time,dim0,z\n")
            f.write("1,1,0,1\n")
            f.write("1,1,2,2\n")
            f.write("2,2,0,2\n")
            f.write("2,2,2,3\n")
            f.write("3,3,0,3\n")
            f.write("3,3,2,4\n")
            f.write("FINISHED,,,")
        self.series0 = FreeSurfaceTimeDependentEvaluation.parse(".", 0)

        with open("1_measurement.csv", "w") as f:
            f.write("step,time,dim0,z\n")
            f.write("1,1,0,1\n")
            f.write("1,1,2,2\n")
            f.write("2,1.5,0,2\n")
            f.write("2,1.5,2,3\n")
            f.write("3,2.5,0,3\n")
            f.write("3,2.5,2,4\n")
            f.write("4,3.5,0,4\n")
            f.write("4,3.5,2,5\n")
            f.write("FINISHED,,,")
        self.series1 = FreeSurfaceTimeDependentEvaluation.parse(".", 1)

        os.remove("0_measurement.csv")
        os.remove("1_measurement.csv")

    def test_read_in(self):
        self.assertEqual(self.series0.locations, [0, 2])
        self.assertEqual(self.series0.times, [1, 2, 3])
        self.assertEqual(self.series0.data, [[1, 2], [2, 3], [3, 4]])
        self.assertEqual(self.series1.locations, [0, 2])
        self.assertEqual(self.series1.times, [1, 1.5, 2.5, 3.5])
        self.assertEqual(self.series1.data, [[1, 2], [2, 3], [3, 4], [4, 5]])

    def test_numpy_array(self):
        self.assertTrue(np.allclose(
            self.series0.getNumpyArray(),
            np.array([1, 2, 2, 3, 3, 4])))

        self.assertTrue(np.allclose(
            self.series1.getNumpyArray(),
            np.array([1, 2, 2, 3, 3, 4, 4, 5])))

    def test_numpy_array_like(self):
        self.assertTrue(np.allclose(
            self.series1.getNumpyArrayLike(self.series0),
            np.array([1, 2, 2.5, 3.5, 3.5, 4.5])))

        self.assertTrue(np.allclose(
            self.series0.getNumpyArrayLike(self.series1),
            np.array([1, 2, 1.5, 2.5, 2.5, 3.5, 3, 4])))

    def test_numpy_array_like_same_format(self):
        self.assertTrue(np.allclose(
            self.series0.getNumpyArrayLike(self.series0),
            np.array([1, 2, 2, 3, 3, 4])))

if __name__ == '__main__':
    unittest.main()