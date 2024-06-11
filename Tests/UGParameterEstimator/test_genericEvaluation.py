import unittest
import os
import sys
from UGParameterEstimator import ErroredEvaluation, GenericEvaluation
import numpy as np

sys.path.insert(0, os.path.abspath('../..'))

class GenericEvaluationTests(unittest.TestCase):
    """
    A test class for validating the functionality of the GenericEvaluation class.

    This class performs various unit tests to ensure that the methods and functions 
    of the GenericEvaluation class are working correctly. This includes setting up test data, 
    performing the tests, and verifying the results.

    Attributes:
        None

    Methods:
        setUp: A special method that is called before each test case. It is used to 
               set up test data or define the initial state for tests.
    """
    def setUp(self):
        with open("0_measurement.json", "w") as f:
            f.write("""
            {
                "metadata": {
                    "finished": true
                },
                "data": [
                    {
                        "time": 1,
                        "value": 1
                    },
                    {
                        "time": 2,
                        "value": 2
                    },
                    {
                        "time": 3,
                        "value": 2
                    },
                    {
                        "time": 4,
                        "value": 1
                    },
                    {
                        "time": 5,
                        "value": 0
                    }
                ]
            }            
            """)

        with open("1_measurement.json", "w") as f:
            f.write("""
            {
                "metadata": {
                    "finished": true
                },
                "data": [
                    {
                        "time": 1.5,
                        "value": 1.5
                    },
                    {
                        "time": 2.5,
                        "value": 2.5
                    },
                    {
                        "time": 3.5,
                        "value": 1.5
                    },
                    {
                        "time": 4,
                        "value": 1
                    }
                ]
            }            
            """)

        with open("4_measurement.json", "w") as f:
            f.write("""
            {
                "data": [
                    {
                        "time": 1.5,
                        "value": 1.5
                    },
                    {
                        "time": 2.5,
                        "value": 2.5
                    },
                    {
                        "time": 3.5,
                        "value": 1.5
                    },
                    {
                        "time": 4,
                        "value": 1
                    }
                ]
            }            
            """)

        
        with open("2_measurement.csv", "w") as f:
            f.write(
            "step,time,value\n"
            "1,0.5,3\n"
            "2,1,2.4\n"
            "3,2,4.623\n"
            "FINISHED,,"           
            )

        with open("3_measurement.csv", "w") as f:
            f.write(
            "step,time,value\n"
            "1,0.5,3\n"
            "2,1,2.4\n"
            "3,2,4.623\n"           
            )

        self.series0 = GenericEvaluation.parse(".", 0)
        self.series1 = GenericEvaluation.parse(".", 1)

        if isinstance(self.series0, ErroredEvaluation):
            print(self.series0.reason)

        if isinstance(self.series1, ErroredEvaluation):
            print(self.series1.reason)

    def tearDown(self):
        os.remove("0_measurement.json")
        os.remove("1_measurement.json")
        os.remove("2_measurement.csv")
        os.remove("3_measurement.csv")
        os.remove("4_measurement.json")

    def test_read_in(self):
        self.assertFalse(isinstance(self.series0, ErroredEvaluation))
        self.assertFalse(isinstance(self.series1, ErroredEvaluation))

        self.assertEqual(self.series0.times, [1, 2, 3, 4, 5])
        self.assertEqual(self.series1.times, [1.5, 2.5, 3.5, 4])
        self.assertEqual(self.series0.data, [1, 2, 2, 1, 0])
        self.assertEqual(self.series1.data, [1.5, 2.5, 1.5, 1])

    def test_parse_csv(self):
        self.series2 = GenericEvaluation.parse(".", 2)
        if isinstance(self.series2, ErroredEvaluation):
            print(self.series2.reason)
        self.assertFalse(isinstance(self.series2, ErroredEvaluation))
        self.assertEqual(self.series2.times, [0.5, 1, 2])
        self.assertEqual(self.series2.data, [3, 2.4, 4.623])

    def test_error_on_not_finished(self):
        self.series3 = GenericEvaluation.parse(".", 3)
        self.assertTrue(isinstance(self.series3, ErroredEvaluation))

    def test_error_on_malformed(self):
        self.series4 = GenericEvaluation.parse(".", 4)
        self.assertTrue(isinstance(self.series4, ErroredEvaluation))

    def test_numpy_array(self):
        self.assertTrue(np.allclose(
            self.series0.getNumpyArray(), 
            np.array([1, 2, 2, 1, 0])))

        self.assertTrue(np.allclose(
            self.series1.getNumpyArray(), 
            np.array([1.5, 2.5, 1.5, 1])))

    def test_numpy_array_like(self):
        self.assertTrue(np.allclose(
            self.series1.getNumpyArrayLike(self.series0),
            np.array([1.5, 2, 2, 1, 1])))

        self.assertTrue(np.allclose(
            self.series0.getNumpyArrayLike(self.series1),
            np.array([1.5, 2, 1.5, 1])))

    def test_numpy_array_like_same_format(self):
        self.assertTrue(np.allclose(
            self.series1.getNumpyArrayLike(self.series1),
            np.array([1.5, 2.5, 1.5, 1])))

if __name__ == '__main__':
    unittest.main()
