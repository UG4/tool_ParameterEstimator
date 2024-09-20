"""Unit tests for the GenericEvaluation class in UGParameterEstimator"""
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
        """
        Test the read-in functionality of the evaluation series.
        This test verifies that the series0 and series1 objects are not instances
        of the ErroredEvaluation class. It also checks that the 'times' and 'data'
        attributes of these series match the expected values.
        Assertions:
            - series0 is not an instance of ErroredEvaluation.
            - series1 is not an instance of ErroredEvaluation.
            - series0.times equals [1, 2, 3, 4, 5].
            - series1.times equals [1.5, 2.5, 3.5, 4].
            - series0.data equals [1, 2, 2, 1, 0].
            - series1.data equals [1.5, 2.5, 1.5, 1].
        """
        self.assertFalse(isinstance(self.series0, ErroredEvaluation))
        self.assertFalse(isinstance(self.series1, ErroredEvaluation))

        self.assertEqual(self.series0.times, [1, 2, 3, 4, 5])
        self.assertEqual(self.series1.times, [1.5, 2.5, 3.5, 4])
        self.assertEqual(self.series0.data, [1, 2, 2, 1, 0])
        self.assertEqual(self.series1.data, [1.5, 2.5, 1.5, 1])

    def test_parse_csv(self):
        """
        Test the `parse` method of the `GenericEvaluation` class.
        This test checks the following:
        - The method correctly parses a CSV file located in the current directory.
        - The parsed result is not an instance of `ErroredEvaluation`.
        - The `times` attribute of the parsed result matches the expected list `[0.5, 1, 2]`.
        - The `data` attribute of the parsed result matches the expected list `[3, 2.4, 4.623]`.
        If the parsed result is an instance of `ErroredEvaluation`, the reason for the error is 
        printed.
        """
        series2 = GenericEvaluation.parse(".", 2)
        if isinstance(series2, ErroredEvaluation):
            print(series2.reason)
        self.assertFalse(isinstance(series2, ErroredEvaluation))
        self.assertEqual(series2.times, [0.5, 1, 2])
        self.assertEqual(series2.data, [3, 2.4, 4.623])

    def test_error_on_not_finished(self):
        """
        Test case to verify that an error is raised when the evaluation is not finished.
        This test checks if the `GenericEvaluation.parse` method correctly returns an 
        instance of `ErroredEvaluation` when provided with an unfinished evaluation.
        Steps:
        1. Parse the evaluation with a specific identifier.
        2. Assert that the returned object is an instance of `ErroredEvaluation`.
        Asserts:
            - The parsed evaluation should be an instance of `ErroredEvaluation`.
        """
        series3 = GenericEvaluation.parse(".", 3)
        self.assertTrue(isinstance(series3, ErroredEvaluation))

    def test_error_on_malformed(self):
        """
        Test case for handling malformed input in GenericEvaluation.
        This test verifies that the `GenericEvaluation.parse` method correctly
        identifies and handles malformed input by returning an instance of
        `ErroredEvaluation`.
        Steps:
        1. Parse a malformed input using `GenericEvaluation.parse`.
        2. Assert that the result is an instance of `ErroredEvaluation`.
        Raises:
            AssertionError: If the result is not an instance of `ErroredEvaluation`.
        """
        series4 = GenericEvaluation.parse(".", 4)
        self.assertTrue(isinstance(series4, ErroredEvaluation))

    def test_numpy_array(self):
        """
        Test the conversion of series data to numpy arrays.
        This test verifies that the `getNumpyArray` method of the series objects
        correctly converts the series data to numpy arrays. It checks the following:
        - numpy array representation of `self.series0` matches expected array [1, 2, 2, 1, 0].
        - numpy array representation of `self.series1` matches expected array [1.5, 2.5, 1.5, 1].
        Test uses `np.allclose` to ensure that the arrays are element-wise equal within a tolerance.
        """
        self.assertTrue(np.allclose(
            self.series0.getNumpyArray(),
            np.array([1, 2, 2, 1, 0])))

        self.assertTrue(np.allclose(
            self.series1.getNumpyArray(),
            np.array([1.5, 2.5, 1.5, 1])))

    def test_numpy_array_like(self):
        """
        Test the `getNumpyArrayLike` method for numpy array-like conversion.
        This test verifies that the `getNumpyArrayLike` method correctly converts
        the series to a numpy array-like structure that matches the expected output.
        Assertions:
            - The numpy array-like structure obtained from `series1` when converted
              using `series0` matches the expected numpy array `[1.5, 2, 2, 1, 1]`.
            - The numpy array-like structure obtained from `series0` when converted
              using `series1` matches the expected numpy array `[1.5, 2, 1.5, 1]`.
        """
        self.assertTrue(np.allclose(
            self.series1.getNumpyArrayLike(self.series0),
            np.array([1.5, 2, 2, 1, 1])))

        self.assertTrue(np.allclose(
            self.series0.getNumpyArrayLike(self.series1),
            np.array([1.5, 2, 1.5, 1])))

    def test_numpy_array_like_same_format(self):
        """
        Test that the `getNumpyArrayLike` method returns a numpy array with the 
        same format as the input series.
        This test checks if the `getNumpyArrayLike` method of `series1` returns 
        a numpy array that is element-wise equal to the expected numpy array 
        [1.5, 2.5, 1.5, 1].
        Asserts:
            True if the numpy array returned by `getNumpyArrayLike` is element-wise 
            equal to the expected numpy array.
        """
        self.assertTrue(np.allclose(
            self.series1.getNumpyArrayLike(self.series1),
            np.array([1.5, 2.5, 1.5, 1])))

    def test_numpy_array_like_different_length(self):
        """
        Test the `getNumpyArrayLike` method for handling series with different lengths.
        This test sets up `series0` with specific `times` and `data` values, and then
        checks if the `getNumpyArrayLike` method of `series1` correctly transforms
        `series0` into a numpy array that matches the expected output.
        The test verifies that the resulting numpy array is close to the expected
        array `[1.5, 2, 2, 1]` using `np.allclose`.
        Asserts:
            True if the numpy array returned by `getNumpyArrayLike` matches the expected
            array `[1.5, 2, 2, 1]`.
        """

        self.series0.times = [1, 2, 3, 4]
        self.series0.data = [1, 2, 2, 1]
        self.assertTrue(np.allclose(
            self.series1.getNumpyArrayLike(self.series0),
            np.array([1.5, 2, 2, 1])))

    def test_numpy_array_like_discontinuities(self):
        """
        Test the `getNumpyArrayLike` method for handling discontinuities in time series data.
        The test sets up a time series with discontinuous time points and corresponding data values.
        It then verifies that the `getNumpyArrayLike` method raises an `IncompatibleFormatError`
        when attempting to process this discontinuous data.
        Raises:
            GenericEvaluation.IncompatibleFormatError: If the time data contains discontinuities.
        """

        self.series0.times = [1, 2, 3, 1, 2, 3]
        self.series0.data = [1, 2, 2, 1, 2, 3]
        with self.assertRaises(GenericEvaluation.IncompatibleFormatError):
            self.series0.getNumpyArrayLike(self.series1)

    def test_numpy_array_like_times_not_sorted(self):
        """
        Test case for the getNumpyArrayLike method to ensure it raises an
        IncompatibleFormatError when the times attribute of the series is not sorted.
        This test sets up a series with unsorted times and corresponding data,
        and verifies that the getNumpyArrayLike method raises the appropriate
        exception when called with another series.
        Raises:
            GenericEvaluation.IncompatibleFormatError: If the times attribute
            of the series is not sorted.
        """

        self.series0.times = [1, 2, 3, 2, 1]
        self.series0.data = [1, 2, 2, 1, 2]
        with self.assertRaises(GenericEvaluation.IncompatibleFormatError):
            self.series0.getNumpyArrayLike(self.series1)

if __name__ == '__main__':
    unittest.main()
