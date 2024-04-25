"""Module for generic evaluations, which are evaluations containing only one scalar value at 
multiple timesteps, stored in json or csv format."""
import os
import csv
import json
import numpy as np
from .evaluation import Evaluation, ErroredEvaluation


class GenericEvaluation(Evaluation):
    """Class implementing a parser for evaluations containing only one
    scalar value at multiple timesteps, stored in the following json format:

    .. code-block:: none

        {
            "metadata": {
                "finished": true
            },
            "data": [
                {
                    "time": 0.1,
                    "value": 0.35
                },
                {
                    "time": 0.2,
                    "value": 0.34
                }
                ....
            ]
        }

    or this CSV format:

    .. code-block:: none

        step,time,value
        1,0.1,0.35
        2,0.2,0.34
        ...
        FINISHED,,

    """

    data = []
    times = []

    def __init__(self, data, times, eval_id=-1, parameters=None, runtime=None):
        """ Class constructor

        :param data: 1d array of numbers representing the measured for each timestep
        :type data: list of numbers
        :param times: the times measured (in simulation time)
        :type times: list of numbers
        :param eval_id: id of the evaluation this data resulted from
        :type eval_id: int, optional
        :param parameters: (transformed) parameters of the evaluation this data resulted from
        :type parameters: numpy array, optional
        :param runtime: runtime of the evaluation this data resulted from, in seconds
        :type runtime: int, optional
        """
        self.data = data
        self.times = times
        self.eval_id = eval_id
        self.parameters = parameters
        self.runtime = runtime

    @property
    def timeCount(self):
        """Returns the number of measurements stored in this object

        :return: Number of measurements stored in this object
        :rtype: Integer
        """
        return len(self.times)

    def getNumpyArray(self):
        """Returns stored measurements as a 1d numpy array

        :return: stored measurements as a 1d numpy array
        :rtype: numpy array, 1d
        """
        return np.array(self.data)

    def getNumpyArrayLike(self, target):
        """Used to interpolate between different evaluations, when timestamps might differ because
        of the used time control schemes.

        :param target: Evaluation whichs format should be matched and interpolated to
        :type target: Evaluation
        :raises IncompatibleFormatError: When the two Evaluations can not be interpolated between
        :return: the data of this evaulation, interpolated to the targets format
        :rtype: numpy array
        """

        if not isinstance(target, GenericEvaluation):
            raise Evaluation.IncompatibleFormatError("Target not compatible!")

        array = np.zeros(target.timeCount)
        array_values = []

        # split array at discontinuities
        def split_sorted_array(arr, split_arr):
            split_indices = np.where(np.diff(arr) < 0)[0] + 1
            split_arr = np.split(split_arr, split_indices)
            return split_arr

        split_times = split_sorted_array(self.times, self.times)
        split_data = split_sorted_array(self.times, self.data)
        split_target = split_sorted_array(target.times, target.times)

        if len(split_times) > 1 and len(split_times) != len(split_target):
            raise Evaluation.IncompatibleFormatError("Target and data not the same discontinuities")

        # interpolate
        for i, target_group in enumerate(split_target):  # iterate over target groups
            array_values.append(np.zeros(len(target_group)))  # create array for this group
            for j, target_time in enumerate(target_group):  # iterate over target times
                # find nearest entries in this instances time list
                nearest_lower = 0

                # first, find the time with the maximum index lower or equal to targettime
                while True:
                    if split_times[i][nearest_lower] == target_time:
                        # found perfect match!
                        array_values[i][j] = split_data[i][nearest_lower]
                        break

                    if nearest_lower == len(split_times[i]) - 1:
                        # at the edge...
                        array_values[i][j] = split_data[i][nearest_lower]
                        break

                    if split_times[i][nearest_lower] < target_time:
                        if split_times[i][nearest_lower + 1] > target_time:
                            # found it
                            # interpolate
                            higherdata = np.array(split_data[i][nearest_lower + 1])
                            highertime = split_times[i][nearest_lower + 1]
                            lowerdata = np.array(split_data[i][nearest_lower])
                            lowertime = split_times[i][nearest_lower]

                            percentage = (target_time - lowertime) / (highertime - lowertime)
                            interpolated = percentage * higherdata + (1 - percentage) * lowerdata

                            array_values[i][j] = interpolated
                            break
                        nearest_lower += 1

                    # if we are here, split_times[i][nearest_lower] > target_time
                    if nearest_lower == 0:
                        array_values[i][j] = split_data[i][nearest_lower]
                        break

        array = np.concatenate(array_values)
        return array

    @classmethod
    def fromCSV(cls, filename, evaluation_id=-1, parameters=None, runtime=None):
        """Parses this evaluation from the csv format described

        :param filename: file to parse
        :type filename: string
        :param evaluation_id: id of the evaluation this data resulted from
        :type evaluation_id: int, optional
        :param parameters: (transformed) parameters of the evaluation this data resulted from
        :type parameters: numpy array, optional
        :param runtime: runtime of the evaluation this data resulted from, in seconds
        :type runtime: int, optional
        :return: the parsed evaluation, or ErroredEvaluation if an error occurred.
        :rtype: Evaluation
        """

        parsedevaluation = cls([], [], evaluation_id, parameters, runtime)

        with open(filename) as csvfile:
            reader = csv.DictReader(csvfile)

            isfinished = False
            for row in reader:
                if list(row.values())[0].startswith("#") or list(row.keys())[0].startswith("#"):  # skip lines starting with #
                    continue

                if "value" not in row or "time" not in row:
                    return ErroredEvaluation(parameters,
                                             "Malformed data entry!",
                                             evaluation_id,
                                             runtime)

                if list(row.values())[0] == "FINISHED":
                    isfinished = True
                    break

                # ignore header in all possibilities because of multiprocessing
                numeric_data = True
                for key in ["value", "time"]:
                    if "value" in row[key] or "time" in row[key] or "step" in row[key]:
                        numeric_data = False
                        break
                if not numeric_data:
                    continue

                parsedevaluation.data.append(float(row["value"]))
                parsedevaluation.times.append(float(row["time"]))

        if isfinished:
            return parsedevaluation
        return ErroredEvaluation(parameters,
                                 "Evaluation did not finish correctly",
                                 evaluation_id,
                                 runtime)

    @classmethod
    def fromJSON(cls, filename, evaluation_id=-1, parameters=None, runtime=None):
        """Parses this evaluation from the json format described

        :param filename: file to parse
        :type filename: string
        :param evaluation_id: id of the evaluation this data resulted from
        :type evaluation_id: int, optional
        :param parameters: (transformed) parameters of the evaluation this data resulted from
        :type parameters: numpy array, optional
        :param runtime: runtime of the evaluation this data resulted from, in seconds
        :type runtime: int, optional
        :return: the parsed evaluation, or ErroredEvaluation if an error occurred.
        :rtype: Evaluation
        """
        # parse the file
        parsedjson = {}
        with open(filename) as jsonfile:
            try:
                parsedjson = json.load(jsonfile)
            except json.JSONDecodeError as exception:
                return ErroredEvaluation(parameters,
                                         "Error parsing json file: " + exception.msg,
                                         evaluation_id,
                                         runtime)

        # check correct format
        if "data" not in parsedjson \
           or "metadata" not in parsedjson \
           or "finished" not in parsedjson["metadata"]:
            return ErroredEvaluation(parameters,
                                     "Evaluation json is malformed.",
                                     evaluation_id,
                                     runtime)

        # check that the evaluation did finish correctly
        if not parsedjson["metadata"]["finished"]:
            return ErroredEvaluation(parameters,
                                     "Evaluation did not finish correctly",
                                     evaluation_id,
                                     runtime)

        parsedevaluation = cls([], [], evaluation_id, parameters, runtime)

        # parse data into internal arrays
        for element in parsedjson["data"]:
            if "value" not in element or "time" not in element:
                return ErroredEvaluation(parameters,
                                         "Malformed data entry!",
                                         evaluation_id,
                                         runtime)
            parsedevaluation.data.append(element["value"])
            parsedevaluation.times.append(element["time"])

        return parsedevaluation

    @classmethod
    def parse(cls, directory, evaluation_id, parameters=None, runtime=None):
        """Factory method, parses the evaluation with a given id from the given folder.
        Sets the parameters and runtime as metaobjects for later analysis.

        :param directory: directory to read the evaluation from
        :type directory: string
        :param evaluation_id: id of the evaluation to find the correct file fron directory
        :type evaluation_id: int
        :param parameters: the (transformed) parameters of this evaluation
        :type parameters: numpy array
        :param runtime: runtime of the evaluation, in seconds
        :type runtime: int
        :return: Parsed Evaluation
        :rtype: Evaluation
        """

        # construct filename
        filename_json = os.path.join(directory, str(evaluation_id) + "_measurement.json")
        filename_csv = os.path.join(directory, str(evaluation_id) + "_measurement.csv")

        if not os.path.isfile(filename_csv):
            if not os.path.isfile(filename_json):
                return ErroredEvaluation(parameters,
                                         "No measurement file found.",
                                         evaluation_id,
                                         runtime)
            return GenericEvaluation.fromJSON(filename_json, evaluation_id, parameters, runtime)
        return GenericEvaluation.fromCSV(filename_csv, evaluation_id, parameters, runtime)
