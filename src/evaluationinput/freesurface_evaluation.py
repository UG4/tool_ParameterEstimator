"""Module for freesurface evaluations in the UGParameterEstimator."""
from enum import Enum
import math
import os
import csv
import struct
import subprocess
import numpy as np
from .evaluation import Evaluation, ErroredEvaluation

class FreeSurfaceEvaluation(Evaluation):
    """Base class for all Evaluation classes containing measurements of free surface positions.
    """

    # 2d array containg measured heights
    data = [[]]

    # array containing locations (as metadata)
    locations = []

    # detected dimension (2d or 1d)
    dimension = -1

    # array containg times of measurements
    times = []

    NaNHandling = Enum("NaNHandling", "none replace")

    nanhandling = NaNHandling.none
    nanreplacevalue = 0.0

    @property
    def timeCount(self):
        """Returns the number of measurements stored in this object

        :return: Number of measurements stored in this object
        :rtype: Integer
        """
        return len(self.times)

    @property
    def locationCount(self):
        """Returns the number of locations/measurements points for every measurement in this object

        :return: Number of measurement points for each measurement stored in this object
        :rtype: Integer
        """
        return len(self.locations)

    @property
    def totalCount(self):
        """Returns the number of measured free surface heights stored.

        :return: number of measured free surface heights stored.
        :rtype: Integer
        """
        return len(self.times)*len(self.locations)

    def getNumpyArray(self):
        """Returns stored measurements as a 1d numpy array

        :return: stored measurements as a 1d numpy array
        :rtype: numpy array with size totalCount
        """
        return np.reshape(np.array(self.data),-1)

    @staticmethod
    def hasSameLocations(A, B):
        """Compares the locations of 2 free surface measurement objects

        :param A: FreeSurfaceEvaluation A for comparison
        :type A: FreeSurfaceEvaluation
        :param B: FreeSurfaceEvaluation B for comparison
        :type B: FreeSurfaceEvaluation
        :return: true, if the 2 objects have the same locations, false, if not
        :rtype: boolean
        """
        if A.locationCount != B.locationCount:
            return False
        for l in range(A.locationCount):
            if A.dimension == 2:
                if math.fabs(B.locations[l]-A.locations[l]) > 0.001:
                    print("At location " +
                          str(l) +
                          ": target: " +
                          str(B.locations[l]) +
                          ", measurement: " +
                          str(A.locations[l]))
                    return False
            else:
                if math.fabs(B.locations[l][0] - A.locations[l][0]) > 0.001 or \
                     math.fabs(B.locations[l][1] - A.locations[l][1]) > 0.001:
                    print("At location " + str(l) + ": ")
                    print("target: " + str(B.locations[l]) + ", measurement: " + str(A.lcations[l]))
                    return False

        return True

    @classmethod
    def parse(cls, directory, evaluation_id, parameters, runtime):
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
        :raises IncompatibleFormatError: When the Evaluation can not be parsed
        :return: Parsed FreeSurfaceEvaluation
        :rtype: FreeSurfaceEvaluation
        """
        raise NotImplementedError("Abstract class FreeSurfaceEvaluation doesn't implement parse()")

    def getNumpyArrayLike(self, target):
        """Used to interpolate between different evaluations, when timestamps might differ because
        of the used time control schemes.

        :param target: FreeSurfaceEvaluation whichs format should be matched and interpolated to
        :type target: FreeSurfaceEvaluation
        :raises IncompatibleFormatError: When the two Evaluations can not be interpolated between
        :return: the data of this evaulation, interpolated to the targets format
        :rtype: numpy array with the dimensions 1 x target.totalCount
        """
        raise NotImplementedError("Abstract class FreeSurfaceEvaluation doesn't implement getNumpyArrayLike()")


class FreeSurfaceEquilibriumEvaluation(FreeSurfaceEvaluation):
    """Class representing the measurement of the free surface position when the free surface has
    reached its equlibrium state. this means this evaluation onyl contains data for one timestep,
    but for multiple locations.
    """
    def __init__(self, data, locations, dimension, time=0):
        """Class constructor
        :param data: 1-dimensional data array
        :type data: list of numbers
        :param locations: list of locations for this measurement
        :type locations: list of numbers
        :param dimension: dimension of the problem
        :type dimension: int
        :param time: time of the measurement (only one time here!)
        :type time: number, optional
        """
        self.data = data
        self.locations = locations
        self.dimension = dimension
        self.times = [time]

    @classmethod
    def fromCSV(cls, filename, dim, delimiter=',', valuecolumn="Value", dimcolumns=["X", "Y"]):
        """ Reads the free surface evaluation from a .csv file.

        :param filename: filename to load
        :type filename: string
        :param dim: dimension of the problem
        :type dim: int
        :param valuecolumn: name of the column containg the measured height, defaults to "Value"
        :type valuecolumn: string, optinal
        :param dimcolumns: names of the columns containg the locations data, defaults to ["X", "Y"]
        :type dimcolumns: list of strings, optional, size == dimension
        """
        data = [[]]
        locations = []
        with open(filename) as csvfile:
            reader = csv.DictReader(csvfile, delimiter=delimiter)
            for row in reader:
                value = float(row[valuecolumn])
                if math.isnan(value) and FreeSurfaceEquilibriumEvaluation.nanhandling == FreeSurfaceEvaluation.NaNHandling.replace:
                    value = FreeSurfaceEquilibriumEvaluation.nanreplacevalue
                data[-1].append(value)
                if dim == 2:
                    locations.append(float(row[dimcolumns[0]]))
                elif dim == 3:
                    locations.append((float(row[dimcolumns[0]]), float(row[dimcolumns[1]])))

        return cls(data, locations, dim)

    @classmethod
    def fromTimedependentTimeseries(cls, series):
        """ Constructs a FreeSurfaceEquilibriumEvaluation from a timedependent evaluation 
        by only using the last measurement. Only use this if the timedependent evaluation was in 
        a equilibrium state!

        :param series: evaluation to convert
        :type series: FreeSurfaceTimeDependentEvaluation
        :return: the constructed FreeSurfaceEquilibriumEvaluation
        :rtype: FreeSurfaceEquilibriumEvaluation
        """
        data_reformatted = [series.data[-1]]
        dim = 2
        if hasattr(series, "dimension"):
            dim = series.dimension
        returnval = cls(data_reformatted, series.locations, dim, series.times[-1])

        returnval.parameters = series.parameters

        return returnval

    @classmethod
    def fromNumpyArray(cls, data, seriesformat):
        """ Constructs a FreeSurfaceEquilibriumEvaluation from a numpy array 
        and a given FreeSurfaceEquilibriumEvaluation of the same format (locations and dimension)

        :param data: data to use
        :type data: numpy array of length seriesformat.totalCount
        :param seriesformat: evaluation to use the format of
        :type seriesformat: FreeSurfaceEquilibriumEvaluation
        :return: the constructed FreeSurfaceEquilibriumEvaluation
        :rtype: FreeSurfaceEquilibriumEvaluation
        """
        data_reformatted = np.array(data).reshape((seriesformat.timeCount,
                                                   seriesformat.locationCount)).tolist()
        dim = 2
        if hasattr(seriesformat, "dimension"):
            dim = seriesformat.dimension
        return cls(data_reformatted, seriesformat.locations, dim, seriesformat.times[0])

    @staticmethod
    def writePlots(series, filename, yaxislabel="$m(l,t,\\vec{\\theta})$ {[m]}"):
        """ write multiple plots as latex to a file.
        This uses the tikzpicture/pgfplots package in the generated latex code.

        :param series: multiple evaluations to plot, given as a dictionary. 
                The key will be used as the name in the legend, the value being the evaluation.
        :type series: dictionary<string, FreeSurfaceEquilibriumEvaluation>
        :param filename: filename to save to
        :type filename: string
        :param yaxislabel: label of the y axis
        :type yaxislabel: string
        """
        plot = ""
        plot += "\t\\begin{tikzpicture}\n"
        plot += "\t\t\\begin{axis}[\n"
        plot += "	xlabel=Ort l {[m]},\n"
        plot += "	width=10cm,\n"
        plot += "	ylabel={" + yaxislabel + "},\n"
        plot += "	legend style={\n"
        plot += "		anchor=north west,at={(axis description cs:1.01,1)}} ]\n"

        for k in series:
            seriesobject = series[k]["eval"]

            if "dashed" in series[k] and series[k]["dashed"]:
                plot += "\t\t\\addplot+[thick,mark=*,dashed]\n"
            else:
                plot += "\t\t\\addplot+[thick,mark=*]\n"
            plot += "\t\t table [x={location}, y={value}]{ \n"
            plot += "location\t value\n"        

            sortedindices = np.argsort(np.array(seriesobject.locations))
            for l in range(seriesobject.locationCount):
                plot += str(seriesobject.locations[sortedindices[l]]) + \
                    "\t" + str(seriesobject.data[0][sortedindices[l]]) + "\n"

            plot += "};\n"

            plot += "\t\t\\addlegendentry{" + k + "};"


        plot += "\t\t\\end{axis}\n"
        plot += "\t\\end{tikzpicture}\n"

        if not filename is None:
            with open(filename, "w") as f:
                f.write(plot)
        return plot

    def getNumpyArrayLike(self, target):
        raise NotImplementedError("FreeSurfaceEquilibrium doesn't implement getNumpyArrayLike()")

class BinaryReader:
    """ helper class to read from a binary stram"""

    def __init__(self, filename, endian="<"):
        self.file = open(filename, "rb")
        self.endian = endian

    @property
    def readable(self):
        peek = self.file.peek(1)
        return peek != b''

    def read_int(self):
        return struct.unpack(self.endian + "i", self.file.read(4))[0]

    def read_double(self):
        return struct.unpack(self.endian + "d", self.file.read(8))[0]

    def read_char(self):
        return struct.unpack(self.endian + "b", self.file.read(1))[0]

    def close(self):
        self.file.close()

class FreeSurfaceTimeDependentEvaluation(FreeSurfaceEvaluation):
    """Class representing the measurement of the free surface position at multiple time points.
    The underlying data is a 2d array.
    """
    EQUILIBRIUM_CONSTANT = 10

    def __init__(self, data, times, locations, dimension, eval_id=-1, parameters=None, runtime=None):
        """ Class constructor

        :param data: 2d array of numbers, first dimension: time, second(inner) dimension location
        :type data: list of list of numbers
        :param times: the times measured (in simulation time)
        :type times: list of numbers
        :param locations: the locations measured
        :type locations: list of numbers or list of tuples (3d case)
        :param dimension: dimension of the problem
        :type dimension: int
        :param eval_id: id of the evaluation this data resulted from
        :type eval_id: int, optional
        :param parameters: (transformed) parameters of the evaluation this data resulted from
        :type parameters: numpy array, optional
        :param runtime: runtime of the evaluation this data resulted from, in seconds
        :type runtime: int, optional
        """
        self.data = data
        self.times = times
        self.locations = locations
        self.dimension = dimension
        self.eval_id = eval_id
        self.parameters = parameters
        self.runtime = runtime

    @classmethod
    def parseBinary(cls, file, evaluation_id=-1, parameters=None, runtime=None):
        """ Factory method to parse a binary measurement file.
        This uses the format defined in fs_measurement.hpp in the d3f-plugin.

        :param file: file to parse
        :type file: string
        :param evaluation_id: id of the evaluation this data resulted from
        :type evaluation_id: int, optional
        :param parameters: (transformed) parameters of the evaluation this data resulted from
        :type parameters: numpy array, optional
        :param runtime: runtime of the evaluation this data resulted from, in seconds
        :type runtime: int, optional
        """
        data = []
        times = []
        locations = []
        finished = False
        dimension = -1

        reader = BinaryReader(file)

        dimension = reader.read_int()

        if dimension not in [2,3]:
            return ErroredEvaluation(parameters, "Error parsing dimension.", evaluation_id, runtime)

        while reader.readable:
            status = reader.read_char()
            if status == 1:
                time = reader.read_double()
                if not time in times:
                    times.append(time)
                    data.append([])

                if dimension == 2:
                    location = reader.read_double()
                elif dimension == 3:
                    location = (reader.read_double(), reader.read_double())

                if location not in locations:
                    locations.append(location)


                value = reader.read_double()
                if math.isnan(value) and FreeSurfaceTimeDependentEvaluation.nanhandling == \
                    FreeSurfaceEvaluation.NaNHandling.replace:
                    value = FreeSurfaceTimeDependentEvaluation.nanreplacevalue
                data[-1].append(value)

            elif status == 2:
                finished = True
                break

        reader.close()

        if finished:
            return cls(data, times, locations, dimension, evaluation_id, parameters, runtime)
        return ErroredEvaluation(parameters, "UG run did not finish.", evaluation_id, runtime)

    @classmethod
    def parse(cls, directory, evaluation_id, parameters=None, runtime=None):
        """ Factory method to parse a measurement file.
        Parses the measurement as binary, if the corresponding file exists, or as csv, if not.
        This uses the format defined in fs_measurement.hpp in the d3f-plugin.

        :param directory: directory of the evaluation to parse
        :type directory: string
        :param evaluation_id: id of the evaluation to parse
        :type evaluation_id: int
        :param parameters: (transformed) parameters of the evaluation this data resulted from
        :type parameters: numpy array, optional
        :param runtime: runtime of the evaluation this data resulted from, in seconds
        :type runtime: int, optional
        """
        filenameBin = os.path.join(directory, str(evaluation_id) + "_measurement.bin")
        filenameCSV = os.path.join(directory, str(evaluation_id) + "_measurement.csv")

        if os.path.isfile(filenameBin):
            return FreeSurfaceTimeDependentEvaluation.parseBinary(filenameBin,
                                                                  evaluation_id,
                                                                  parameters,
                                                                  runtime)
        if os.path.isfile(filenameCSV):
            return FreeSurfaceTimeDependentEvaluation.parseFromCSV(filenameCSV,
                                                                   evaluation_id,
                                                                   parameters,
                                                                   runtime)
        return ErroredEvaluation(parameters, "No measurement file found.", evaluation_id, runtime)


    @classmethod
    def parseFromCSV(cls, filename, evaluation_id=-1, parameters=None, runtime=None):
        """ Factory method to parse a csv measurement file.
        This uses the format defined in fs_measurement.hpp in the d3f-plugin.

        :param filename: file to parse
        :type filename: string
        :param evaluation_id: id of the evaluation this data resulted from
        :type evaluation_id: int, optional
        :param parameters: (transformed) parameters of the evaluation this data resulted from
        :type parameters: numpy array, optional
        :param runtime: runtime of the evaluation this data resulted from, in seconds
        :type runtime: int, optional
        """
        data = []
        times = []
        locations = []
        finished = False
        dimension = -1

        with open(filename) as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                if dimension == -1:
                    if "dim1" in row:
                        dimension = 3
                    elif "dim0" in row:
                        dimension = 2
                    else:
                        raise Evaluation.IncompatibleFormatError("Could not parse " + filename)

                if row["step"] == "FINISHED":
                    finished = True
                    break

                time = float(row["time"])
                if time not in times:
                    times.append(time)
                    data.append([])

                if dimension == 2:
                    location = float(row["dim0"])
                elif dimension == 3:
                    location = (float(row["dim0"]), float(row["dim1"]))

                value = float(row["z"])
                if math.isnan(value) and FreeSurfaceTimeDependentEvaluation.nanhandling == \
                    FreeSurfaceEvaluation.NaNHandling.replace:
                    value = FreeSurfaceTimeDependentEvaluation.nanreplacevalue
                data[-1].append(value)

                if not location in locations:
                    locations.append(location)

        if finished:
            return cls(data, times, locations, dimension, evaluation_id, parameters, runtime)
        return ErroredEvaluation(parameters, "UG run did not finish.", evaluation_id, runtime)

    @classmethod
    def fromNumpyArray(cls, data, seriesformat):
        """ Constructs a FreeSurfaceTimeDependentEvaluation from a numpy array 
        and a given FreeSurfaceTimeDependentEvaluation of the same format (locations and dimension)

        :param data: data to use
        :type data: numpy array of length seriesformat.totalCount
        :param seriesformat: evaluation to use the format of
        :type seriesformat: FreeSurfaceTimeDependentEvaluation
        :return: the constructed FreeSurfaceTimeDependentEvaluation
        :rtype: FreeSurfaceTimeDependentEvaluation
        """
        data_reformatted = np.array(data).reshape((seriesformat.timeCount,
                                                   seriesformat.locationCount)).tolist()
        dim = 2
        if hasattr(seriesformat, "dimension"):
            dim = seriesformat.dimension
        return cls(data_reformatted, seriesformat.times, seriesformat.locations, dim)

    def getNumpyArrayLike(self, target: FreeSurfaceEvaluation):
        """Used to interpolate between different evaluations, when timestamps might differ because
        of the used time control schemes.

        Important Note: Care has to be taken with the simulation time frames. If the target 
        timeframe lasts longer or starts earlier, the next suitable timestep (i.e. the first or 
        last one) will be used automatically. This is not always desirable!

        :param target: FreeSurfaceEvaluation whichs format should be matched and interpolated to
        :type target: FreeSurfaceEvaluation
        :return: the data of this evaulation, interpolated to the targets format
        :rtype: numpy array with the dimensions 1 x target.totalCount
        """
        if (not isinstance(target, FreeSurfaceEquilibriumEvaluation)) and \
            (not isinstance(target, FreeSurfaceTimeDependentEvaluation)):
            raise Evaluation.IncompatibleFormatError("Target not compatible!")

        if not FreeSurfaceTimeDependentEvaluation.hasSameLocations(self, target):
            raise Evaluation.IncompatibleFormatError("Not the same locations!")

        if isinstance(target, FreeSurfaceEquilibriumEvaluation):
            return np.array(self.data[-1])


        array = np.zeros(len(target.times)*len(target.locations))
        for i, targettime in enumerate(target.times):
            # find nearest entries in this instances time field
            nearest_lower = 0

            # first, find the time with the maximum index lower or equal to targettime
            while True:
                if self.times[nearest_lower] == targettime:
                    # found a perfect match!
                    array[i*len(target.locations): \
                        ((i+1)*len(target.locations))] = \
                        self.data[nearest_lower]
                    break

                if nearest_lower == self.timeCount-1:
                    # at the edge...
                    array[i*len(target.locations): \
                        ((i+1)*len(target.locations))] = \
                        self.data[nearest_lower]
                    break

                if self.times[nearest_lower] < targettime:
                    if self.times[nearest_lower+1] > targettime:
                        # found it
                        # interpolate
                        higherdata = np.array(self.data[nearest_lower+1])
                        highertime = self.times[nearest_lower+1]
                        lowerdata = np.array(self.data[nearest_lower])
                        lowertime = self.times[nearest_lower]

                        percentage = (targettime-lowertime) / (highertime-lowertime)
                        interpolated = percentage*higherdata + (1-percentage)*lowerdata

                        array[i*len(target.locations): \
                            ((i+1)*len(target.locations))] = \
                            interpolated
                        break
                    nearest_lower += 1

                # if we are here, self.times[nearest_lower] > targettime....
                if nearest_lower == 0:
                    array[i*len(target.locations): \
                        ((i+1)*len(target.locations))] = \
                        self.data[nearest_lower]
                    break

        return array

    def writeCSVAveragedOverLocation(self, filename):
        """Writes a tsv with a entry for every timestep measured. The entry will be the
        average measured height over all locations at this timestep.

        :param filename: filename to write to
        :type filename: string
        """
        with open(filename,"w") as f:
            f.write("time \t value\n")
            for t in range(self.timeCount):
                summed_up = sum(self.data[t])
                average = summed_up/self.locationCount
                f.write(str(self.times[t]) + "\t" + str(average) + "\n")

    def writeCSVAtLocation(self, filename, location):
        """Writes a tsv with a entry for every timestep measured. The entry will be the
        measured height at the given location.

        :param filename: filename to write to
        :type filename: string
        :param location: location to use
        :type location: number (2d) or tuple of numbers (3d)
        """
        if location not in self.locations:
            print("illegal location specified!")
            return

        locindex = self.locations.index(location)

        with open(filename,"w") as f:
            f.write("time \t value\n")
            for t in range(self.timeCount):
                value = self.data[t][locindex]
                f.write(str(self.times[t]) + "\t" + str(value) + "\n")

    def writeCSVAveragedOverTimesteps(self, filename):
        """Writes a tsv with a entry for every location measured. The entry will be the
        average measured height over all timesteps at this location.

        :param filename: filename to write to
        :type filename: string
        """
        summed_up = np.zeros(self.locationCount)
        for t in range(self.timeCount):
            summed_up += np.array(self.data[t])

        with open(filename,"w") as f:
            f.write("location \t value\n")
            for l in range(self.locationCount):
                f.write(str(self.locations[l]) + "\t" + str(summed_up[l]) + "\n")

    def writeCSVAtTimestep(self, filename,timestep):
        """Writes a tsv with a entry for every location measured. The entry will be the
        measured height at the location for a given timestep.

        :param filename: filename to write to
        :type filename: string
        :param timestep: timestep (index) to use
        :type timestep: int
        """
        if timestep == -1:
            timestep = self.timeCount-1

        if(timestep < 0 or timestep >= self.timeCount):
            print("Illegal timestep specified!")
            return

        with open(filename,"w") as f:
            f.write("location \t value\n")
            for l in range(self.locationCount):
                f.write(str(self.locations[l]) + "\t" + str(self.data[timestep][l]) + "\n")

    def writeCSV(self, filename):
        """Writes a tsv with all times and locations measured.
        The data will be "flattened", with an table entry for every combination
        of time and location, with the columns time, location and value.

        :param filename: filename to write to
        :type filename: string
        """
        with open(filename, "w") as f:
            f.write("time\tlocation\t value\n")
            for t in range(self.timeCount):
                for l in range(self.locationCount):
                    f.write(str(self.times[t]) +
                            "\t" +
                            str(self.locations[l]) +
                            "\t" +
                            str(self.data[t][l]) +
                            "\n")

    def write3dPlot(self, filename, zlabel="$m(l,t,\\beta)$", scale=1, stride=3):
        """Writes a 3d plot in latex of this evaluation. On the x-axis will be time, on the 
        yaxis location and the z-axis will represent the data stored. The generated requires 
        tikzpicture/pgfplots to compile. Only 2d cases are supported.

        :param filename: filename to write to
        :type filename: string
        :param zlabel: label of the z axis
        :type zlabel: string
        :param scale: scale, will be passes to pgfplots, defualts to 1
        :type scale: number, optional
        :param stride: only plot every x-th timestep, defaults to 3
        :type stride: int, optional
        """

        if self.dimension != 2:
            raise Evaluation.IncompatibleFormatError("3d plot not available for 3d measurements")

        plot = ""

        if not filename is None:
            plot += "\\documentclass{standalone}\n"
            plot += "\\usepackage{pgfplots}\n"
            plot += "\\usepackage{tikz}\n"
            plot += "\\begin{document}\n"

        plot += "\t\\begin{tikzpicture}\n"
        plot += "\t\t\\begin{axis}[view/h=45,\n"
        plot += "	xlabel=Zeit $t$ {[s]},\n"
        plot += "	width=10cm,\n"
        plot += "	ylabel=Ort $l$ {[m]},\n"
        plot += "   scale=" + str(scale) + ",\n"
        plot += "	zlabel= " + zlabel + "]\n"
        plot += "\t\t\\addplot3 [surf,mesh/ordering=y varies,mesh/rows=" + \
            str(self.locationCount) + "]\n"
        plot += "\t\t table { \n"
        plot += "time\tlocation\t value\n"
        for t in range(self.timeCount):
            if t % stride != 0:
                continue
            for l in range(self.locationCount):
                plot += str(self.times[t]) + "\t" + str(self.locations[l]) + \
                    "\t" + str(self.data[t][l]) + "\n"
        plot += "};\n"
        plot += "\t\t\\end{axis}\n"
        plot += "\t\\end{tikzpicture}\n"

        if not filename is None:
            plot += "\\end{document}"
            with open(filename, "w") as f:
                f.write(plot)
            directory = os.path.dirname(filename)
            subprocess.run(
                ["pdflatex", "-interaction=nonstopmode", "-output-directory=" + directory,filename],
                stdout=subprocess.DEVNULL,
                check=True)

        return plot

    def writePlots(self, filename, num):
        """Writes 'num' 2d plots in latex in one diagram of this evaluation to show the 
        timedependency of the problem.
        On the x-axis will be location.
        The timesteps are chosen to be linear distributed in the time space, with 'num' entries.
        The generated requires tikzpicture/pgfplots to compile.
        Only 2d cases are supported.

        :param filename: filename to write to
        :type filename: string
        :param num: number of plots
        :type num: int
        """
        if self.dimension != 2:
            raise NotImplementedError("plot not available for 3d measurements")

        plot = ""
        plot += "\t\\begin{tikzpicture}\n"
        plot += "\t\t\\begin{axis}[\n"
        plot += "	xlabel={Ort $l$ {[m]}},\n"
        plot += "	width=10cm,\n"
        plot += "	ylabel={$m(l,t,\\vec{\\theta})$ {[m]}},\n"
        plot += "yticklabel style={/pgf/number format/fixed, /pgf/number format/precision=3},"
        plot += "	legend style={\n"
        plot += "		anchor=north west,at={(axis description cs:1.01,1)}} ]\n"

        idx = np.round(np.linspace(0, len(self.times) - 1, num)).astype(int)
        for t in idx:
            time = self.times[t]
            plot += "\t\t\\addplot+[thick,mark=*]\n"
            plot += "\t\t table [x={location}, y={value}]{ \n"
            plot += "location\t value\n"
            for l in range(self.locationCount):
                plot += str(self.locations[l]) + "\t" + str(self.data[t][l]) + "\n"
            plot += "};\n"
            plot += "\\addlegendentry{t=" + str(round(time, 3)) + "};\n"
        plot += "\t\t\\end{axis}\n"
        plot += "\t\\end{tikzpicture}\n"

        if not filename is None:
            with open(filename, "w") as f:
                f.write(plot)

        return plot

    def writeDifferentialPlot(self, filename):
        """Writes a plot in latex to show the amount the free surface moves in dependency of the
        time. On the x-axis will be time. The y-axis will represent the norm of the difference
        vector between two measurements.
        The generated requires tikzpicture/pgfplots to compile.

        :param filename: filename to write to
        :type filename: string
        """
        plot = ""
        plot += "\t\\begin{tikzpicture}\n"
        plot += "\t\t\\begin{axis}[\n"
        plot += "	xlabel=Zeit {[s]},\n"
        plot += "	width=10cm,\n"
        plot += "	ylabel=$||\\frac{\\delta m}{\\delta t}||_2$]\n"
        plot += "\t\t\\addplot [thick]\n"
        plot += "\t\t table [x={time}, y={value}]{ \n"
        plot += "time\t value\n"
        for t in range(self.timeCount-1):
            change = np.linalg.norm(np.array(self.data[t])-np.array(self.data[t+1]))
            plot += str(self.times[t]) + "\t" + str(change) + "\n"
        plot += "};\n"
        plot += "\t\t\\end{axis}\n"
        plot += "\t\\end{tikzpicture}\n"

        if not filename is None:
            with open(filename, "w") as f:
                f.write(plot)

        return plot

    def getFactorOfEquilibrium(self):
        """Calculate the equilibrium factor of the evaluation, a measurment for if the
        free surface has stopped moving.

        :return: the equilibrium factor
        :rtype: number
        """
        max_change = -float('inf')
        for t in range(self.timeCount-1):
            change = np.linalg.norm(np.array(self.data[t])-np.array(self.data[t+1]))
            change /= self.times[t+1]-self.times[t]

            max_change = max(max_change, change)
            # print("change between timestep " + str(t) + " and " + str(t+1) + " is " + str(change))

        last_change = np.linalg.norm(np.array(self.data[-1])-np.array(self.data[-2]))
        last_change /= self.times[-1]-self.times[-2]

        return max_change/last_change

    def isInEquilibrium(self):
        """Return if the free surface measured in this evaluation has reached the statically set
        equlibrium factor.

        :return: only true if the equilibrium factor of this measurment is higher than the
                 statically set factor.
        :rtype: boolean
        """
        return self.getFactorOfEquilibrium() > self.EQUILIBRIUM_CONSTANT
