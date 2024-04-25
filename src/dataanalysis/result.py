import os
import pickle
import copy
from scipy import stats
from math import floor, log10
from UGParameterEstimator import FreeSurfaceTimeDependentEvaluation, FreeSurfaceEquilibriumEvaluation
from datetime import datetime
import numpy as np

# helper functions to write numbers in scientific notation
def fexp(f):
    return int(floor(log10(abs(f)))) if f != 0 else 0

def fman(f):
    return f/10**fexp(f)

# A class containing the result of the calibration operation
#
# This class contains all logentries and all data written away during
# the iterations of the calibration.
# The data can be saved by calling save() and will be written to a .pkl file.
class Result:
    """This class saves all data created during the calibration process.
    This includes temporary results of the optimization algorithm (here called metrics,
    stored every iteration)
    logging entries, metadata (additional data stored for the whole optimization process)
    and evaluations.

    Arbitrary metrics and metadata fields can be added using the respecitive methods. they are
    stored a key-value-pairs.
    Only requirement is that they need to be
    `picklable <https://docs.python.org/3/library/pickle.html#what-can-be-pickled-and-unpickled>`_

    If a filename is specified, the results object will be saved whenever new data is available.

    :param filename: filename to save. if a path is specified, the directories will be created if
    not yet existant.
    :type filename: string, optional
    """

    def __init__(self,filename=None):
        """Constructor

        :param filename: filename to save. if a path is specified, the directories will be create 
        if not yet existant.
        :type filename: string, optional
        """
        self.iterations = []
        self.logentries = []

        # stores the current iterations metrics before it is committed to the iterations arry
        self.currentIteration = {}

        self.metadata = {}

        self.filename = filename

        if filename:
            directory = os.path.dirname(self.filename)
            if directory != "":
                os.makedirs(directory, exist_ok=True)

    @property
    def iterationCount(self):
        """Returns the number of iterations stored in this object

        :return: Number of iterations store din this object
        :rtype: Integer
        """

        return len(self.iterations)

    def writeTable(self, filename, metrics=[]):
        """writes the iteration data as a simple tab-separated-text file for postprocessing.
        Includes all parameter data for each iteration, and optionally metrics stored during the
        optimization.
        :param filename: the name of the file the result should be written to
        :type filename: string
        :param metrics: the metadata fields to include in the table as columns, additionally to the
        parameters
        :type metrics: Array containing tuples: First string the table header, second string the
        name of the metadata to store there
        """

        pm = self.metadata["parametermanager"]
        with open(filename,"w") as f:

            # write table header
            f.write("step" + "\t")
            for p in pm.parameters:
                f.write(p.name+"\t")
            for p in metrics:
                f.write(p[0]+"\t")

            f.write("\n")

            i = 0
            for iteration in self.iterations:
                f.write(str(i) + "\t")
                for j in range(len(pm.parameters)):
                    f.write(str(iteration["parameters"][j]) + "\t")

                for p in metrics:
                    if p[1] in iteration:
                        f.write(str(iteration[p[1]]) + "\t")
                    else:
                        f.write("NaN\t")

                f.write("\n")
                i += 1

    def writeLatexTable(self, filename, metrics=[], nameoverride=None):
        """writes the iteration data as a latex table.
        Includes all parameter data for each iteration, and optionally metrics stored during the
        optimization.
        :param filename: the name of the file the result should be written to
        :type filename: string
        :param metrics: the metrics fields to include in the table as columns, additionally to the
        parameters, optional
        :type metrics: Array containing tuples, optional: First string the table header, second
        string the name of the metrics to store there
        :param nameoverride: allows the user to provide better readable names as parameter names
        :type nameoverride: Array containing new parameter names, optional
        """

        pm = self.metadata["parametermanager"]
        with open(filename,"w") as f:

            # write table header
            f.write("\\begin{tabular}{")

            count = len(pm.parameters) + len(metrics)
            f.write("c||")
            for i in range(count-1):
                f.write("c|")
            f.write("c}\\\\\n")

            f.write("Schritt $l$" + " & ")
            for i, p in enumerate(pm.parameters):
                if nameoverride is None:
                    f.write("$\\hat{\\theta}_"+str(i+1)+"^{(l)}$ (\\verb|"+p.name+"|) & ")
                else:
                    f.write("$\\hat{\\theta}_"+str(i+1)+"^{(l)}$ (" + nameoverride[i] + ") & ")

            for i in range(len(metrics)-1):
                f.write(metrics[i][0]+"&")
            f.write(metrics[-1][0]+"\\\\\hline\n")

            i = 0
            for iteration in self.iterations:
                f.write(str(i+1) + " & ")
                for j, p in enumerate(pm.parameters):
                    entry = p.getTransformedParameter(iteration["parameters"][j])
                    f.write("$" + Result.getLatexString(entry) + "$" + " & ")

                for j, m in enumerate(metrics):
                    p = m[1]
                    if p in iteration:
                        f.write("$" + Result.getLatexString(iteration[p]) + "$" )
                    else:
                        f.write("--")
                    if j == len(metrics)-1:
                        f.write("\\\\\n")
                    else:
                        f.write(" & ")

                i += 1

            f.write("\\end{tabular}")

    def writeSimpleErrorTable(self, file, metrics, nameoverride=None):
        """writes the error data as a latex table.
        Includes all parameter data for each iteration, standard errors for the parameters, and
        optionally metadata stored during the optimization.
        Only can be used if the errors are stored in the results object (currently only by
        GaussNewtonOptimizer)
        :param filename: the name of the file the result should be written to
        :type filename: string
        :param metrics: the metrics fields to include in the table as columns, additionally to the
        parameters
        :type metrics: Array containing tuples: First string the table header, second string the
        name of the metadata to store there
        :param nameoverride: allows the user to provide better readable names as parameter names
        :type nameoverride: Array containing new parameter names, optional
        """

        pm = self.metadata["parametermanager"]
        with open(file,"w") as f:

            # write table header
            f.write("\\begin{tabular}{")

            count = 2*len(pm.parameters) + len(metrics)
            f.write("c||")
            for i in range(count-1):
                f.write("c|")
            f.write("c}\\\\\n")

            f.write("Schritt $l$" + " & ")
            for i, p in enumerate(pm.parameters):
                if nameoverride is None:
                    f.write("$\\hat{\\theta}_"+str(i+1)+"^{(l)}$ (\\verb|"+p.name+"|) & ")
                else:
                    f.write("$\\hat{\\theta}_"+str(i+1)+"^{(l)}$ (" + nameoverride[i] + ") & ")
                f.write("se($\\theta_" + str(i+1) + "^{(l)}$) & ")

            for i in range(len(metrics)-1):
                f.write(metrics[i][0]+"&")
            f.write(metrics[-1][0]+"\\\\\hline\n")

            i = 0
            for iteration in self.iterations:
                f.write(str(i+1) + " & ")
                for j, p in enumerate(pm.parameters):
                    entry = p.getTransformedParameter(iteration["parameters"][j])
                    error = iteration["errors"][j]
                    f.write("$" + Result.getLatexString(entry) + "$"+ " & ")
                    f.write("$" + Result.getLatexString(error) + "$ & ")

                for j, m in enumerate(metrics):
                    p = m[1]
                    if p in iteration:
                        f.write("$" + Result.getLatexString(iteration[p]) + "$" )
                    else:
                        f.write("--")
                    if j == len(metrics)-1:
                        f.write("\\\\\n")
                    else:
                        f.write(" & ")

                i += 1

            f.write("\\end{tabular}")


    def writeErrorTable(self, file):
        """writes the error data as a latex table.
        Includes all parameter data for each iteration, standard errors for the parameters, and an
        estimated confidence interval
        Only can be used if the errors are stored in the results object (currently only by
        GaussNewtonOptimizer)
        :param filename: the name of the file the result should be written to
        :type filename: string
        """
        pm = self.metadata["parametermanager"]
        with open(file,"w") as f:

            # write table header
            f.write("\\begin{tabular}{")

            count = len(pm.parameters) * 2
            f.write("c||")
            for i in range(count-1):
                f.write("c|")
            f.write("c}\n")

            f.write("step" + " & ")
            for i in range(len(pm.parameters)-1):
                p = pm.parameters[i].name
                f.write("$\\beta_"+str(i)+"$ ("+p+") & ")
                f.write("se($\\beta_" + str(i) + "$) & ")

            i = len(pm.parameters)-1
            f.write("$\\beta_"+str(i)+"$ (\\verb|"+pm.parameters[-1].name+"|) & ")
            f.write("se($\\beta_" + str(i) + "$) \\\\\hline\n")

            i = 0
            for iteration in self.iterations:
                f.write(str(i+1) + " & ")
                for j, p in enumerate(pm.parameters):
                    entry = p.getTransformedParameter(iteration["parameters"][j])
                    error = iteration["errors"][j]
                    if "confidenceinterval" in iteration:
                        interval = iteration["confidenceinterval"][j]
                        f.write("$"
                                + Result.getLatexString(entry)
                                + "\\pm" + Result.getLatexString(interval)
                                + "$"
                                + " & ")
                    else:
                        f.write("$" + Result.getLatexString(entry) + "$" + " & ")

                    f.write("$" + Result.getLatexString(error) + "$")

                    if j == len(pm.parameters)-1:
                        f.write("\\\\\n")
                    else:
                        f.write(" & ")

                i += 1

            f.write("\\end{tabular}")

    def writeMatrix(self, file, name, symbol, iterations_to_print=[-1]):
        """Writes an numpy matrix stored as iteration data to a file, formatted for direct use in
        latex using a pmatrix element. The matrix has to be saved by the optimizer. Can be used to
        print the matrix from multiple iterations.

        :param filename: the name of the file the result should be written to
        :type filename: string
        :param name: the name of the matrix, as stored with addMetric
        :type name: string
        :param symbol: the name of the matrix, as printed in latex. this can be any valid latex
            code. Will be subscripted by the iteration index
        :type symbol: string
        :param iterations_to_print: array of iterations to print this meteric for. If unspecified,
            will print the data fromthe last iteration.
        :type iterations_to_print: Array of Integers, optional
        """

        with open(file,"w") as f:
            for i in iterations_to_print:

                if i == -1:
                    i = self.iterationCount-1

                data = self.iterations[i][name]

                f.write("$$" + symbol + "^{(" + str(i+1) + ")} = \\begin{pmatrix}\n")

                for x in range(np.shape(data)[0]):
                    for y in range(np.shape(data)[1]):

                        f.write(Result.getLatexString(data[x][y]))

                        if y == np.shape(data)[1]-1:
                            f.write("\\\\\n")
                        else:
                            f.write("&")

                f.write("\\end{pmatrix}$$\n")

    def writeSensitivityPlots(self,
                              filename,
                              iteration,
                              averaged=True,
                              zlabel=lambda i:
                                  "$\\frac{\\partial \\vec{m}}{\delta \\theta_"
                                  + str(i+1)
                                  + "}$ {[m]}"):
        """prints a sensitivity plot in latex describing the sensitivity of the calibrated values
        to each parameter. This works if the target is a FreeSurfaceTimeDependentEvaluation or a
        FreeSurfaceEquilibriumEvaluation. Uses tikzpicture package in the generated latex code.

        :param filename: the name of the file the result should be written to
        :type filename: string
        :param iteration: the iteration the data will be used of. if -1 is specified, the data from
            the last iteration will be used.
        :type iteration: integer
        :param averaged: print averaged sensitivity over all timesteps and over all locations (for
            timedependent case). If false a 3d plot will be printed instead.
        :type averaged: bool, optional
        :param zlabel: label of the z axis, dependent on the current iteration. Has to be valid
            latex code.
        :type zlabel: function integer => string, optional
        """

        pm = self.metadata["parametermanager"]
        if iteration == -1:
            iteration = self.iterationCount-1

        if len(self.iterations) <= iteration or iteration < 0:
            print("Illegal iteration index")
            return

        iterationdata = self.iterations[iteration]
        jacobi = iterationdata["jacobian"]
        p = iterationdata["parameters"]
        m = iterationdata["measurement"]

        if len(pm.parameters) != jacobi.shape[1]:
            print("Mismatch of parameter count!")
            return

        for i, p in enumerate(pm.parameters):
            dg = jacobi[:,i]
            partial = (p[i]/(np.max(m)))*dg
            if isinstance(self.metadata["target"], FreeSurfaceTimeDependentEvaluation):
                partial_series = FreeSurfaceTimeDependentEvaluation.fromNumpyArray( partial, self.metadata["target"])

                if averaged:
                    partial_series.writeCSVAveragedOverTimesteps(filename + "-" + p.name + "-over-time.csv")
                    partial_series.writeCSVAveragedOverLocation(filename + "-" + p.name + "-over-location.csv")

                    with open(filename + "-" + p.name + ".tex","w") as f:
                        f.write("\\begin{center}\n")
                        f.write("\\begin{minipage}{0.4\\textwidth}\n")
                        f.write("\t\\begin{tikzpicture}[scale=0.8]\n")
                        f.write("	\\begin{axis}[\n")
                        f.write("	xlabel=Zeit,\n")
                        f.write("	ylabel=$\\frac{\\delta m}{\\delta \\theta_"+ str(i) + "}$,\n")
                        f.write("	legend style={\n")
                        f.write("		at={(0,0)},\n")
                        f.write("		anchor=north,at={(axis description cs:0.5,-0.18)}}]\n")
                        f.write("	\\addplot [thick] table [x={time}, y={value}] {"+filename + "-" + p.name + "-over-location.csv"+"};\n")
                        f.write("	\\end{axis}\n")
                        f.write("	\\end{tikzpicture}\n")
                        f.write("\\end{minipage}	 \n")
                        f.write("\\begin{minipage}{0.4\\textwidth}\n")
                        f.write("		\\begin{tikzpicture}[scale=0.8]\n")
                        f.write("		\\begin{axis}[\n")
                        f.write("		xlabel=Ort,\n")
                        f.write("		ylabel=$\\frac{\\delta m}{\delta \\beta_"+ str(i) + "}$,\n")
                        f.write("		legend style={\n")
                        f.write("			at={(0,0)},\n")
                        f.write("			anchor=north,at={(axis description cs:0.5,-0.18)}} ]\n")
                        f.write("		\\addplot [thick] table [x={location}, y={value}] {"+filename + "-" + p.name + "-over-time.csv};\n")
                        f.write("		\\end{axis}\n")
                        f.write("		\\end{tikzpicture}\n")
                        f.write("\\end{minipage}\\\\\n")  
                        f.write("\\end{center}")
                else:
                    partial_series.write3dPlot(filename + "-" + p.name.replace("_","-") + ".tex", zlabel(i), scale=0.8)
            elif isinstance(self.metadata["target"], FreeSurfaceEquilibriumEvaluation):
                partial_series = FreeSurfaceEquilibriumEvaluation.fromNumpyArray(partial, self.metadata["target"])
                FreeSurfaceEquilibriumEvaluation.writePlots({"Sensitivität":{"eval":partial_series}}, filename + "-" + p.name.replace("_","-") + ".tex", zlabel(i))

    def plotComparison(self, filename, force2d=False):
        """Saves a latex document plotting a comparison between target data, the evealuation using
        the initial parameters and the evaluation using the calibrated parameters. This works if the
        target is a FreeSurfaceTimeDependentEvaluation or a FreeSurfaceEquilibriumEvaluation.

        :param filename: the name of the file the result should be written to
        :type filename: string
        :param force2d: In timedependent case: Print the last evaluation for all 3 plots as a 2d
            plot. If false, 3 3d plots will be printed.
        :type force2d: bool, optional
        """

        target = self.metadata["target"]
        result = self.iterations[self.iterationCount-1]["measurementEvaluation"]
        start = self.iterations[0]["measurementEvaluation"]

        if isinstance(target, FreeSurfaceEquilibriumEvaluation):
            result = FreeSurfaceEquilibriumEvaluation.fromTimedependentTimeseries(result)
            start = FreeSurfaceEquilibriumEvaluation.fromTimedependentTimeseries(start)
            FreeSurfaceEquilibriumEvaluation.writePlots({"Nach Kalibrierung":{"eval":result}, "Kalibrierungsziel":{"eval":target, "dashed":True}, "Startparameter":{"eval":start}}, filename)
        else:
            if force2d:
                result = FreeSurfaceEquilibriumEvaluation.fromTimedependentTimeseries(result)
                start = FreeSurfaceEquilibriumEvaluation.fromTimedependentTimeseries(start)
                target = FreeSurfaceEquilibriumEvaluation.fromTimedependentTimeseries(target)
                FreeSurfaceEquilibriumEvaluation.writePlots({"Nach Kalibrierung":{"eval":result}, "Kalibrierungsziel":{"eval":target, "dashed":True}, "Startparameter":{"eval":start}}, filename)
            else:
                with open(filename,"w") as f:
                    f.write("\\begin{center}\n")
                    f.write("\\begin{minipage}{0.3\\textwidth}\n")
                    f.write(target.write3dPlot(None, scale=0.4))
                    f.write("\\\ntarget")
                    f.write("\\end{minipage}	 \n")
                    f.write("\\begin{minipage}{0.3\\textwidth}\n")
                    f.write(start.write3dPlot(None, scale=0.4))
                    f.write("\\\nstart")
                    f.write("\\end{minipage}	 \n")
                    f.write("\\begin{minipage}{0.3\\textwidth}\n")
                    f.write(result.write3dPlot(None, scale=0.4))
                    f.write("\\\nresult")
                    f.write("\\end{minipage}\\\\\n")
                    f.write("\\end{center}")

    @staticmethod
    def getLatexString(number):
        """Returns a number as a string formatted for usage in latex. This gives the scientific
            notation with 4 significant digits.

        :param number: the number to convert
        :type number: number
        :return: number formatted for usage in latex.
        :rtype: string
        """

        if number is None:
            return "--"
        exp = fexp(number)
        man = fman(number)

        man = round(man, 3)


        if -1 <= exp <= 1:
            return str(round(number,4))

        return str(man) + "\\cdot 10^{" + str(exp) + "} "

    def addRunMetadata(self, name, value):
        """Adds an object as metadata.

        :param name: the key for this metadata
        :type name: string
        :param value: value for this metadata key
        :type value: any picklable python type
        """
        self.metadata[name] = value

    def addEvaluations(self, evaluations, tag=None):
        """Adds evaluations to the current iteration.
        The evaluations can be tagged with an additional string for later analysis.

        :param evaluations: the evaluations to add
        :type evaluations: string
        :param tag: additional tag for this iteration
        :type tag: string
        """
        if "evaluations" not in self.currentIteration:
            self.currentIteration["evaluations"] = []
        self.currentIteration["evaluations"].append((copy.deepcopy(evaluations), tag, self.iterationCount))

    def addMetric(self, name, value):
        """Adds a metric to the current evaluation

        :param name: name of the metric to add
        :type name: string
        :param value: value for this metric key
        :type value: any picklable python type
        """
        self.currentIteration[name] = value

    def commitIteration(self):
        """Stores the current iteration to iterations array.
        If a filename was specified at construction, also saves the results object.
        """
        self.iterations.append(copy.deepcopy(self.currentIteration))
        self.currentIteration.clear()
        self.save()

    def save(self,filename=None):
        """Saves the results object in pickle format to a file.

        :param filename: filename to save to. if not specified, the filename set when constructing
            this object will be used.
        :type filename: string
        """
        if filename is None:
            filename = self.filename

        if filename is None:
            return

        with open(filename,"wb") as f:
            pickle.dump(self.__dict__,f)

    def log(self, text):
        """Adds an logentry.

        The logentry is printed in the process. Logs are addionally written to a separate file
        "<filename>_log" in plain text format to allow for easier debugging.
        
        :param text: logtext to add.
        :type text: string
        """
        logtext = "[" + str(datetime.now()) + "] " + text
        print(logtext)
        self.logentries.append(logtext)
        with open(self.filename + "_log","a") as f:
            f.write(logtext + "\n")

    def printlog(self):
        """Prints all logentries stored in the object."""
        for l in self.logentries:
            print(l)

    @classmethod
    def load(cls, filename, printInfo=True):
        """Loads a result object stored pickled in a file.

        :param filename: path to the file to load.
        :type filename: string
        :param printInfo: print information about the loaded results object (default: true)
        :type printInfo: bool, optional
        """
        result = cls()
        with open(filename, "rb") as f:
            result.__dict__.update(pickle.load(f))

        if printInfo:
            print(result)

        return result

    @staticmethod
    def plotMultipleRuns(resultnames, outputfilename, log=True, paramnames=None):
        """Saves a plot describing the optimization success of multiple different optimization runs
        with the same parameters, starting at different initial values. This uses the tikzpicture
        latex package in the generated code.

        :param resultnames: paths to the files to load.
        :type resultnames: array of strings
        :param outputfilename: filename to save to
        :type outputfilename: string
        :param log: use a logarithmic y axis
        :type log: bool, optional
        :param paramnames: optionally override the parameternames using this array of names
        :type paramnames: array of string, optional, size has to equal the number of parameters
        """

        with open(outputfilename, "w") as f:
            f.write("\t\\begin{tikzpicture}[scale=0.9]\n")
            f.write("	\\begin{axis}[\n")
            f.write("	xlabel=Iteration,\n")
            f.write("	ylabel=$f$,\n")

            if log:
                f.write("	ymode=log,\n")

            f.write("	legend style={\n")
            f.write("		at={(0,0)},\n")
            f.write("		anchor=north,at={(axis description cs:0.5,-0.3)}}]\n")

            for resultfilename in resultnames:
                result = Result.load(resultfilename)

                f.write("\t\t\\addplot+[thick]\n")
                f.write("\t\t table [x={it}, y={f}]{ \n")
                f.write("it\t f\n")

                for t in range(result.iterationCount):
                    f.write(str(t) + "\t" + str(result.iterations[t]["residualnorm"]) + "\n")
                f.write("};\n")

                legtext = ""
                pm = result.metadata["parametermanager"]
                paramcount = len(pm.parameters)
                for p in range(paramcount):
                    if paramnames is None:
                        legtext += "$\\theta^{(1)}_"+ str(p) + "=" + Result.getLatexString(pm.parameters[p].startvalue)+ "$"
                    else:
                        legtext += "$" + paramnames[p] + "^{(1)} =" + Result.getLatexString(pm.parameters[p].startvalue) + "$"
                    if p != paramcount-1:
                        legtext += ", "
                f.write("\\addlegendentry{" + legtext + "};\n")

            f.write("	\\end{axis}\n")
            f.write("	\\end{tikzpicture}\n")

    def __str__(self):
        res = "######################################################\n"
        res += "filename: " + self.filename + "\n"
        res += "iterationCount: " + str(self.iterationCount)+ "\n"
        res += "paramcount: " + str(len(self.metadata["parametermanager"].parameters)) + "\n"
        res += "first res norm: " + str(self.iterations[0]["residualnorm"]) + "\n"
        res += "last res norm: " + str(self.iterations[self.iterationCount-1]["residualnorm"]) + "\n"

        for k, v in self.metadata.items():
            res += f"{k}: {v}\n"

        res += "######################################################"

        return res
