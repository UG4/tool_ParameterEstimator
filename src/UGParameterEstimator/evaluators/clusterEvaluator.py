import subprocess
import os
import io
import time
import csv
from shutil import copyfile
from UGParameterEstimator import ParameterManager, Evaluation, ParameterOutputAdapter, ErroredEvaluation, setup_logger
from .evaluator import Evaluator

cluster_logger = setup_logger.logger.getChild("clusterEvaluator")

class ClusterEvaluator(Evaluator):
    """Evaluator for Clusters supporting UGSUBMIT.

    Implements the Evaluator AbcstractBaseClass.
    Calls UGSUBMIT and UGINFO using the subprocess module to schedule tasks and get infos about them.

    Implements a handler to catch unexpected program interruption while evaluating (for example, if the user
    cancels the operation). In this case, all open jobs will be cancelled.

    Output of UG4 is redirected into a separate <id>_ug_output.txt file.

    """
    def __init__(self, luafilename, directory, parametermanager: ParameterManager, evaluation_type, parameter_output_adapter: ParameterOutputAdapter, fixedparameters={}, threadcount=10, cliparameters=[], ugsubmitparameters=[], weight=[]):
        """Class constructor

        :param luafilename: path to the luafile to call for every evaluation
        :type luafilename: string
        :param directory: directory to use for exchanging data with UG4
        :type directory: string
        :param parametermanager: ParameterManager to transform the parameters/get parameter information
        :type parametermanager: ParameterManager
        :param evaluation_type: TYPE the evaluation shoould be parsed as.
        :type evaluation_type: type implementing Evaluation
        :param parameter_output_adapter: output adapter to write the parameters
        :type parameter_output_adapter: ParameterOutputAdapter
        :param fixedparameters: optional dictionary of fixed parameters to pass
        :type fixedparameters: dictionary<string, string|number>, optional
        :param threadcount: optional maximum number of threads per job in UGSUBMIT, defaults to 10
        :type threadcount: int, optional
        :param cliparameters: list of command line parameters to append to subprocess call. use separate entries
                for places that would normally require a space.
        :param weight: list of weights for each parameter
        :type cliparameters: list of strings, optional
        """
        self.directory = directory
        self.parametermanager = parametermanager
        self.id = 0
        self.fixedparameters = {"output": 0}
        self.fixedparameters.update(fixedparameters)
        self.evaluation_type = evaluation_type
        self.parameter_output_adapter = parameter_output_adapter
        self.threadcount = threadcount
        self.jobids = []
        self.luafilename = luafilename
        self.cliparameters = cliparameters
        self.ugsubmitparameters = ugsubmitparameters
        self.weight = weight

        if not os.path.exists(self.directory):
            os.mkdir(self.directory)

        filelist = list(os.listdir(self.directory))
        for f in filelist:
            os.remove(os.path.join(self.directory, f))

    @property
    def parallelism(self):
        """Returns the parallelism of the evaluator

        :return: parallelism of the evaluator
        :rtype:  int
        """
        return self.threadcount

    def evaluate(self, evaluationlist, transform=True, tag=""):
        """Evaluates the parameters given in evaluationlist using UG4, and the adapters set in the constructor.

        :param evaluationlist: parametersets to evaluate
        :type evaluationlist: list of numpy arrays
        :param transform: wether to transform the parameters with parametermanager set in this object, defaults to true
        :type transform: boolean, optional
        :param tag: tag-string attached to all produced evaluations for analysis purposes
        :type tag: string
        :return: list of parsed evaluation objects with the type given in the constructor, or ErroredEvaluation
        :rtype: list of Evaluation
        """
        results = [None] * len(evaluationlist)
        self.jobids = [None] * len(evaluationlist)
        evaluationids = [None] * len(evaluationlist)
        beta = [None] * len(evaluationlist)
        starttimes = [None] * len(evaluationlist)

        for j, beta_j in enumerate(evaluationlist):
            if transform:
                beta[j] = self.parametermanager.getTransformedParameters(beta_j)
                if beta[j] is None:
                    results[j] = ErroredEvaluation(None, reason="Infeasible parameters")
            else:
                beta[j] = beta_j

            if results[j] is None:
                results[j] = self.checkCache(beta[j])

        for j in range(len(evaluationlist)):

            if results[j] is not None:
                continue

            starttimes[j] = time.time()

            absolute_directory_path = os.getcwd() + "/" + self.directory
            absolute_script_path = os.getcwd() + "/" + self.luafilename

            if not os.path.isfile(absolute_script_path):
                cluster_logger.error(f"Luafile not found! {absolute_script_path}")
                exit()
            if not os.path.exists(absolute_directory_path):
                cluster_logger.error(f"Exchange directory not found! {absolute_directory_path}")
                exit()

            callParameters = ["ugsubmit", str(self.threadcount)]

            callParameters += self.ugsubmitparameters

            callParameters += ["---", "ugshell", "-ex", absolute_script_path, "-evaluationId", str(self.id), "-communicationDir", absolute_directory_path]

            callParameters += self.cliparameters

            evaluationids[j] = self.id

            # output the parameters however needed for the application
            self.parameter_output_adapter.writeParameters(self.directory, self.id, self.parametermanager, beta[j], self.fixedparameters)

            self.id += 1

            # submit the job and parse the received id
            cluster_logger.debug(f"Starting process {j} with command: {callParameters}")
            process = subprocess.Popen(callParameters, stdout=subprocess.PIPE)
            proc_id = process.pid
            process.wait()

            cluster_logger.debug(f"Job id with process.pid: {proc_id}")

            for line in io.TextIOWrapper(process.stdout, encoding="UTF-8"):
                if line.startswith("Received job id"):
                    cluster_logger.debug(line)
                    try:
                        self.jobids[j] = int(line.split(" ")[3])
                        cluster_logger.debug(f"Job id from ugsubmit: {self.jobids[j]}")
                    except ValueError:
                        cluster_logger.warning("Error parsing job id!")
                        cluster_logger.debug(f"direct process id from 'process.pid': {process.pid}")
                        cluster_logger.debug(f"line from process.stdout: {line} ")
                        cluster_logger.warning("Tmp-Fix: taking direct process id as job id\n")
                        self.jobids[j] = proc_id


            if self.jobids[j] is None:
                cluster_logger.warning("Job id from ugsubmit is None! Taking direct process id as job id")
                self.jobids[j] = proc_id

            # to avoid bugs with the used scheduler on cesari
            time.sleep(1)

        while True:

            # wait until all jobs are finished
            # for this, call uginfo and parse the output
            cluster_logger.debug("Waiting for jobs to finish...")
            process = subprocess.Popen(["uginfo"], stdout=subprocess.PIPE)
            process.wait()
            lines = io.TextIOWrapper(process.stdout, encoding="UTF-8").readlines()
            while True:
                if "JOBID" not in lines[0]:
                    lines.remove(lines[0])
                else:
                    break

            reader = csv.DictReader(lines, delimiter=" ", skipinitialspace=True)

            # are all of our jobs finished?
            finished = True

            cluster_logger.debug(f"TMP: iteration over uginfo output; lines: {lines}")

            for row in reader:
                jobid = int(row["JOBID"])
                if (jobid in self.jobids) and (row["STATE"] == "RUNNING" or row["STATE"] == "PENDING"):
                    cluster_logger.debug(f"Job {jobid} is still running.")
                    finished = False
                    break

            if finished:
                cluster_logger.debug("All jobs finished.")
                break

            time.sleep(30)  # default: 5

        cluster_logger.debug(f"TMP: iteration over evaluationlist; evaluationlist: {evaluationlist}")
        cluster_logger.debug(f"TMP: iteration over evaluationlist: len(evaluationlist): {len(evaluationlist)}")

        # now we can parse the measurement files
        for i in range(len(evaluationlist)):

            if results[i] is not None:
                continue

            # parse the result
            data = self.evaluation_type.parse(self.directory, evaluationids[i], beta[i], time.time() - starttimes[i])

            # preserve the association between the ugoutput and th einternal avaluation id.
            # this allows for better debugging
            stdoutfile = os.path.join(self.directory, str(evaluationids[i]) + "_ug_output.txt")
            copyfile("jobid." + str(self.jobids[i]) + "/job.output", stdoutfile)

            results[i] = data

        self.handleNewEvaluations(results, tag)
        self.jobids.clear()

        return results

    def __enter__(self):
        pass

    def __exit__(self, type, value, traceback):

        # make sure all (of our) jobs are cancelled or finished when the evaluation is finished

        cluster_logger.info("Got exit signal, cancelling jobs.")
        cluster_logger.debug(f"type: {type}")
        cluster_logger.debug(f"value: {value}")
        cluster_logger.debug(f"traceback: {traceback}")

        if not self.jobids:
            return None

        # call uginfo to find out which jobs are still running
        process = subprocess.Popen(["uginfo"], stdout=subprocess.PIPE)
        process.wait()
        lines = io.TextIOWrapper(process.stdout, encoding="UTF-8").readlines()
        while True:
            if "JOBID" not in lines[0]:
                lines.remove(lines[0])
            else:
                break

        reader = csv.DictReader(lines, delimiter=" ", skipinitialspace=True)

        cluster_logger.debug(f"TMP: iteration over reader in exit; lines: {lines}")
        for row in reader:
            jobid = int(row["JOBID"])
            if jobid in self.jobids:
                print("Cancelling " + str(jobid))
                cluster_logger.info(f"Cancelling job {jobid} in exit function")

                # cancel them using ugcancel
                process2 = subprocess.Popen(["ugcancel", str(jobid)], stdout=subprocess.PIPE)
                process2.wait()
