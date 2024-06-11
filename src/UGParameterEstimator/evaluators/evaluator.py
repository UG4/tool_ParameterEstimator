import numpy as np
import math
import os
from abc import ABC, abstractmethod
from UGParameterEstimator import ParameterManager, Evaluation, ParameterOutputAdapter, ErroredEvaluation, setup_logger

evaluator_logger = setup_logger.logger.getChild("evaluator")

class Evaluator(ABC):
    """Evaluator abstract base class

    Defines the interface for evaluators.
    Implements a cache to avoid unnecessary evaluations.

    """

    resultobj = None
    total_evaluation_count = 0
    serial_evaluation_count = 0
    cached_evaluation_count = 0
    cache = set()

    @property
    @abstractmethod
    def parallelism(self):
        """Returns the parallelism of the evaluator

        :return: parallelism of the evaluator
        :rtype:  int
        """
        pass

    @abstractmethod
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
        pass

    def setResultObject(self, res):
        """Sets the result object to write statistics to.

        :param res: resultobject to set
        :type res: Result
        """
        self.resultobj = res
    
    def handleNewEvaluations(self, evaluations, tag):
        """Updates the internal evaluation cache and writes evaluations
        and new caching statistics to the set result object.

        :param evaluations: evaluations to handle
        :type evaluations: list of Evaluation
        :param tag: tag to store the evaluations under in the result object
        :type tag: string
        """
        self.cache.update(evaluations)
        self.serial_evaluation_count += 1
        self.total_evaluation_count += len(evaluations)
        if self.resultobj is not None:
            self.resultobj.addEvaluations(evaluations, tag)
            self.resultobj.addRunMetadata("evaluator_totalcount", self.total_evaluation_count)
            self.resultobj.addRunMetadata("evaluator_serialcount", self.serial_evaluation_count)
            self.resultobj.addRunMetadata("evaluator_cachehits", self.cached_evaluation_count)

    def checkCache(self, parameters):
        """Checks the internal evaluation cache and returns the stored result, if
        there is one, or None, if not.

        :param parameters: parameters to check
        :type parameters: numpy array
        :return: Evaluation, if in cache, or None
        :rtype: Evaluation
        """
        for evaluation in self.cache:
            if evaluation.parameters is None:
                continue
            if np.array_equal(evaluation.parameters, parameters):
                if self.resultobj is not None:
                    self.resultobj.log("Served evaluation " + str(evaluation.eval_id) + " from cache!")
                self.cached_evaluation_count += 1
                return evaluation
        return None

    def reset(self):
        """resets the internal cache and statistics
        """
        self.cache = set()
        self.cached_evaluation_count = 0
        self.serial_evaluation_count = 0
        self.total_evaluation_count = 0


    def getStatistics(self):
        """returns the internal statistics as a string representation
        :return: string with statistics information
        :rtype: string
        """    
        string = "Total count of evaluations: " + str(self.total_evaluation_count) + "\n"
        string += "Taken from cache: " + str(self.cached_evaluation_count) + "\n"
        string += "Serial count: " + str(self.serial_evaluation_count)
        return string

    def __str__(self):
        string = "Currently cached Evaluations " + str(len(self.cache)) + "\n" 
        string += self.getStatistics()
        return string

    @classmethod
    def ConstructEvaluator(self, luafile, directory, parametermanager: ParameterManager, evaluation_type: Evaluation, parameter_output_adapter: ParameterOutputAdapter, fixedparameters={}, threadcount=10, cliparameters=[], ugsubmitparameters=[], weight=[]):
        """Factory method to construct a suitable evaluator.

        If UGSUBMIT can be detected on the system, a ClusterEvaluator will be used, if not, a LocalEvaluator.

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
        :param jobcount: optional maximum number of parallel jobs to submit in UGSUBMIT, defaults to 10
        :type jobcount: int, optional
        :param cliparameters: list of command line parameters to append to subprocess call. use separate entries
                for places that would normally require a space.
        :param weight: list of weights for each parameter
        :type cliparameters: list of strings, optional
        """
        import UGParameterEstimator
        if "UGSUBMIT_TYPE" in os.environ:
            print("Detected cluster " + os.environ["UGSUBMIT_TYPE"] + ", using ClusterEvaluator")
            evaluator_logger.debug("Detected cluster " + os.environ["UGSUBMIT_TYPE"] + ", using ClusterEvaluator")
            return UGParameterEstimator.ClusterEvaluator(
                luafilename=luafile,
                directory=directory,
                parametermanager=parametermanager,
                evaluation_type=evaluation_type,
                parameter_output_adapter=parameter_output_adapter,
                fixedparameters=fixedparameters,
                threadcount=threadcount,
                cliparameters=cliparameters,
                ugsubmitparameters=ugsubmitparameters,
                weight=weight)
        else:
            print("No cluster detected, using LocalEvaluator")
            evaluator_logger.debug("No cluster detected, using LocalEvaluator")
            return UGParameterEstimator.LocalEvaluator(luafile, directory, parametermanager, evaluation_type, parameter_output_adapter, fixedparameters, threadcount, cliparameters, weight=weight)
