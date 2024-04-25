from abc import ABC, abstractmethod

class Evaluation(ABC):
    """Base class for all Evaluation classes.

    Contains metadata which all Evaluations should have, and defines
    the 3 methods all evaluations should implement.
    """

    parameters = None
    eval_id = None
    runtime = None

    @abstractmethod
    def getNumpyArray(self):
        """Returns stored measurements as a 1d numpy array

        :return: stored measurements as a 1d numpy array
        :rtype: numpy array, 1d
        """
        pass

    @abstractmethod
    def getNumpyArrayLike(self, target):
        """Used to interpolate between different evaluations, when timestamps might differ because
        of the used time control schemes.

        :param target: Evaluation whichs format should be matched and interpolated to
        :type target: Evaluation
        :raises IncompatibleFormatError: When the two Evaluations can not be interpolated between
        :return: the data of this evaulation, interpolated to the targets format
        :rtype: numpy array
        """
        pass

    @classmethod
    @abstractmethod
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
        :return: Parsed Evaluation
        :rtype: Evaluation
        """
        pass

    class IncompatibleFormatError(Exception):
        pass

class ErroredEvaluation(Evaluation):
    """An Implementation of Evaluation indicating an error has occurred during Evaluation.

    The reason might be found in the reason field.
    """

    def __init__(self, parameters, reason, eval_id=None, runtime=None):
        """Class constructor

        :param parameters: the (transformed) parameters of this evaluation
        :type parameters: numpy array
        :param runtime: reason for the error
        :type runtime: string
        :param eval_id: id of the evaluation to find the correct file fron directory
        :type eval_id: int, optional
        :param runtime: runtime of the evaluation, in seconds
        :type runtime: int, optional
        """
        self.parameters = parameters
        self.eval_id = eval_id
        self.runtime = runtime
        self.reason = reason

    def getNumpyArray(self):
        pass

    def getNumpyArrayLike(self, target: Evaluation):
        pass

    @classmethod
    def parse(cls, directory, evaluation_id, parameters, eval_id, runtime):
        pass
