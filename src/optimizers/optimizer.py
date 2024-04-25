#!/usr/bin/env python3

import subprocess
import os
import time
import numpy as np
from enum import Enum
from UGParameterEstimator import Result, LineSearch, Evaluator, ParameterManager, ErroredEvaluation
from abc import ABC, abstractmethod


class Optimizer(ABC):
    """A base class for all optimizers, defining the interface and common helper methods
    """

    Differencing = Enum("Differencing", "central forward pure_forward pure_central")

    def __init__(self, epsilon, differencing: Differencing):
        """Class constructor. Should be called by all classes implementing an optimizer.

        :param epsilon: The value of epsilon to use when doing finite differencing. If a value lower than 0 is supplied,
        a good guess (sqrt of machine precision) is used.
        :type epsilon: float
        :param differencing: the type of differencing to use to calculate the jacobi matrix
        :type differencing: Optimizer.Differencing
        """
        self.differencing = differencing
        self.finite_differencing_epsilon = epsilon

        if epsilon < 0:
            epsilon = np.sqrt(np.finfo(np.float).eps)

    def measurementToNumpyArrayConverter(self, evaluations, target):
        """Helper function to convert an array of Evaluation.
        Each evaluation will be converted and interpolated using it's
        getNumpyArrayLike method.
        None-values or Errors will be converted to None.

        :param evaluations: the evaluations to convert
        :type evaluations: list of Evaluations
        :param target: Evaluation describing the format/time steps
        each evaluation should be converted/interpolated to
        :type target: Evaluation
        :return: the results of the covnertions
        :rtype: list of numpy arrays
        """
        results = []
        for e in evaluations:
            if e is None or isinstance(e, ErroredEvaluation):
                results.append(None)
            else:
                results.append(e.getNumpyArrayLike(target))
        return results

    def getJacobiMatrix(self, point, evaluator, target, result):
        """Calculates the jacobi matrix in parallel using finite differencing.
        To do so, a number of jobs equal to the number of parameters will
        be passed to the given evaluator.
        As approximation the finite differencing with epsilon set via the
        class constructor will be used.

        :param point: The point in parameter space to calculate the jacobi matrix at
        :type point: numpy array
        :param evaluator: the evaluator to use
        :type evaluator: Evaluator
        :param target: the target of the calibration, needed only
         to convert all evaluations to the correct format
        :type target: Evaluation
        :param result: The result object to log to
        :type result:  Result
        :return: the jacobi matrix, and the evaluation at 'point'
        :rtype: tuple (numpy array, Evaluation)
        """
        jacobi = []

        neededevaluations = []
        neededevaluations.append(point)

        if self.differencing == Optimizer.Differencing.forward:
            for i in range(len(point)):
                changed = np.copy(point)
                if changed[i] == 0:
                    changed[i] = self.finite_differencing_epsilon
                else:
                    changed[i] *= 1 + self.finite_differencing_epsilon
                neededevaluations.append(changed)
        elif self.differencing == Optimizer.Differencing.pure_forward:
            for i in range(len(point)):
                changed = np.copy(point)
                changed[i] += self.finite_differencing_epsilon
                neededevaluations.append(changed)
        elif self.differencing == Optimizer.Differencing.central:
            for i, p in enumerate(point):
                changedPos = np.copy(point)
                changedNeg = np.copy(point)
                if p == 0:
                    changedPos[i] = self.finite_differencing_epsilon
                    changedNeg[i] = -self.finite_differencing_epsilon
                else:
                    changedNeg[i] *= 1 - self.finite_differencing_epsilon
                    changedPos[i] *= 1 + self.finite_differencing_epsilon
                neededevaluations.append(changedPos)
                neededevaluations.append(changedNeg)
        elif self.differencing == Optimizer.Differencing.pure_central:
            for i in range(len(point)):
                changedPos = np.copy(point)
                changedNeg = np.copy(point)
                changedNeg[i] -= self.finite_differencing_epsilon
                changedPos[i] += self.finite_differencing_epsilon
                neededevaluations.append(changedPos)
                neededevaluations.append(changedNeg)

        with evaluator:
            evaluations = evaluator.evaluate(neededevaluations, True, "jacobi-matrix")

        result.log("jacobi matrix calculated. evaluations:")

        for ev in evaluations:
            if isinstance(ev, ErroredEvaluation):
                result.log("\tid=" + str(ev.eval_id) + ", " + str(ev.reason))
            else:
                result.log("\tid=" + str(ev.eval_id) + ", timeCount=" + str(ev.timeCount))

        for ev in evaluations:
            if isinstance(ev, ErroredEvaluation):
                # At least one measurement failed
                return None

        # get the numpy arrays for the evaluation results
        results = self.measurementToNumpyArrayConverter(evaluations, target)  # len: c * n_v
        undisturbed = results[0]

        # calculate the jacobi matrix
        # point len: n_v
        for i, p in enumerate(point):
            if self.differencing == Optimizer.Differencing.forward:
                if p == 0:
                    column = (results[i + 1] - undisturbed) / (self.finite_differencing_epsilon)
                else:
                    column = (results[i + 1] - undisturbed) / (self.finite_differencing_epsilon * p)
            elif self.differencing == Optimizer.Differencing.pure_forward:
                column = (results[i + 1] - undisturbed) / (self.finite_differencing_epsilon)
            elif self.differencing == Optimizer.Differencing.central:
                if p == 0:
                    column = (results[2 * i + 1] - results[2 * i + 2]) / (2 * self.finite_differencing_epsilon)
                else:
                    column = (results[2 * i + 1] - results[2 * i + 2]) / (2 * self.finite_differencing_epsilon * p)
            elif self.differencing == Optimizer.Differencing.pure_central:
                column = (results[2 * i + 1] - results[2 * i + 2]) / (2 * self.finite_differencing_epsilon)
            jacobi.append(column)

        return (np.array(jacobi).transpose(), evaluations[0])

    @abstractmethod
    def run(self, evaluator, initial_parameters, target, result=Result()):
        """Runs this optimizer.

        :param evaluator: the evaluator to use for each Evaluation needed
        :type evaluator: Evaluator
        :param initial_parameters: The initial parameters to start the optimization from
        :type initial_parameters: numpy array
        :param target: The target of the calibration
        :type target: Evaluation
        :param result: Results object to write metadata and iterations to, defaults to Result()
        :type result: Result, optional
        """
        pass
