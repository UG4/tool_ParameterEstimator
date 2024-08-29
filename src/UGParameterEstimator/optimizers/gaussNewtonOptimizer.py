"""This module contains the GaussNewtonOptimizer class, which is a subclass of Optimizer."""
from UGParameterEstimator import LineSearch, Result, setup_logger
import numpy as np
from .optimizer import Optimizer

newtonOptimizer_logger = setup_logger.logger.getChild("gaussNewtonOptimizer")

class GaussNewtonOptimizer(Optimizer):
    """This class implements the Gauss-Newton optimization algorithm."""
    def __init__(self,
                 linesearchmethod: LineSearch,
                 maxiterations=15,
                 epsilon=1e-3,
                 minreduction=1e-4,
                 differencing=Optimizer.Differencing.forward
                 ):
        super().__init__(epsilon, differencing)
        self.linesearchmethod = linesearchmethod
        self.maxiterations = maxiterations
        self.minreduction = minreduction

    def run(self, evaluator, initial_parameters, target, result=Result()):

        guess = initial_parameters

        evaluator.resultobj = result

        result.addRunMetadata("target", target)
        result.addRunMetadata("optimizertype", type(self).__name__)
        result.addRunMetadata("linesearchmethod", type(self.linesearchmethod).__name__)
        result.addRunMetadata("epsilon", self.finite_differencing_epsilon)
        result.addRunMetadata("differencing", self.differencing.value)
        result.addRunMetadata("fixedparameters", evaluator.fixedparameters)
        result.addRunMetadata("parametermanager", evaluator.parametermanager)

        result.log("-- Starting newton method. --")
        newtonOptimizer_logger.info("Starting newton method.")

        targetdata = target.getNumpyArray()

        # prepare weight vector in correct length if weight is given
        if len(evaluator.weight) > 0:
            weight_vector = np.ones(len(target.times))
            curr_weight_index = 0
            weight_vector[0] = evaluator.weight[curr_weight_index]
            try:
                for i in range(1, len(target.times)):
                    if target.times[i - 1] > target.times[i]:
                        curr_weight_index += 1
                    weight_vector[i] = evaluator.weight[curr_weight_index]
            except IndexError as exc:
                result.log("Error: Not enough weights given.")
                newtonOptimizer_logger.error("Error: Not enough weights given.")
                raise IndexError from exc

        last_S = -1
        first_S = -1

        for i in range(self.maxiterations):
            newtonOptimizer_logger.debug("Starting iteration %s with parameters %s", i, guess)

            jacobi_result = self.getJacobiMatrix(guess, evaluator, target, result)
            if jacobi_result is None:
                result.log("Error calculating Jacobi matrix, UG run did not finish")
                result.log(evaluator.getStatistics())
                result.save()
                newtonOptimizer_logger.error(
                    "Error calculating Jacobi matrix"
                    "UG run did not finish, iteration %s, parameters %s",
                    i, guess
                )
                newtonOptimizer_logger.info(evaluator.getStatistics())
                return

            V, measurement_evaluation = jacobi_result
            old_V = V
            # check if weight was given and apply it
            if len(evaluator.weight) > 0:
                old_V = V
                V = np.zeros((len(old_V), len(old_V[0])))
                for i, row in enumerate(old_V):
                    new_Vi = np.zeros(len(row))
                    for j, entry in enumerate(row):
                        new_Vi[j] = weight_vector[i] * entry
                    V[i] = new_Vi
            measurement = measurement_evaluation.getNumpyArrayLike(target)

            r = measurement - targetdata
            if len(evaluator.weight) > 0:
                r = weight_vector * r

            sigma = np.dot(np.transpose(old_V), old_V)

            S = 0.5 * r.dot(r)

            # save the residualnorm S for calculation of the relative reduction
            if first_S == -1:
                first_S = S

            n = len(targetdata)
            p = len(guess)
            dof = n - p

            # calculate s^2 = residual mean square / variance estimate (p.6 Bates/Watts)
            variance = None if dof == 0 else S / dof

            result.addMetric("residuals", r)
            result.addMetric("residualnorm", S)
            result.addMetric("parameters", guess)
            result.addMetric("jacobian", V)
            result.addMetric("variance", variance)
            result.addMetric("measurement", measurement)
            result.addMetric("measurementEvaluation", measurement_evaluation)
            result.addMetric("sigma", sigma)

            if last_S != -1:
                result.addMetric("reduction", S / last_S)

            result.log("[" + str(i) + "]: x=" + str(guess) + ", residual norm S=" + str(S))
            newtonOptimizer_logger.info("guess: %s, residual norm S: %s", guess, S)

            # calculate Gauss-Newton step direction (p. 40)
            Q1, R1 = np.linalg.qr(V, mode='reduced')
            w = Q1.transpose().dot(r)
            delta = -np.linalg.solve(R1, w)

            result.log("stepdirection is " + str(delta))
            newtonOptimizer_logger.info("stepdirection: %s", delta)

            # approximation of the hessian (X^T * X)^-1 = (R1^T * R1)^-1
            hessian = np.linalg.inv(np.matmul(np.transpose(R1), R1))
            covariance_matrix = variance * hessian

            result.addMetric("covariance", covariance_matrix)
            result.addMetric("hessian", hessian)
            newtonOptimizer_logger.info("covariance matrix: %s", covariance_matrix)
            newtonOptimizer_logger.info("hessian: %s", hessian)

            # construct correlation matrix (see p. 22 of Bates/Watts)
            R1inv = np.linalg.inv(R1)
            Dinv = np.diag(1 / np.sqrt(np.diag(hessian)))
            L = np.matmul(Dinv, R1inv)
            C = np.matmul(L, np.transpose(L))
            result.addMetric("correlation", C)
            newtonOptimizer_logger.info("correlation matrix: %s", C)

            # calculate standard error for the parameters (p.21)
            s = np.sqrt(variance)
            errors = s * np.linalg.norm(R1inv, axis=1)
            result.addMetric("errors", errors)
            newtonOptimizer_logger.info("errors: %s", errors)

            # cancel the optimization when the reduction of the norm of the residuals is below
            # the threshhold and the confidence of the calibrated parameters is sufficiently low
            if S / first_S < self.minreduction:
                result.log("-- Newton method converged. --")
                newtonOptimizer_logger.info("Newton method converged.")
                result.commitIteration()
                break

            # do linesearch in the gauss-newton search direction
            nextguess = self.linesearchmethod.doLineSearch(delta, guess, target, V, r, result)[0]

            if nextguess is None:
                result.log("-- Newton method did not converge. --")
                newtonOptimizer_logger.debug("No next guess found.")
                newtonOptimizer_logger.error("Newton method did not converge.")
                result.commitIteration()
                result.log(evaluator.getStatistics())
                newtonOptimizer_logger.info(evaluator.getStatistics())
                result.save()
                return result

            result.commitIteration()

            guess = nextguess
            last_S = S

        if i == self.maxiterations - 1:
            result.log("-- Newton method did not converge. --")
            newtonOptimizer_logger.debug("Max iterations reached with %s iterations.", i)
            newtonOptimizer_logger.error("Newton method did not converge.")

        result.log(evaluator.getStatistics())
        result.save()
        return result
