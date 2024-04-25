from .optimizer import Optimizer
from UGParameterEstimator import LineSearch, Result, ErroredEvaluation
import numpy as np

class GainedLevMarOptimizer(Optimizer):
        
    def __init__(self, maxiterations = 15, tau = 0.01, presteps = 5, epsilon=1e-3, minreduction=1e-4, initial_lam = None, differencing=Optimizer.Differencing.forward):
        super().__init__(epsilon, differencing)
        self.maxiterations = maxiterations
        self.minreduction = minreduction
        self.tau = tau
        self.presteps = presteps
        self.initial_lam = initial_lam

    def calculateDelta(self, V, r, p, lam):

        # calculate Lev-Mar step direction  (p.7)
        A = V.transpose().dot(V)
        g = V.transpose().dot(r)
                
        M = A + lam*np.diag(np.ones(p))
        Q, R = np.linalg.qr(M)
        w = Q.transpose().dot(g)
        delta = -np.linalg.solve(R, w)
        return delta

    def calculateGainRatio(self, S, newS, delta, lam, grad):
        denum = 0.5*delta.transpose().dot(lam*delta-grad)
        num = S-newS

        return num/denum

    def run(self, evaluator, initial_parameters, target, result = Result()):

        guess = initial_parameters

        evaluator.resultobj = result    

        result.addRunMetadata("target", target)
        result.addRunMetadata("optimizertype", type(self).__name__)
        result.addRunMetadata("epsilon", self.finite_differencing_epsilon)
        result.addRunMetadata("differencing", self.differencing.value)
        result.addRunMetadata("tau", self.tau)
        result.addRunMetadata("fixedparameters", evaluator.fixedparameters)
        result.addRunMetadata("parametermanager", evaluator.parametermanager)

        result.log("-- Starting Gained Levenberg-Marquardt method. --")

        targetdata = target.getNumpyArray()

        first_S = -1
        lam = -1
        nu = 2

        for i in range(self.maxiterations):

            jacobi_result = self.getJacobiMatrix(guess, evaluator, target, result)
            if jacobi_result is None:
                result.log("Error calculating Jacobi matrix, UG run did not finish")
                result.log(evaluator.getStatistics())
                result.save()
                return

            V, measurementEvaluation = jacobi_result
            measurement = measurementEvaluation.getNumpyArrayLike(target)

            r = measurement-targetdata

            S = 0.5*r.dot(r)

            # save the residualnorm S for calculation of the relative reduction
            if i == 0:
                first_S = S
                H = V.transpose().dot(V)
                lam = self.tau * np.max(np.diag(H))
                if self.initial_lam is not None:
                    lam = self.initial_lam

            n = len(targetdata)
            p = len(guess)
            dof = n-p

            # calculate s^2 = residual mean square / variance estimate (p.6 Bates/Watts)

            variance = None if dof == 0 else S/dof

            result.addMetric("residuals",r)
            result.addMetric("residualnorm",S)
            result.addMetric("parameters",guess)
            result.addMetric("jacobian", V)
            result.addMetric("variance", variance)
            result.addMetric("measurement", measurement)
            result.addMetric("measurementEvaluation", measurementEvaluation)

            result.log("[" + str(i) + "]: x=" + str(guess) + ", residual norm S=" + str(S) + ", lambda=" + str(lam))
          
            # cancel the optimization when the reduction of the norm of the residuals is below the threshhold
            if (S/first_S < self.minreduction):
                result.log("-- Gained Levenberg-Marquardt method converged. --")
                result.commitIteration()
                break

            lambdas = [lam]
            nus = [nu]
            deltas = [self.calculateDelta(V, r, p, lam)]
            points = [guess + deltas[-1]]

            for z in range(self.presteps):
                new_nu = nus[-1]*2
                nus.append(new_nu)
                new_lam = lambdas[-1]*nus[-1]
                lambdas.append(new_lam)
                deltas.append(self.calculateDelta(V, r, p, new_lam))
                points.append(guess + deltas[-1])

            evals = evaluator.evaluate(points)
            evalvecs = self.measurementToNumpyArrayConverter(evals, target)

            g = V.transpose().dot(r)

            costs = [None if x is None else 0.5*(x-targetdata).dot(x-targetdata) for x in evalvecs]
            gainratios = [self.calculateGainRatio(S, costs[i], deltas[i], lambdas[i], g) for i in range(self.presteps+1)]

            for z in range(self.presteps+1):
                if isinstance(evals[z], ErroredEvaluation):
                    result.log("\t lam=" + str(lambdas[z]) + ", nu=" + str(nus[z]) + ": " + evals[z].reason)
                else:
                    result.log("\t lam=" + str(lambdas[z]) + ", nu=" + str(nus[z]) + ": f=" + str(costs[z]) + ", new gainration=" + str(gainratios[z]))


            for z in range(self.presteps+1):
                if gainratios[z] > 0:   # step acceptable
                    new_S = costs[z]
                    nu = 2
                    lam = lambdas[z]*max(1/3, 1-(2*gainratios[z]-1)**3)
                    nextguess = points[z]
                    break
            else:
                result.log("-- Gained Levenberg-Marquardt method did not converge. Increase presteps. --")
                result.commitIteration()
                result.log(evaluator.getStatistics())
                result.save()
                return result


            result.addMetric("lambda", lam)
            result.addMetric("nu", nu)
            result.log("["+str(i) + "] new lam = " + str(lam) + " with f=" + str(new_S) + ", new nu = " + str(nu))
            
            result.addMetric("residualnorm_new", new_S)
            result.addMetric("reduction", new_S/S)

            
                        
            result.commitIteration()

            guess = nextguess

        if(i == self.maxiterations-1):
            result.log("-- Gained Levenberg-Marquardt method did not converge. --")
        
        result.log(evaluator.getStatistics())
        result.save()
        return result