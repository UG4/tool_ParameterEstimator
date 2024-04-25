from .optimizer import Optimizer
from UGParameterEstimator import LineSearch, Result
import numpy as np

class LevMarOptimizer(Optimizer):
        
    def __init__(self, maxiterations = 15, initial_lam = 0.01, nu=10, P=10, P_iteration_count=3,scaling=False, epsilon=1e-3, minreduction=1e-4, differencing=Optimizer.Differencing.forward):
        super().__init__(epsilon, differencing)
        self.maxiterations = maxiterations
        self.minreduction = minreduction
        self.nu = nu
        self.initial_lam = initial_lam
        self.scaling = scaling
        self.P = P
        self.P_iteration_count = P_iteration_count

    def calculateDelta(self, V, r, p, lam):

        scaling = self.scaling

        # calculate Lev-Mar step direction  (p.7)
        A = V.transpose().dot(V)
        g = V.transpose().dot(r)
        
        AStar = np.copy(A)
        gStar = np.copy(g)

        if scaling:
            for x in range(p):
                for y in range(p):
                    AStar[x,y] = A[x,y] / (np.sqrt(A[x,x])*np.sqrt(A[y,y]))
                gStar[x] = g[x] / np.sqrt(A[x,x])
        
        M = AStar + lam*np.diag(np.ones(p))
        Q,R = np.linalg.qr(M)
        w = Q.transpose().dot(g)
        deltaStar = -np.linalg.solve(R, w)
        delta = np.copy(deltaStar)

        if scaling:
            for x in range(p):
                delta[x] = deltaStar[x] / np.sqrt(A[x,x])

        return delta

    def run(self, evaluator, initial_parameters, target, result = Result()):

        guess = initial_parameters

        evaluator.resultobj = result    

        result.addRunMetadata("target", target)
        result.addRunMetadata("optimizertype", type(self).__name__)
        result.addRunMetadata("epsilon", self.finite_differencing_epsilon)
        result.addRunMetadata("differencing", self.differencing.value)
        result.addRunMetadata("lambda_init", self.initial_lam)
        result.addRunMetadata("nu", self.nu)
        result.addRunMetadata("fixedparameters", evaluator.fixedparameters)
        result.addRunMetadata("parametermanager", evaluator.parametermanager)

        result.log("-- Starting Levenberg-Marquardt method. --")

        targetdata = target.getNumpyArray()

        first_S = -1
        lam = self.initial_lam

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
            if first_S == -1:
                first_S = S

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
                result.log("-- Levenberg-Marquardt method converged. --")
                result.commitIteration()
                break
            
            delta_lower_lam = self.calculateDelta(V,r,p,lam/self.nu)
            delta_prev_lam = self.calculateDelta(V,r,p,lam)
            delta_higher_lam = self.calculateDelta(V,r,p,lam*self.nu)

            evals = evaluator.evaluate([guess+delta_lower_lam, guess+delta_prev_lam, guess+delta_higher_lam])
            evalvecs = self.measurementToNumpyArrayConverter(evals, target)

            S_lower_lam = None if evalvecs[0] is None else 0.5*(evalvecs[0]-targetdata).dot(evalvecs[0]-targetdata)
            S_prev_lam = None if evalvecs[1] is None else 0.5*(evalvecs[1]-targetdata).dot(evalvecs[1]-targetdata)           
            S_higher_lam = None if evalvecs[2] is None else 0.5*(evalvecs[2]-targetdata).dot(evalvecs[2]-targetdata)


            found = False
            if S_lower_lam is None:
                result.log("\t lam = " + str(lam/self.nu) + ": " + evals[0].reason)
            else:
                result.log("\t lam = " + str(lam/self.nu) + ": f=" + str(S_lower_lam))

            if S_prev_lam is None:
                result.log("\t lam = " + str(lam) + ": " + evals[1].reason)
            else:
                result.log("\t lam = " + str(lam) + ": f=" + str(S_prev_lam))

            if S_higher_lam is None:
                result.log("\t lam = " + str(lam*self.nu) + ": " + evals[2].reason)
            else:  
                result.log("\t lam = " + str(lam*self.nu) + ": f=" + str(S_higher_lam))

            if S_lower_lam is not None and S_lower_lam <= S:
                lam = lam/self.nu
                new_S = S_lower_lam
                nextguess = guess+delta_lower_lam
            elif S_prev_lam is not None and S_prev_lam <= S:
                new_S = S_prev_lam
                nextguess = guess+delta_prev_lam
            elif S_higher_lam is not None and S_higher_lam < S:
                lam = lam*self.nu
                new_S = S_higher_lam
                nextguess = guess+delta_higher_lam
            else:
                for zl in range(self.P_iteration_count):
                    points = []
                    for z in range(self.P):
                        new_lam = lam*self.nu**(zl*self.P+z)
                        delta = self.calculateDelta(V,r,p,new_lam)
                        points.append(guess + delta)

                    evals = evaluator.evaluate(points)
                    evalvecs = self.measurementToNumpyArrayConverter(evals, target)

                    costs = [None if x is None else 0.5*(x-targetdata).dot(x-targetdata) for x in evalvecs]

                    for z in range(self.P):
                        if costs[z] is None:
                            result.log("\t lam = " + str(lam*self.nu**z) + ": " + evals[z].reason)
                        else:
                            result.log("\t lam = " + str(lam*self.nu**z) + ": f=" + str(costs[z]))
                    for z in range(self.P):
                        if costs[z] is not None and costs[z] < S:
                            lam = lam*self.nu**z
                            new_S = costs[z]
                            nextguess = points[z]
                            found = True
                    if found:
                        break
                if not found:
                    result.log("-- Levenberg-Marquardt method did not converge. --")
                    result.commitIteration()
                    result.log(evaluator.getStatistics())
                    result.save()
                    return result

            result.log("["+str(i) + "] best lam was = " + str(lam) + " with f=" + str(new_S))
            
            result.addMetric("lambda", lam)
            result.addMetric("residualnorm_new", new_S)
            result.addMetric("reduction", new_S/S)

           
                        
            result.commitIteration()

            guess = nextguess

        if(i == self.maxiterations-1):
            result.log("-- Levenberg-Marquardt method did not converge. --")
        
        result.log(evaluator.getStatistics())
        result.save()
        return result