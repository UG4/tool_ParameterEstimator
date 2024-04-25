from UGParameterEstimator.optimizers import Optimizer
from UGParameterEstimator import Result
import numpy as np

class MMAOptimizer(Optimizer):

    def __init__(self, maximum, minimum, minreduction=1e-4, max_iterations=15, epsilon=1e-3, differencing=Optimizer.Differencing.forward):
        super().__init__(epsilon, differencing)
        self.maximum = maximum
        self.minimum = minimum
        self.minreduction = minreduction
        self.max_iterations = max_iterations
        
    def run(self, evaluator, initial_parameters, target, result = Result()):

        evaluator.resultobj = result    

        result.addRunMetadata("target", target)
        result.addRunMetadata("optimizertype", type(self).__name__)
        result.addRunMetadata("epsilon", self.finite_differencing_epsilon)
        result.addRunMetadata("differencing", self.differencing.value)
        result.addRunMetadata("fixedparameters", self.evaluator.fixedparameters)
        result.addRunMetadata("parametermanager", self.evaluator.parametermanager)

        result.log("-- Starting newton method. --")

        targetdata = target.getNumpyArray()

        first_S = -1

        x = initial_parameters
        n = len(x)

        U = x + (self.maximum - self.minimum)
        L = x - (self.maximum - self.minimum)

        for iteration in range(self.max_iterations):
            
            print("x=" + str(x))
            print("U=" + str(U))
            print("L=" + str(L))

            jacobi_result = self.getJacobiMatrix(x, evaluator, target, result)
            if jacobi_result is None:
                result.log("Error calculating Jacobi matrix, UG run did not finish")
                return

            J, measurement_evaluation = jacobi_result
            measurement = measurement_evaluation.getNumpyArrayLike(target)

            residual = measurement - targetdata
            S = np.linalg.norm(residual)
            
            # save the residualnorm S for calculation of the relative reduction
            if first_S == -1:
                first_S = S

            # calculate the gradient at the current point
            # this is since f = 1/2 \sum r^2, grad = J^T r
            grad = J.transpose().dot(measurement)  
            print("grad=" + str(grad))

            result.addMetric("residuals",residual)
            result.addMetric("residualnorm",S)
            result.addMetric("parameters",x)
            result.addMetric("jacobian", J)
            result.addMetric("measurement", measurement)
            result.addMetric("measurementEvaluation", measurement_evaluation)

            result.log("\t [" + str(iteration) + "]: Residual norm S=" + str(S))

            p = np.zeros_like(x)
            q = np.zeros_like(x)
            r = residual
            next_x = np.zeros_like(x)
            alpha = np.zeros_like(x)
            beta = np.zeros_like(x)

            l_deriv = lambda arg,j: (p[j] / np.square(U[j]-arg))-(q[j]/np.square(arg-L[j]))

            for i in range(n):
                if grad[i] > 0:
                    p[i] = np.square(U[i] - x[i]) * grad[i]
                elif grad[i] < 0:
                    q[i] = - np.square(x[i]-L[i]) * grad[i] 

                r -= p[i] / (U[i]-x[i])
                r -= q[i] / (x[i]-L[i])

                alpha[i] = max(self.minimum[i], 0.9*L[i]+0.1*x[i])
                beta[i] = min(self.maximum[i], 0.9*U[i]+0.1*x[i])

                if l_deriv(alpha[i],i) >= 0:
                    next_x[i] = alpha[i]
                elif l_deriv(beta[i], i) <= 0:
                    next_x[i] = beta[i]
                elif l_deriv(alpha[i],i) < 0 and l_deriv(beta[i],i) > 0:
                    next_x[i] = (np.sqrt(p[i])*L[i]+np.sqrt(q[i])*U[i])/(np.square(p[i])+np.square(q[i]))
                else:
                    next_x[i] = x[i]
                    print("l_deriv strange")

            if(S/first_S < self.minreduction):
                result.log("-- MMA converged. --")
                result.commitIteration()
                break

            result.commitIteration()

            last_S = S
            x = next_x

        if(iteration == self.max_iterations-1):
            result.log("-- MMA did not converge. --")

        return result
    