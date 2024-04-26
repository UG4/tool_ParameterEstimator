from .optimizer import Optimizer
from UGParameterEstimator import ParameterManager, Result, ErroredEvaluation
import numpy as np
import scipy

class ScipyNonlinearLeastSquaresOptimizer(Optimizer):

    def __init__(self, parametermanager: ParameterManager, epsilon=1e-3, differencing=Optimizer.Differencing.forward):
        super().__init__(epsilon, differencing)
        self.parametermanager = parametermanager

    def run(self, evaluator, initial_parameters, target, result = Result()):

        guess = initial_parameters
        
        evaluator.setResultObject(result)

        result.addRunMetadata("target", target)
        result.addRunMetadata("optimizertype", type(self).__name__)
        result.addRunMetadata("epsilon", self.finite_differencing_epsilon)
        result.addRunMetadata("differencing", self.differencing.value)
        result.addRunMetadata("fixedparameters", evaluator.fixedparameters)
        result.addRunMetadata("parametermanager", evaluator.parametermanager)

        result.log("-- Starting scipy optimization. --")

        targetdata = target.getNumpyArray()


        # assemble bounds
        bounds = ([],[])

        for p in self.parametermanager.parameters:

            if p.maximumValue is None:
                bounds[1].append(np.inf)
            else:
                bounds[1].append(p.optimizationSpaceUpperBound/(1+self.finite_differencing_epsilon))
            
            if p.minimumValue is None:
                bounds[0].append(-np.inf)
            else:
                bounds[0].append(p.optimizationSpaceLowerBound)


        # define the callbacks for scipy
        def scipy_fun(x):
            evaluation = evaluator.evaluate([x], True, "function-evaluation")[0]
            if isinstance(evaluation, ErroredEvaluation):
                result.log("Got a ErroredEvaluation: " + evaluation.reason)
                result.log(evaluator.getStatistics())
                return

            return evaluation.getNumpyArrayLike(target)-targetdata

        def jac_fun(x):
            jacobi_result = self.getJacobiMatrix(x, evaluator, target, result)
            if jacobi_result is None:
                result.log("Error calculating Jacobi matrix, UG run did not finish")
                result.log(evaluator.getStatistics())
                return

            V, measurementEvaluation = jacobi_result
            return V

        scipy_result = scipy.optimize.least_squares(scipy_fun, guess, jac=jac_fun, bounds=bounds)

        result.log("point is " + str(scipy_result.x))
        result.log("cost is " + str(scipy_result.cost))

        result.log(evaluator.getStatistics())
        result.save()

        return result

    

class ScipyMinimizeOptimizer(Optimizer):

    # opt_method must be one of "L-BFGS-B", "SLSQP" or "TNC"
    def __init__(self, parametermanager, opt_method="L-BFGS-B", epsilon=1e-4, callback_root=False, callback_scaling=1, differencing=Optimizer.Differencing.forward):
        super().__init__(epsilon, differencing)
        self.parametermanager = parametermanager
        self.opt_method = opt_method
        self.callback_root = callback_root
        self.callback_scaling = callback_scaling

    def run(self, evaluator, initial_parameters, target, result = Result()):

        guess = initial_parameters

        evaluator.setResultObject(result)

        result.addRunMetadata("target", target)
        result.addRunMetadata("optimizertype", type(self).__name__)
        result.addRunMetadata("epsilon", self.finite_differencing_epsilon)
        result.addRunMetadata("differencing", self.differencing.value)
        result.addRunMetadata("fixedparameters", evaluator.fixedparameters)
        result.addRunMetadata("parametermanager", self.parametermanager)

        result.log("-- Starting scipy optimization. --")

        targetdata = target.getNumpyArray()


        iteration_count = [0]
        last_S = [-1]


        # assemble bounds
        upper = []
        lower = []

        for p in self.parametermanager.parameters:
            
            if p.maximumValue is None:
                upper.append(np.inf)
            else:
                # this is needed to still have some space to do the finite differencing for the jacobi matrix
                upper.append(p.optimizationSpaceUpperBound/(1+self.finite_differencing_epsilon))
            
            if p.minimumValue is None:
                lower.append(-np.inf)
            else:
                lower.append(p.optimizationSpaceLowerBound)

        bounds = scipy.optimize.Bounds(lower, upper)

        # define the callbacks for scipy
        def scipy_function(x):
            result.log("\tEvaluating cost function at x=" + str(x))
            evaluation = evaluator.evaluate([x], True, "function-evaluation")[0]
            if isinstance(evaluation, ErroredEvaluation):
                result.log("Got a ErroredEvaluation: " + evaluation.reason)
                result.log(evaluator.getStatistics())
                result.save()
                exit()

            measurement = evaluation.getNumpyArrayLike(target)
            r = measurement-targetdata
            S = 0.5*r.dot(r)

            result.log("\t cost function is " + str(S))
            
            result.addMetric("parameters", x)
            result.addMetric("residualnorm",S)
            result.addMetric("measurement", measurement)
            result.addMetric("measurementEvaluation", evaluation)
            result.addMetric("residuals", r)

            if(last_S[0] != -1):
                result.addMetric("reduction", S/last_S[0])

            last_S[0] = S

            # https://stackoverflow.com/a/47443343

            if self.callback_root:
                return self.callback_scaling*np.sqrt(S)
            else:
                return self.callback_scaling*S

        def scipy_jacobi(x):
            result.log("\tEvaluating jacobi matrix at at x=" + str(x))
            jacobi_result = self.getJacobiMatrix(x, evaluator, target, result)
            if jacobi_result is None:
                result.log("Error calculating Jacobi matrix, UG run did not finish")
                result.log(evaluator.getStatistics())
                result.save()
                return

            V, measurementEvaluation = jacobi_result
            result.addMetric("jacobian", V)
            V = V.transpose()
            measurement = measurementEvaluation.getNumpyArrayLike(target)
            r = (measurement-targetdata)
            grad = V.dot(r)
            return grad

        def scipy_callback(xk):

            iteration_count[0] += 1

            result.log("[" + str(iteration_count[0]) + "]: parameters=" + str(xk))

            result.commitIteration()
            return False

        scipy_result = scipy.optimize.minimize( fun=scipy_function, x0=guess, jac=scipy_jacobi, 
                                                bounds=bounds, callback=scipy_callback, method=self.opt_method)

        result.log("result is " + str(scipy_result))

        result.log(evaluator.getStatistics())
        result.save()

        return result

