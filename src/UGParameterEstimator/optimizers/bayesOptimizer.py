
from .optimizer import Optimizer
from UGParameterEstimator import ParameterManager, Result, ErroredEvaluation
import numpy as np
import skopt

class BayesOptimizer(Optimizer):

    def __init__(self, parametermanager: ParameterManager, epsilon=1e-4, minreduction=1e-4, max_iterations=20):
        super().__init__(epsilon, Optimizer.Differencing.forward)
        self.parametermanager = parametermanager
        self.minreduction = minreduction
        self.max_iterations = max_iterations

    def run(self, evaluator, initial_parameters, target, result = Result()):

        evaluator.resultobj = result    

        result.addRunMetadata("target", target)
        result.addRunMetadata("optimizertype", type(self).__name__)
        result.addRunMetadata("fixedparameters", evaluator.fixedparameters)
        result.addRunMetadata("parametermanager", self.parametermanager)

        result.log("-- Starting bayesian optimization. --")

        targetdata = target.getNumpyArray()

        dimensions = []

        for p in self.parametermanager.parameters:

            if p.maximumValue is None:
                print("No maximum value for parameter " + p.name + " provided!") 
                exit(1)
            
            if p.minimumValue is None:
                print("No minimum value for parameter " + p.name + " provided!") 
                exit(1)

            print(p.name, p.minimumValue, p.maximumValue, p.optimizationSpaceLowerBound, p.optimizationSpaceUpperBound)
            dimensions.append(
                skopt.space.Real(p.optimizationSpaceLowerBound, p.optimizationSpaceUpperBound)
            )

        bayes_optimizer = skopt.Optimizer(
            dimensions,
            base_estimator="GP",
            n_initial_points=2,
            acq_func="EI",
            random_state=1
        )

        for iteration in range(self.max_iterations):

            needed_evaluations = bayes_optimizer.ask(evaluator.parallelism)

            result_evaluations = evaluator.evaluate(needed_evaluations, "bayes-opt")

            results = self.measurementToNumpyArrayConverter(result_evaluations, target)

            for ev in result_evaluations:
                if ev is None or isinstance(ev, ErroredEvaluation):                    
                    result.log("got a None-result! UG run did not finish or parameter was out of bounds...")                    
                    result.log(evaluator.getStatistics())
                    return

            Y = []

            min_S = None
            min_Index = None

            for i in range(len(results)):

                residual = results[i] - targetdata
                S = 0.5*residual.dot(residual)

                if min_S is None or min_S > S:
                    min_S = S
                    min_Index = i

                Y.append(S)

            print(Y)

            bayes_optimizer.tell(needed_evaluations, Y)
            
            result.addMetric("residualnorm", min_S)
            result.addMetric("parameters", needed_evaluations[min_Index])
            result.addMetric("measurement", results[min_Index])
            result.addMetric("measurementEvaluation", result_evaluations[min_Index])

            result.log("[" + str(iteration) + "]: best_param=" + str(needed_evaluations[min_Index]) + ", residual norm S=" + str(min_S))

            result.commitIteration()

        result.addRunMetadata("skopt-res", skopt.utils.create_result(bayes_optimizer.Xi, bayes_optimizer.yi, space=bayes_optimizer.space, models=bayes_optimizer.models))
        result.save()

        if(iteration == self.max_iterations-1):
            result.log("-- Bayesian optimization did not converge. --")
        
        result.log(evaluator.getStatistics())
        return result
    