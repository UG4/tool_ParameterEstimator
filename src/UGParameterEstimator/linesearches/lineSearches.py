import numpy as np
import math
from abc import ABC, abstractmethod
from UGParameterEstimator import ErroredEvaluation

class LineSearch(ABC):
    """Base class for all line searches describing the interface and 
    providing a helper function
    """

    def __init__(self, evaluator):
        """Class constructor setting the evaluator to use

        :param evaluator: Evaluator to evaluate the target function
        :type evaluator: Evaluator
        """
        self.evaluator = evaluator
    
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

    @abstractmethod
    def doLineSearch(self, stepdirection, guess, target, J, r, result):
        """Executes the line search along a given direction

        :param stepdirection: the direction to search
        :type stepdirection: numpy array in parameter space
        :param guess: the initial guess to do the step from
        :type guess: numpy array in parameter space
        :param target: the target of the calibration process
        :type target: Evaluation
        :param J: the jacobian of the cost function in relation to the parameters
        :type J: numpy array PxP
        :param r: the last residual vector
        :type r: numpy array in measurement space
        :param result: a tuple of final best guess, and overall 
        lowest residual value. Or (None, None), if an error occurred.
        :type result: tuple (numpy array, scalar)
        """
        pass

class LinearParallelLineSearch(LineSearch):
    """A line search executing multiple, uniformly distributed
    line search steps in parallel.

    Multiple iterations, controlled by max_iterations, are
    performed sequentially, each with a smaller search window 
    around the smallest window found in the last iteration.

    If a suitable step size was found, as defined by the first wolfe condition with c = 1e-3,
    the search is stopped IF this value was not at the bounds of the search window.
    If it is, the search is continued in hope of a better step size.
    """

    c = 1e-3                    # c for the wolfe lower bound  
    max_iterations = 1         # number of maximum iterations of the line search
    parallel_evaluations = 10   # number of parallel evaluations during the parallel line search

    def __init__(self, evaluator, max_iterations = 3, parallel_evaluations = 10):
        """Class constructor

        :param evaluator: Evaluator to evaluate the target function
        :type evaluator: Evaluator
        :param max_iterations: maximum number of iterations, defaults to 3
        :type max_iterations: int, optional
        :param parallel_evaluations: parallel evaluations in each 
        iteration, defaults to 10
        :type parallel_evaluations: int, optional
        """
        super().__init__(evaluator)
        self.max_iterations = max_iterations
        self.parallel_evaluations = parallel_evaluations


    def doLineSearch(self, stepdirection, guess, target, J, r, result):

        # calculate the gradient at the current point
        grad = J.transpose().dot(r)   

        low = 0                     # current lowest value of the search window
        top = 1                     # current highest value of the search window
        l = 0                       # current interation   
        
        overall_minnorm = float("inf")
        overall_minalpha = -1

        all_alphas = []

        result.addRunMetadata("ls_maxiterations", self.max_iterations)
        result.addRunMetadata("ls_parallel_evaluations", self.parallel_evaluations)

        while True:
            l += 1
            alphas = np.linspace(low, top, num=self.parallel_evaluations)
            evaluations = []
            for i in range(self.parallel_evaluations):           
                evaluations.append(guess+alphas[i]*stepdirection)                         

            with self.evaluator:
                nextevaluations = self.evaluator.evaluate(evaluations, True, "linesearch")

            nextfunctionvalues = self.measurementToNumpyArrayConverter(nextevaluations, target)

            allNone = True
            minnorm = float("inf")
            minindex = -1

            # find the evaluation with lowest residualnorm, and check if all evaluations returned none, i.e. did not finish in UG
            for i in range(self.parallel_evaluations):
                if isinstance(nextevaluations[i], ErroredEvaluation):
                    result.log("\t\talpha_" + str(i)+ " = " + str(alphas[i])+" errored: " + nextevaluations[i].reason)
                    all_alphas.append((alphas[i], None))
                    continue

                allNone = False

                residual = nextfunctionvalues[i]-target.getNumpyArray()
                residualnorm = 0.5*residual.dot(residual)
                all_alphas.append((alphas[i], residualnorm))
               
                result.log("\t\talpha_" + str(i) + " = " + str(alphas[i]) + ", evalid=" + str(nextevaluations[i].eval_id) + ", residual = " + str(residualnorm))  
            
                if(residualnorm  < minnorm):
                    minnorm = residualnorm
                    minindex = i
                    if(minnorm < overall_minnorm):
                        overall_minnorm = minnorm
                        overall_minalpha = alphas[minindex]
            

            if(allNone):
                result.log("\t ["+str(l)+"]: no run finished.")
                
                if l == self.max_iterations:                
                    result.addMetric("lineSearchAlphas", all_alphas)
                    return None, None
                else:
                    low = 0
                    top = top/self.parallel_evaluations
                    continue
                
            minindex_alpha = alphas[minindex]
            continue_override = False

            if minindex == self.parallel_evaluations-1:
                continue_override = True
                next_low = top
                next_top = top + (top-low)
            elif minindex == 0 and low == 0:
                continue_override = True
                next_low = 0
                next_top = top/self.parallel_evaluations
            else:
                next_low = max(0, minindex_alpha - (top-low)/4)
                next_top = minindex_alpha + (top-low)/4
            
            lowerbound = 0.5*r.dot(r) + self.c * overall_minalpha * grad.transpose().dot(stepdirection)
            result.log("\t ["+str(l)+"]: min_alpha = " + str(overall_minalpha) + ", next interval = [" + str(next_low) + ", " + str(next_top) + "], new residualnorm: " + str(overall_minnorm) + ", wolfe lower bound: " + str(lowerbound))
            

            if((overall_minnorm < lowerbound and not continue_override)):
                result.addMetric("alpha", overall_minalpha)
                result.addMetric("lineSearchAlphas", all_alphas)
                return guess+overall_minalpha*stepdirection, overall_minnorm

            if l == self.max_iterations:
                                
                result.addMetric("lineSearchAlphas", all_alphas)
                if overall_minnorm < lowerbound:
                    result.addMetric("alpha", overall_minalpha)
                    return guess+overall_minalpha*stepdirection, overall_minnorm

                return None, None

            low = next_low
            top = next_top            
                

class LogarithmicParallelLineSearch(LineSearch):
    """A parallel implementation of a line search, with the search
    values of the form 1/2^{-i} for multiple values of i in parallel.

    If no suitable value (according to the first wolfe condition with c = 1e-3)
    is found, the next iteration is executed, with higher values for i.
    """
    size = 5
    highest_power = 0               
    parallel_evaluations = 10   # number of parallel evaluations during the parallel line search
    c = 1e-3 
    max_iterations = 2

    def __init__(self, evaluator, max_iterations = 2, size = 5, parallel_evaluations = 10):
        """Class constructor

        :param evaluator: Evaluator to evaluate the target function
        :type evaluator: Evaluator
        :param max_iterations: maximum number of iterations, defaults to 3
        :type max_iterations: int, optional
        :param size: size of the search window, in powers of two. 
        A value of 5 (default), means the initial search window is between 1 and 1/2^5
        :type size: int, optional
        :param parallel_evaluations: parallel evaluations in each 
        iteration, defaults to 10
        :type parallel_evaluations: int, optional
        """
        super().__init__(evaluator)
        self.max_iterations = max_iterations
        self.size = size
        self.parallel_evaluations = parallel_evaluations

    def doLineSearch(self, stepdirection, guess, target, J, r, result):

        # calculate the gradient at the current point
        grad = J.transpose().dot(r)   
        l = 0
        highest_power = self.highest_power
        all_alphas = []
        
        result.addRunMetadata("ls_maxiterations", self.max_iterations)
        result.addRunMetadata("ls_size", self.size)
        result.addRunMetadata("ls_parallel_evaluations", self.parallel_evaluations)

        while True:               
            l += 1
            evaluations = []
            alphas = np.logspace(highest_power-self.size, highest_power, base=2, num=self.parallel_evaluations)
            for i in range(self.parallel_evaluations):          
                evaluations.append(guess+alphas[i]*stepdirection)                         

            with self.evaluator:
                nextevaluations = self.evaluator.evaluate(evaluations, True, "linesearch")

            nextfunctionvalues = self.measurementToNumpyArrayConverter(nextevaluations, target)

            allNone = True
            minnorm = float("inf")
            minindex = -1

            # find the evaluation with lowest residualnorm, and check if all evaluations returned none, i.e. did not finish in UG
            for i in range(self.parallel_evaluations):
                if isinstance(nextevaluations[i], ErroredEvaluation):
                    result.log("\t\talpha_" + str(i)+ " = " + str(alphas[i]) + " did not finish: : " + nextevaluations[i].reason)
                    all_alphas.append((alphas[i], None))
                    continue

                allNone = False

                residual = nextfunctionvalues[i]-target.getNumpyArray()
                residualnorm = 0.5*residual.dot(residual)
                all_alphas.append((alphas[i], residualnorm))

                result.log("\t\talpha_" + str(i) + " = " + str(alphas[i]) + ", evalid=" + str(nextevaluations[i].eval_id) + ", residual = " + str(residualnorm))  
                
                if(residualnorm  < minnorm):
                    minnorm = residualnorm
                    minindex = i
            
            if(allNone):            
                result.log("\tno run finished.")
                
                if l == self.max_iterations:

                    result.addMetric("lineSearchAlphas", all_alphas)
                    return None, None
                else:
                    highest_power -= self.size
                    continue
            
            minindex_alpha = alphas[minindex]

            lowerbound = 0.5*r.dot(r) + self.c * minindex_alpha * grad.transpose().dot(stepdirection)
            result.log("\t ["+str(l)+"]: min_alpha = " + str(minindex_alpha) + ", with cost: " + str(minnorm) + ", wolfe lower bound: " + str(lowerbound))
            
            if minnorm < lowerbound and minindex != 0:          
                result.addMetric("alpha", minindex_alpha)
                result.addMetric("lineSearchAlphas", all_alphas)
                return guess+minindex_alpha*stepdirection, minnorm
            elif l == self.max_iterations:
                if minnorm < lowerbound:
                    result.addMetric("alpha", minindex_alpha)
                    result.addMetric("lineSearchAlphas", all_alphas)
                    return guess+minindex_alpha*stepdirection, minnorm
                result.addMetric("lineSearchAlphas", all_alphas)
                return None, None
            
            highest_power -= self.size
            
class BacktrackingLineSearch(LineSearch):
    """An serial (as in not parallelized) version of a backtracking line search,
    halving the step size if no suitable step size according to the first
    wolfe conditions is found.
    """

    # parameters
    c = 1e-3
    rho = 0.5
    max_iterations = 15

    def doLineSearch(self, stepdirection, guess, target, J, r, result):

        # do backtracking line search
        grad = J.transpose().dot(r)   
        alpha = 1
        l = 0

        while True:                 
            nextguess = guess+alpha*stepdirection
            with self.evaluator:            
                nextevaluation = self.evaluator.evaluate([nextguess], True, "linesearch")

            if nextevaluation is None or isinstance(nextevaluation, ErroredEvaluation):
                return None, None
            
            nextfunctionvalue = self.measurementToNumpyArrayConverter(nextevaluation, target)[0]

            residual = nextfunctionvalue-target.getNumpyArray()
            nextresidualnorm = 0.5*residual.dot(residual)  

            # wolfe bound
            lowerbound = 0.5*r.dot(r) + self.c * alpha * grad.transpose().dot(stepdirection)

            result.log("\t\t ["+str(l)+"]: alpha = " + str(alpha) + ", new residualnorm: " + str(nextresidualnorm) + ", wolfe lower bound: " + str(lowerbound))
            l += 1

            result.addMetric("alpha",alpha)

            if(nextresidualnorm <= lowerbound):
                return nextguess, nextresidualnorm
            alpha = alpha * self.rho

            if(l == self.max_iterations):
                return None, None