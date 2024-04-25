#!/usr/bin/env python3

# import from plugins folder
# please make sure the enivorment variable UG4_ROOT points to your UG4 directory!
import sys, os
sys.path.append(os.path.join(os.environ["UG4_ROOT"], "plugins", "ParameterEstimator", "python")) 

from UGParameterEstimator import *

# specify the parameters
pm = ParameterManager()

# direct parameter: no transformation from parameter used in optimizer to parameter in lua
# the specified value is the initial value used in the optimization
# lower and upper bounds can be supplied as third and fourth argument
pm.addParameter(DirectParameter("alpha_inner", 20))
pm.addParameter(DirectParameter("alpha_door", 0.2))

# create the evaluator object. this will create a LocalEvaluator, which uses MPI for parallelism > 1, if no UGSUBMIT was found,
# or an ClusterEvaluator using UGSUBMIT.
evaluator = Evaluator.ConstructEvaluator(
    luafile="evaluate.lua",             # the lua file to execute for every evaluation
    directory="evaluations",            # the folder used for data exchange
    parametermanager=pm,                # the parameters defined above
    evaluation_type=GenericEvaluation,  # the type the evaluations should be parsed as.
    parameter_output_adapter=UG4ParameterOutputAdapter(),       # the adapter to use to write the parameters
    threadcount=1)                     # threads to use locally or when using UGSUBMIT

# create the optimizer
optimizer = GaussNewtonOptimizer(LinearParallelLineSearch(evaluator))

# specify some fixed parameters if needed (could be done in lua, also)
evaluator.fixedparameters["alpha_wall"] = 0.994

# this will do a measurement with fixed parameters
with evaluator:
    target = evaluator.evaluate([np.array([10, 0.1])], transform=False)[0]

# try to restore these parameters by calibration
# store the calibration process and logging in example.pkl
result = Result("example.pkl")
result = optimizer.run(evaluator, pm.getInitialArray(), target, result=result)
