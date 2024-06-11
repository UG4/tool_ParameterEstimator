# Parameter Estimator package

This package is largely based on the plugin [ParameterEstimation](https://github.com/UG4/plugin_ParameterEstimation) made by Tim Schön.

The Parameter Estimator package is a tool for estimating the parameters of a model from experimental data for [UG4](https://gcsc.uni-frankfurt.de/simulation-and-modelling/ug4). It is *not* a plugin for UG4.

## Installation

This package can be installed with pip:

```bash
pip install -e .
```

### Dependencies

To use this plugin, you need to have UG4 installed (see [ughub](https://github.com/UG4/ughub)).

JSON has to be enabled (`-DUSE_JSON=ON`) and the [JSONToolkit](https://github.com/UG4/JSONToolkit) plugin installed (`-DJSONToolkit=ON`).

It is recommended to install the following dependencies before compiling UG4:
- [BLAS](http://www.netlib.org/blas/)
- [LAPACK](http://www.netlib.org/lapack/)

## Usage

To use the plugin, you need to have a model implemented in UG4. The model should be run by a Lua script. For each model, you need to provide a custom python script that defines the model and the parameters to be estimated, as well as the experimental data. An example of such a script is given below. Examples can be found in [Examples](Examples/) Directory.

### Parameters

There are two types of parameters. The first type are the parameters to be estimated. They are defined using the `ParameterManager` class and the `addParameter()` method. It takes the name of the parameter and the initial value as arguments. Optionally, you can also provide lower and upper bounds for the parameter. The name of the parameter has to be the same as the name used in the Lua script of the model. 

The second type are fixed parameters. They are defined using the `fixedparameters` attribute of the `Evaluator` class. The name of the parameter has to be the same as the name used in the Lua script of the model. These parameters are not estimated, but are kept fixed during the optimization.

### Model

The model is defined using the `Evaluator` class. It takes the following arguments:
| Argument | Description | Optional |
| --- | --- | --- |
| `luafile` | Path to the Lua script of the model | No |
| `cliparameters` | Command line parameters for the model | Yes |
| `directory` | Directory for the evaluations | Yes |
| `parametermanager` | Parameter manager | No |
| `evaluation_type` | Evaluation type | No |
| `parameter_output_adapter` | Parameter output adapter | No |
| `weight` | Weight for the measurements. Only in effect when multiple measurements were found. | Yes |
| `ugsubmitparameters` | Parameters for UGSUBMIT | Yes |
| `threadcount` | Threads to use | Yes |


### Optimizer

There is currently only one optimizer implemented, the Gauss-Newton optimizer.


### Target data

The target data is defined as another evaluation, preferably from a CSV file by using the `fromCSV()` method of the `GenericEvaluation` or `FreesurfaceEvaluation` class. The CSV file for that has to have the following format: the first column is the step number, the second column the time, and the third column the value of the measurement. 

In `GenericEvalution` projects you can have multiple measurements the optimizer should account for. In this case they have to be provided as one single csv file appended to each other. The `weight` parameter of the `Evaluator` class can be used to assign different weights to the measurements. The different measurements are detected by a lower step oder time value than the previous measurement.

:warning: The CSV file has to have the same number of measurements as the model has outputs. It does not have to have the same number of steps or time values, but the measurements have to be in the same order as the outputs of the model.

:warning: Multiple measurements for targets are not (yet) supported in `FreesurfaceEvaluation`.

### Example

Make sure that the environment variable `UG4_ROOT` is set to the root directory of your UG4 installation.

The package can be imported with

```from UGParameterEstimator import *```

#### Complete example

```python
#!/usr/bin/env python3

from UGParameterEstimator import *

# Define the parameters
pm = ParameterManager()
pm.addParameter("parameter1", 0.0, 1.0)  # name, start value, [lower bound], [upper bound]

# Define the model
evaluator = Evaluator.ConstructEvaluator(
    luafile="path/to/model.lua",                            # path to the model's lua file
    cliparameters=["-p", "details.lua"],                    # command line parameters for the model
    directory="path/to/evaluations",                        # directory for the evaluations
    parametermanager=pm,                                    # parameter manager from above
    evaluation_type=GenericEvaluation,                      # evaluation type
    parameter_output_adapter=UG3ParameterOutputAdapter(),   # parameter output adapter
    weight=[2, 1],                                          # weight for the measurements
    ugsubmitparameters=["-walltime", "01:30:00"],           # parameters for UGSUBMIT
    threadcount=1                                           # threads to use
)

# Define the optimizer
optimizer = GaussNewtonOptimizer(LogarithmicParallelLineSearch(evaluator))

# Define fixed parameters
evaluator.fixedparameters["fixed_parameter1"] = 1.0

# Define target data
target = GenericEvaluation([], []).fromCSV("path/to/experimental/data.csv")

# Store the results of the optimization in a pkl file and run the optimization
result = Result("path/to/results.pkl")

result = optimizer.run(evaluator, pm.getInitialArray(), target, result=result)

```
