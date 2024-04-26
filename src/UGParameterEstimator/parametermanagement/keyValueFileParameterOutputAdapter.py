from .parameterOutputAdapter import ParameterOutputAdapter
from .parameterManager import ParameterManager
import os

# writes the parameters to be calibrated and given fixed parameters as a simple key-value file
# each parameter will be in a new line, containing the parameter name, an equal sign and the current parameter value
#
# eg: 
#   porosity=0.1
#   permeability=1e-10
#
class KeyValueFileParameterOutputAdapter(ParameterOutputAdapter):

    def writeParameters(self, directory: str, evaluation_id: int, parametermanager: ParameterManager, parametervalues, fixedparameters):
        
        parameterfile = os.path.join(directory, str(evaluation_id) + "_parameters.txt")
        # write the parameter file parsed in lua
        with open(parameterfile,"w") as f:
            for i in range(len(parametervalues)):
                f.write(parametermanager.parameters[i].name + "=" + str(parametervalues[i]) + "\n")
            for k in fixedparameters:
                f.write(k + "=" + str(fixedparameters[k]) + "\n")