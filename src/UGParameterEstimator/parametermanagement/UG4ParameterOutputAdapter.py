from .parameterOutputAdapter import ParameterOutputAdapter
from .parameterManager import ParameterManager
import json
import os

# Writes the Parameters to calibrate and all fixed parameters to a JSON file understandable by UG4
#
# The parameter can then be used in the lua file by loading them using the ParameterUtil plugin:
#
# local p = util.Parameters:fromfile(params.evaluationDir.."/"..params.evaluationId.."_parameters.json")
# ....
# porosity = p.porosity
# ....
class UG4ParameterOutputAdapter(ParameterOutputAdapter):

    def writeParameters(self, directory: str, evaluation_id: int, parametermanager: ParameterManager, parameter, fixedparameters):
        
        parameterfile = os.path.join(directory, str(evaluation_id) + "_parameters.json")

        # construct array object
        parameterlist = {}
        for i in range(len(parameter)):
            parameterlist[parametermanager.parameters[i].name] = { "type": "number", "value": parameter[i] }
                            

        for k in fixedparameters:                        
            parameterlist[k] = { "type": "number", "value": fixedparameters[k] }

        # write as json file
        # this will be parsed by UG4
        with open(parameterfile,"w") as f:
            json.dump(parameterlist, f)