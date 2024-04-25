
import numpy as np
from enum import Enum
from abc import ABC, abstractmethod

class Parameter:

    def __init__(self, name, startvalue, minimumValue=None, maximumValue=None):
        self.name = name
        self.startvalue = startvalue
        self.minimumValue = minimumValue
        self.maximumValue = maximumValue

    @abstractmethod
    def getTransformedParameter(self, value):
        pass

    @property
    @abstractmethod
    def initialValue(self):
        pass

    @property
    @abstractmethod
    def optimizationSpaceUpperBound(self):
        pass
    
    @property
    @abstractmethod
    def optimizationSpaceLowerBound(self):
        pass

    def isValidOptimizationSpaceParameter(self, value):

        if self.maximumValue is not None and self.optimizationSpaceUpperBound < value:

            print("Parameter " + self.name + " out of bounds, " + str(value) + " > " + str(self.optimizationSpaceUpperBound))

            return False

        if self.minimumValue is not None and self.optimizationSpaceLowerBound > value:
            
            print("Parameter " + self.name + " out of bounds, " + str(value) + " < " + str(self.optimizationSpaceLowerBound))

            return False

        return True

class DirectParameter(Parameter):

    def getTransformedParameter(self, value):

        if not self.isValidOptimizationSpaceParameter(value):
            return None

        return value

    @property
    def initialValue(self):
        return self.startvalue

    @property
    def optimizationSpaceUpperBound(self):
        return self.maximumValue

    @property
    def optimizationSpaceLowerBound(self):
        return self.minimumValue

class LogParameter(Parameter):

    def getTransformedParameter(self, value):

        if not self.isValidOptimizationSpaceParameter(value):
            return None

        return np.exp(value)

    @property
    def initialValue(self):
        return np.log(self.startvalue)

    @property
    def optimizationSpaceUpperBound(self):
        
        return np.log(self.maximumValue)

    @property
    def optimizationSpaceLowerBound(self):
        return np.log(self.minimumValue)

class ScaledParameter(Parameter):

    def getTransformedParameter(self, value):

        if not self.isValidOptimizationSpaceParameter(value):
            return None

        return self.startvalue*value

    @property
    def initialValue(self):
        return 1

    @property
    def optimizationSpaceUpperBound(self):
        return self.maximumValue/self.startvalue

    @property
    def optimizationSpaceLowerBound(self):
        return self.minimumValue/self.startvalue


class ParameterManager:
    
    class WrongMappingError(Exception):
        pass

    def __init__(self):
        self.parameters = []

    def addParameter(self, parameter):
        self.parameters.append(parameter)

    def getInitialArray(self):
        array = np.zeros(len(self.parameters))

        for i in range(len(self.parameters)):
            array[i] = self.parameters[i].initialValue

        return array

    def getTransformedParameters(self, beta):

        returnvalue = []

        for i in range(len(self.parameters)):
            param = self.parameters[i].getTransformedParameter(beta[i])

            if param is None:
                return None

            returnvalue.append(param)

        return returnvalue

    def isValidOptimizationSpaceParameter(self, beta):

        for i in range(len(self.parameters)):
            if not self.parameters[i].isValidOptimizationSpaceParameter(beta[i]):
                return False
        
        return True
