import numpy
from .parameterManager import ParameterManager
from abc import ABC, abstractmethod

# Abstract Base Class / Interface for all ParameterOutputAdapters
class ParameterOutputAdapter(ABC):

    @abstractmethod
    def writeParameters(self, directory: str, evaluation_id: int, parametermanager: ParameterManager, parameter, fixedparameters):
        pass