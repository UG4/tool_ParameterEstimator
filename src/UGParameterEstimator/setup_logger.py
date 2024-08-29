"""This module contains the setup for the logger used in the parameterEstimator module."""
import logging

logging.basicConfig(
    filename='parameterEstimator.log',
    filemode='w', # overwrite log file
    format='[%(asctime)s %(name)s] (%(levelname)s) %(message)s',
    # i.e. [2020-01-01 12:00:00 parameterEstimator] (DEBUG) Starting newton method.
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.DEBUG,
)

logger = logging.getLogger("parameterEstimator")
