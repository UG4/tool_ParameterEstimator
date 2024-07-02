import logging
logging.basicConfig(
    filename='parameterEstimator.log',
    filemode='w', # overwrite log file
    format='[%(asctime)s,%(msecs)d %(name)s] (%(levelname)s) %(message)s', # i.e. 2020-12-08 16:29:52,000 parameterEstimator INFO This is an info message
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.DEBUG,
    )
logger = logging.getLogger("parameterEstimator")
