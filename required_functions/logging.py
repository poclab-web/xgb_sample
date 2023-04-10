from logging import getLogger, DEBUG, Formatter, FileHandler
import os

def set_logger(log_path, logger_name):
    
    if not os.path.exists('./log'):
        print('make directory...')
        os.makedirs('./log')

    logger = getLogger(logger_name)
    handler = FileHandler(log_path)

    logger.setLevel(DEBUG)
    handler.setLevel(DEBUG)
    formatter = Formatter('%(asctime)s:%(levelname)s-%(filename)s-%(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger