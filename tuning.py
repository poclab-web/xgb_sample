import os
import sys

# modules
sys.path.append(os.path.dirname(__file__))
from setting import setting
from required_functions.load_model import load_model
from load_data import get_test_train_data
from required_functions.validate import kfold
from required_functions.pretuning import pretuning
from required_functions.tuning_method import TuningMethod

import logging
logger = logging.getLogger('XGBoost')


def tune(csv_path, task="regression", tuning_method="grid_search"):
    # load data
    x_train, _, y_train, _ = get_test_train_data(csv_path, setting['x_range'], setting["y_column"])

    # define model
    model = load_model(task)

    # first validation
    scores = kfold(model, x_train, y_train)
    print('------------First Validation------------')
    for i, s in enumerate(scores):
        print('step{} : {}'.format(i + 1, s))
    print('----------------------------------------')

    if tuning_method == "grid-search":
        logger.info('grid-search')
        # pretuning
        cv_train = pretuning(model, x_train, y_train)
        logger.info('pre tuning finished!!')
        # tuning
        tuner = TuningMethod(model, cv_train)
        best_params = tuner.grid_search(x_train, y_train)
    
    elif tuning_method == "random-search":
        logger.info('random-search')
        # pretuning
        cv_train = pretuning(model, x_train, y_train)
        logger.info('pre tuning finished!!')
        # tuning
        tuner = TuningMethod(model, cv_train)
        best_params = tuner.random_search(x_train, y_train)
    
    elif tuning_method == "bayes-optimization":
        logger.info("bayes optimization")
        cv_train = None
        tuner = TuningMethod(model, cv_train)
        best_params = tuner.bayse_optimization(x_train, y_train)
    
    else:
        raise ValueError('Choose tuning method from: [1] grid-search [2] random-search [3] bayes-optimization')
    print(best_params)
    return best_params


if __name__ == '__main__':
    csv_path = '/Users/watanabeyuuya/Documents/lab/Projects/photopolymerization_initiator/data/oximesters_fingerprint_T1.csv'
    tune(csv_path, task="regression", tuning_method="bayes-optimization")