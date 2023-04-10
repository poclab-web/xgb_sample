import sys, os
sys.path.append(os.path.dirname((__file__)))
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))

from sklearn.model_selection import validation_curve, KFold
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from loadJson import loadJson
from setting import setting
from utils.mathmatic import ArrayProcessor

import logging
logger = logging.getLogger('XGBoost')

def pretuning(model, x, y):
    """ pretuning hyperparametrs by K-folf method"""

    logger.info('pre tuning start')

    # load json
    cv_params = loadJson('./params/cv.json')
    params_scale = loadJson('./params/scale.json')
    assert len(cv_params) == len(params_scale), 'cv parameters and scales must be same length'

    # train settings
    fit_params = {
        'verbose': 0,
        'early_stopping_rounds': 10,
        'eval_metric': setting["METRICS"],
        'eval_set': [(x, y)]
    }

    # cross validation
    cv = KFold(n_splits=3, shuffle=True, random_state=setting["SEED"]) 

    # check initial model
    cv_params_train = {}
    fig, ax = plt.subplots()
    for k, v in tqdm(cv_params.items()):
        logger.info('now validating:::{}'.format(k))
        train_scores, valid_scores = validation_curve(
                estimator=model,
                X=x, 
                y=y,
                param_name=k,
                param_range=v,
                fit_params=fit_params,
                cv=cv,
                scoring=setting["SCORING"],
                n_jobs=-1
            )
        
        train_score_mean, validation_score_mean = np.mean(train_scores, axis=1), np.mean(valid_scores, axis=1)

        # search local max
        logger.debug('validation score :::{}'.format(validation_score_mean))
        print('validation score :::', validation_score_mean)
        val_arp = ArrayProcessor(validation_score_mean)
        lmax_index = val_arp.arroud_localmax()
        logger.debug('local max at {}:::{}'.format(k, lmax_index))

        # update params
        object_array = np.array(cv_params[k])
        cv_params_train[k] = object_array[lmax_index].tolist()

        logger.debug('cv params updated:::{}'.format(cv_params_train[k]))

    return cv_params_train

if __name__ == '__main__':
    print(loadJson('./params/cv.json'))