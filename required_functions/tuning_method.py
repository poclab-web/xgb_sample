import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
from setting import setting
from loadJson import loadJson

from sklearn.model_selection import KFold
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from bayes_opt import BayesianOptimization

import numpy as np
from utils.bayse_evalfunc import EvalFunc
import datetime

import logging
logger = logging.getLogger('XGBoost')

class TuningMethod:
    def __init__(self, model, cv_params):
        self.cv_params = cv_params
        self.cv = KFold(n_splits=setting['KFOLD_NUM'], shuffle=True, random_state=setting['SEED'])
        self.model = model
    
    def set_params(self, x, y):
        fit_params = {
                'verbose': 0,  
                'early_stopping_rounds': 10, 
                'eval_metric': setting['METRICS'],  
                'eval_set': [(x, y)] 
            }
        return fit_params

    def grid_search(self, x, y):
        logger.info('optimize by using GridSearch')
        gridcv = GridSearchCV(self.model, self.cv_params, cv=self.cv,
                            scoring=setting['SCORING'], n_jobs=-1)
        # fitting step
        fit_params = self.set_params(x, y)

        start = datetime.datetime.now()
        gridcv.fit(x, y, **fit_params)
        elapsed = datetime.datetime.now() - start
        
        best_params = gridcv.best_params_
        best_score = gridcv.best_score_
        logger.debug('best_params::: {}'.format(best_params))
        logger.debug('best_score::: {}'.format(best_score))
        logger.info('time elpased to tuning::: {}'.format(elapsed))

        return best_params
    
    def random_search(self, x, y):
        logger.info('optimize by using RandomSearch')
        randomcv = RandomizedSearchCV(self.model, self.cv_params, cv=self.cv,
                            scoring=setting['SCORING'], n_jobs=-1, random_state=setting['SEED'], n_iter=1000)
        # fitting step
        fit_params = self.set_params(x, y)

        start = datetime.datetime.now()
        randomcv.fit(x, y, **fit_params)
        elapsed = datetime.datetime.now() - start
        
        best_params = randomcv.best_params_
        best_score = randomcv.best_score_
        logger.debug('best_params::: {}'.format(best_params))
        logger.debug('best_score::: {}'.format(best_score))
        logger.info('time elpased to tuning::: {}'.format(elapsed))
        return best_params

    def bayse_optimization(self, x, y):
        # load json file
        logger.info('start loading json file...')
        bayse_params = loadJson('./params/bayse_params.json')
        for k, v in bayse_params.items():
            bayse_params[k] = tuple(v)
        bayse_scale = loadJson('./params/bayse_scale.json')
        fit_params = self.set_params(x, y)

        bayes_params_log = {k: (np.log10(v[0]), np.log10(v[1])) if bayse_scale[k] == 'log' else v for k, v in bayse_params.items()}

        eval_func = EvalFunc(
            model=self.model,
            x=x,
            y=y,
            param_scales=bayse_scale,
            cv=self.cv,
            scoring=setting['SCORING'],
            fit_params=fit_params
        )

        start = datetime.datetime.now()
        bo = BayesianOptimization(eval_func.base_evalfunc, bayes_params_log)
        bo.maximize(init_points=20, n_iter=230, acq='ei')
        elapsed = datetime.datetime.now() - start

        # get best parameters
        best_params = bo.max['params']
        best_score = bo.max['target']

        # adjust data t original type
        best_params = {k: np.power(10, v) if bayse_scale[k] == 'log' else v for k, v in best_params.items()}
        best_params = {k: round(v) if bayse_scale[k] == 'int' else v for k, v in best_params.items()}

        logger.debug('best_params::: {}'.format(best_params))
        logger.debug('best_score::: {}'.format(best_score))
        logger.info('time elpased to tuning::: {}'.format(elapsed))

        return best_params

if __name__ == '__main__':
    from load_model import load_model
    from load_data import data
    # load data
    csv_path = '/Users/watanabeyuuya/Documents/lab/Projects/photopolymerization_initiator/data/oximesters_fingerprint_T1.csv'
    x_train, _, y_train, _ = data(csv_path, setting['x_range'], setting["y_column"])
    model = load_model('regression')
    cv_params = None
    tuner = TuningMethod(model, None)
    tuner.bayse_optimization(x_train, y_train)