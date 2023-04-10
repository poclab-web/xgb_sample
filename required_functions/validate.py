import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))

from setting import *
from sklearn.model_selection import cross_val_score, KFold


def kfold(model, x, y, scoring=setting['SCORING']):
    fit_params = {
                    'verbose': 0,
                    'early_stopping_rounds': 10,
                    'eval_metric': setting["METRICS"],
                    'eval_set': [(x, y)]
                }
    cv = KFold(n_splits=setting['KFOLD_NUM'], shuffle=True, random_state=setting['SEED'])
    scores = cross_val_score(model, x, y, cv=cv, scoring=scoring, n_jobs=-1, fit_params=fit_params)
    return scores
