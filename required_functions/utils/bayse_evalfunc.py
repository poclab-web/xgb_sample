import numpy as np
from sklearn.model_selection import cross_val_score

# だめだったらクラスにする
class EvalFunc:
    def __init__(self, model, x, y, param_scales, cv, scoring, fit_params):
        self.model = model
        self.x = x
        self.y = y
        self.param_scales = param_scales
        self.cv = cv
        self.scoring = scoring
        self.fit_params = fit_params
    
    def base_evalfunc(self, **params):
        # log 
        params = {k: np.power(10, v) if self.param_scales[k] == 'log' else v for k, v in params.items()}
        # int
        params = {k: round(v) if self.param_scales[k] == 'int' else v for k, v in params.items()}
        self.model.set_params(**params)
        # cross_val_scoreでクロスバリデーション
        scores = cross_val_score(self.model, self.x, self.y, cv=self.cv,
                            scoring=self.scoring, fit_params=self.fit_params, n_jobs=-1)
        val = scores.mean()
        return val




