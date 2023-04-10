from setting import setting

def load_model(task):
    if task == 'regression':
        from xgboost import XGBRegressor
        model = model = XGBRegressor(booster='gbtree', objective='reg:squarederror',
                        random_state=setting['SEED'], n_estimators=10000)
    elif task == 'classifier':
        from xgboost import XGBClassifier
        model = XGBClassifier(objective='binary:logistic', n_estimators=1000)
        
    else:
        raise ValueError('model task should be Regression or Classification')
    
    return model