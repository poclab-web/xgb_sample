import os
import argparse
import json
import numpy as np

from load_data import get_test_train_data
from setting import setting
import pickle

from required_functions.load_model import load_model
from required_functions.validate import kfold
from tuning import tune

from required_functions.logging import set_logger

def train(args):
    # start tuning and find best parameters
    print('tuning start')
    best_params = tune(csv_path=args.csv_path, task=args.mode, tuning_method=args.tuning_method)

    result_dir = './result'
    if not os.path.exists(result_dir):
        os.makedirs(result_dir, exist_ok=True)
    save_dir = './result/{}'.format(args.task_id)
    # save json
    json_path = os.path.join(save_dir, 'best_params.json')
    json_file = open(json_path, 'w')
    json.dump(best_params, json_file)

    # training
    # load model
    model = load_model(task=args.mode)
    model.set_params(**best_params)
    x_train, x_test, y_train, y_test = get_test_train_data(args.csv_path, setting['x_range'], setting["y_column"])
    # training
    model.fit(x_train, y_train)
    
    # validate model
    scores = kfold(model, x_test, y_test)
    logger('validate score:::{}'.format(scores))
    logger('validate mead:::{}'.format(np.mean(scores)))

    model_path = './result/{}/model.pickle'
    f = open(model_path, 'r')
    pickle.dump(model, f)
    logger('model saved!!')
    
    return scores

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--task_id', default='xgboost_result', type=str, help="name of task to specify")
    parser.add_argument('--csv_path', required=True, type=str, help="where to save csv")
    parser.add_argument('--mode', required=True, type=str, help="task type choose from regression or classifier")
    parser.add_argument('--tuning_method', required=True, type=str, help="tuning method choose from ")

    args = parser.parse_args()

    # set logging
    logger = set_logger('./log/{}.log'.format(args.task_id), 'XGBoost')

    train(args)
