import polars as pl
from sklearn.model_selection import train_test_split
import os
import time
import logging
from setting import *

logger = logging.getLogger('XGBoost')

def get_test_train_data(csv_path, x_columns_range, y_column):
    # split training data and test data
    assert os.path.splitext(csv_path)[-1] == '.csv', 'input file must be csv!!'
    print('X data range will be extract by slice...')

    start = time.time()

    # load dataframes
    df = pl.read_csv(csv_path)
    logger.info('drop null columns')
    df = df.drop_nulls()

    # load x
    x = df[:, min(x_columns_range):max(x_columns_range)].to_numpy()
    print('shape of x:::', x.shape)
    # load y
    y = df.get_column(y_column).to_numpy()
    print('shape of y:::', y.shape)

    logger.info('time elapsed to load data:::{}'.format(time.time() - start))
    logger.debug('sample of data loaded as X:::{}'.format(x[0]))
    logger.debug('sample of data loaded as Y:::{}'.format(y[0]))
    print('sample of data loaded as X:::{}'.format(x[0]))
    print('sample of data loaded as Y:::{}'.format(y[0])) 

    x_train, x_test, y_train, y_test = train_test_split(
                                                        x, 
                                                        y, 
                                                        test_size=setting['TEST_SIZE'], 
                                                        random_state=setting['SEED']
                                                    )
    logger.info('input data were split')
    logger.info('seed:::{}'.format(setting['SEED']))
    return x_train, x_test, y_train, y_test

def load_data(csv_path, x_columns_range, y_column):
    # load dataframes
    df = pl.read_csv(csv_path)
    logger.info('drop null columns')
    df = df.drop_nulls()

    # load x
    x = df[:, min(x_columns_range):max(x_columns_range)].to_numpy()
    print('shape of x:::', x.shape)
    # load y
    y = df.get_column(y_column).to_numpy()
    print('shape of y:::', y.shape)

    return x, y


if __name__ == '__main__':
    csv_path = '/Users/watanabeyuuya/Documents/lab/Projects/photopolymerization_initiator/data/oximeesters_all.csv'
    x_cols_range = (1, 1000)
    split_data(csv_path, x_cols_range, 'T1')