import numpy as np
from scipy.signal import argrelmax

import logging
loggin = logging.getLogger('XGBoost')

def getUpperIndex(index_array, index, threshold):
    upper_can = index + threshold
    index_larger = np.where(index_array > index)[0].shape[0]
    larger_num = np.where(index_array > upper_can)[0].shape[0]

    if index_larger == larger_num:
        return upper_can
    else:
        # next index
        index_index = np.where(index_array == index)[0][0]
        print(index_index)
        return index_array[index_index][0]

def getLowerIndex(index_array, index, threshold):
    lower_can = index - threshold
    index_larger = np.where(index_array < index)[0].shape[0]
    lower_num = np.where(index_array < lower_can)[0].shape[0]

    if index_larger == lower_num:
        return lower_can
    else:
        # next index
        index_index = np.where(index_array == index)
        return index_index[0][0]



class ArrayProcessor:
    """ processing for value array """
    def __init__(self, array):
        self.array = array
    
    def arroud_localmax(self, local_threshold=None):
        # index
        lmax = argrelmax(self.array)

        # if local_threshold:
        #     for index in lmax:
        #         # calculate index
        #         upper_index = getUpperIndex(lmax, index, local_threshold)
        #         lower = getLowerIndex(lmax, index, local_threshold)
        return lmax

if __name__ == '__main__':
    test_array = np.random.randint(0, 100, (10))
    test_array.sort()
    print('test array:::', test_array)
    index_choose = np.random.choice(test_array)
    print('index choose:::', index_choose)
    index = getUpperIndex(test_array, index_choose, 50)
    print(index)
    print(test_array[index])

    test_array = np.random.randint(0, 100, (10))
    test_array.sort()
    print('test array:::', test_array)
    index_choose = np.random.choice(test_array)
    print('index choose:::', index_choose)
    index = getLowerIndex(test_array, index_choose, 50)
    print(index)
    print(test_array[index])