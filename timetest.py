import tensorflow_datasets as tdfs
import tensorflow as tf
import numpy as np

import inspect
import cProfile
from itertools import product

# change for EAGER mode
@tf.function
def len_ds(ds):
    length_np = 0
    for _ in ds.map(lambda x: 1):
        length_np += 1
    return length_np

@tf.function
def len_ds_auto(ds):
    length_np = 0
    for _ in ds.map(lambda x: 1, 
            num_parallel_calls=tf.data.AUTOTUNE, deterministic=False):
        length_np += 1
    return length_np

@tf.function
def len_ds1(ds):
    return ds.reduce(lambda x, _: x + 1).numpy()

@tf.function
def len_ds2(ds):
    return np.fromiter(ds.map(lambda x: 1).as_numpy_iterator()).sum()

@tf.function
def len_ds2_auto(ds):
    return np.fromiter(ds.map(lambda x: 1, 
        num_parallel_calls=tf.data.AUTOTUNE, deterministic=False).as_numpy_iterator()).sum()

@tf.function
def len_ds3(ds):
    return np.sum(np.fromiter(ds.map(lambda x: 1)).as_numpy_iterator())

@tf.function
def len_ds3_auto(ds):
    return np.sum(np.fromiter(ds.map(lambda x: 1),
        num_parallel_calls=tf.data.AUTOTUNE, deterministic=False).as_numpy_iterator())


train, test = tdfs.load('mnist', download=False, split=['train', 'test'])

function_ptrs = [len_ds, len_ds_auto, len_ds1, len_ds2, len_ds2_auto, len_ds3, len_ds3_auto]
function_str_dict = [inspect.getsource(func) for func in function_ptrs]

def profiler(DEBUG, EAGER):
    for i, func in enumerate(function_str_dict):
        if DEBUG:
           cProfile.run(func, 'statsdebug' + str(i) + '.prof') 
        if EAGER:
           cProfile.run(func, 'statseager' + str(i) + '.prof') 
        if EAGER and DEBUG:
           cProfile.run(func, 'statseagerdebug' + str(i) + '.prof') 
        else:
           cProfile.run(func, 'stats' + str(i) + '.prof') 

if __name__ == '__main__':
    for [eager, debug] in product([0, 1], repeat=2):
        profiler(bool(debug), bool(eager))