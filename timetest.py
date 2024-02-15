import tensorflow_datasets as tdfs
import tensorflow as tf
import numpy as np

import inspect
import cProfile
from itertools import product

tf.config.run_functions_eagerly(True)
tf.data.experimental.enable_debug_mode()

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
    return ds.reduce(np.int64(0), lambda x, _: x + 1)

def len_ds2(ds):
    return np.fromiter(ds.map(lambda x: 1).as_numpy_iterator(), np.int64).sum()

@tf.function
def len_ds2_auto(ds):
    return np.fromiter(ds.map(lambda x: 1, 
        num_parallel_calls=tf.data.AUTOTUNE, deterministic=False).as_numpy_iterator(), np.int64).sum()

@tf.function
def len_ds3(ds):
    return np.sum(np.fromiter(
        ds.map(lambda x: 1).as_numpy_iterator(), np.int64))

@tf.function
def len_ds3_auto(ds):
    return np.sum(np.fromiter(ds.map(lambda x: 1,
        num_parallel_calls=tf.data.AUTOTUNE, deterministic=False).as_numpy_iterator(), np.int64))


[train, test] = tdfs.load('mnist', download=False, split=['train', 'test'])

# obfuscate length so it has to calculate
train = train.filter(lambda x: x == x)

function_ptrs = [len_ds, len_ds_auto, len_ds1, len_ds2, len_ds2_auto, len_ds3, len_ds3_auto]

if __name__ == '__main__':
    for i, func in enumerate(function_ptrs):
        with cProfile.Profile() as pr:
            print(inspect.getsource(func))
            func(train)
            pr.print_stats()