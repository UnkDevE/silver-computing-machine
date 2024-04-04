
# vector multiplication because sympy doesn't do it
# with matricies is done by repeating the column in numpy  
# if not a vector it leaves it alone
def vecmul(vec_m, target, transpose):
    inner = vec_m
    if vec_m.shape[1] == 1:
       inner = np.repeat(inner, repeats=target, axis=1)
       if transpose:
           inner = inner.transpose()
    return syp.Matrix(inner)


# this is our optimized len fn
@tf.function
def len_ds(ds):
    length_np = 0
    for _ in ds.map(lambda x: 1, 
            num_parallel_calls=tf.data.AUTOTUNE, deterministic=False):
        length_np += 1
    return tf.cast(length_np, tf.int64)

# normalize first
def normalize(tensor, constant_idx):
    constant_idx = tensor.pop(constant_idx)
    squared = np.sqrt(sum(tensor**2))
    tensor.append(constant_idx)
    return map(lambda t: t / squared, tensor)
#