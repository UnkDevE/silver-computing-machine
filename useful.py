
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
# assume planes in hessian
def hyperplane_intersect(ta, tb):
    # assume normalized
    # so as ta and tb are bases so we have to calcualate vectors        
    # both are in abs form so we can take diff
    # we then set theta 
    amat = np.asarray([ta.transpose(),tb.transpose()])
    b = np.abs(ta - tb)
    theta = np.arccos(ta.transpose()*tb)

    if np.sin(theta).any(0):
        # vectors are colinear so throw soft error to be dealt with later
        if np.all([ta, tb]) : return ta
        raise BaseException("vectors are colinear")

    xln = amat.transpose() * sci.linalg.inv(amat * amat.transpose()) * b
   
    from scipy.linalg import null_space
    nulls = null_space(amat)        
    return xln + nulls

# find intersects
tensorsect = hyperplane_intersect(sumtensors[0], sumtensors[1])
for tensor in sumtensors[2:]:
    tensorsect = hyperplane_intersect(tensorsect, tensor)

