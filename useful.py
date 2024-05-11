
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

from itertools import cycle
cs = common_shape(arr, prev_input)
acs = [x if x != y else None for x, y in zip(list(arr.shape),
         cycle(cs))]
acs = list(filter(lambda x: x is not None, acs))



   def calc_expr(coeff, prev_input, ops, exp):
        # expands out vector shapes
        # input is the given vector
        def reshaper(arr, input):
            from itertools import cycle
            cs = common_shape(arr, input)
            acs = [x if x != y else None for x, y in zip(list(arr.shape),
                             cycle(cs))]
            acs = list(filter(lambda x: x is not None, acs))
            reshape = flatten([cs, [1 for x in acs]])
            
            vec = input.reshape(reshape)
            return np.repeat(vec, acs, axis=len(acs))

        # nabbed from sympy source and edited for use case
        arr = empty(coeff.shape, dtype=object)
        vec = np.power(prev_input, exp)

        if len(vec.shape) < len(coeff.shape): # vector
            if sum(vec.shape) < sum(coeff.shape) and len(vec.shape) < len(coeff.shape):
                vec = reshaper(coeff, vec)
            else:
                coeff = reshaper(vec, coeff)

            arr = ops(vec, coeff)
        else:
            arr = ops(vec, coeff)

        return matrix(SR, *arr.shape, arr) 


    def classify_out(i, l):
        return [1 if i == x else 0 for x in range(0, l)]
    

    def reduce(system, i, ops):
        # gives list of vecs
        eqs = list(system)
        eq0 = eqs.pop(i)
        for eq in eqs:
            eq0 = ops(eq0, eq)
        return eq0