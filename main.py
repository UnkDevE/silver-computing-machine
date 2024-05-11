"""
    SILVER-COMPUTING-MACHINE converts Nerual nets into human readable code or maths 
    Copyright (C) 2024 Ethan Riley 

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""
import sys
import os

import tensorflow as tf
import tensorflow_datasets as tdfs 

import numpy as np
import pandas
# heaveside is used here it's just eval'ed in
from sage.all import var, heaviside
from sage.matrix.constructor import matrix

# list of activation functions
ACTIVATION_LIST = """activation;function;
linear;x;
step;heaviside(x)
logistic;1/(1+e^-x);
tanh;e^x - e^-x/e^x+e^-x;
smht;e^ax - e^-bx/e^cx + e^-dx;
relu;x>0;
softplus;ln(1+e^x);
"""

INPUT_SYMBOL = var("x")
BATCH_SIZE = 1024

tf.config.run_functions_eagerly(True)
# do not remove forces tf.data to run eagerly
# tf.data.experimental.enable_debug_mode()

# saver function helper
def save(filename, write):
    import os.path
    file = None
    if os.path.isfile(filename):
        file = open(filename, "at")
    else:
        file = open(filename, "xt")
    file.write(write)
    file.close()


# hang fire with convolutions
# helper 
def complete_bias(shapes, targets):
    def tup_complete(tup, tar): 
        return (tup[0], tar) if len(tup) < 2 else tup

    shapes_new = [tup_complete(shapes[0], targets[0])]
    for shape, target in zip(shapes[1:], targets[1:]):
        new_shape = []
        for (n, tuples) in enumerate(shape):
            if n < 2:
                new_shape.append(tup_complete(tuples, target))
            else:
                new_shape.append(tup_complete(tuples, targets[0]))
        shapes_new.append(new_shape)
        
    return shapes_new

def activation_fn_lookup(activ_src, csv):
    sourcefnc = activ_src.__name__
    for (i, acc) in enumerate(csv['activation']):
        if acc == sourcefnc:
            return eval(csv['function'][i])
    return eval(csv['function']['linear'])

def dtft(eq_system):
    from sage.all import I, e
    dtft_v = []
    #dtft 
    for i, system in enumerate(eq_system):
        np_sys = (system * (e ** -i*I))
        dtft_v.append(np_sys)

    return dtft_v
 
def common_shape(x, y):
    if x.shape == y.shape: 
        return list(x.shape)
    else:
        xs = []
        for x_s in x.shape:
            for y_s in y.shape:
                if x_s == y_s:
                    xs.append(x_s)
        if len(xs) < 1:
            raise "No common shape for" + str(x) + "and" + str(y)
        else:
            return xs

def get_poly_eqs_subst(shapes, activ_obj, fft_layers):
    from numpy import empty, ndindex
    from sage.all import vector
    from sage.all import SR
    from sage.misc.flatten import flatten
       
    # pre generates a matrix with symbols in
    # USES ONLY NUMPY ARRAYS as coeff and prev_input
    def calc_expr(coeff, prev_input, ops, exp):
        # expands out vector shapes
        # input is the given vector        
        arr = empty(coeff.shape, dtype=object)
        vec = np.power(prev_input, exp)
        arr = ops(vec, coeff)

        if len(arr.shape) < 1:
            return matrix(SR, *arr.shape, arr) 
        else:
            return vector(arr)

    x_input = empty(shapes[0], dtype=object)
    for n, index in enumerate(ndindex(shapes[0])):
        x_input[index] = var("x_" + str(n))
    # to system 
    eq_system = [vector(x_input.flatten())]

    # summate equations to get output
    for i in range(1, len(shapes)):
        layer = fft_layers[i-1]

        # get our matrix exprs 
        from operator import  __add__, __mul__, __matmul__
        prev = eq_system[-1].numpy()

        if layer[0].shape == prev.shape or len(prev.shape) == 1:
            weight = calc_expr(layer[0], prev, __matmul__, i)
        else:
            weight = calc_expr(layer[0], prev, __mul__, i)


        # power is always 1 here because bias is linear
        bias = calc_expr(layer[1], weight.numpy(), __add__, 1)
        
        # lookup the activation function 
        lookup = activation_fn_lookup(layer[2], activ_obj)
        
        # if at end of list don't apply activation fn
        if len(shapes) == i:
            eq_system.append(bias)
        else:
            eq_system.append((bias).apply_map(lambda x: lookup(x=x)))
       
    return eq_system
    
def evaluate_system(shapes, eq_system, tex_save, outputs):
    # set as a power series
    from sage.all import latex, pi, e, I
    from sage.plot.plot import plot
    
    # returns a the largest expression on the LHS
    def gr_operand(system):
        r = system.rhs() 
        l = system.lhs() 
        rhs, lhs = None, None
        if len(l.operands()) > 1:
            rhs = l
            if len(rhs.operands()) > len(r.operands()):
                lhs = rhs
                rhs = r
            else:
                lhs = r
        elif len(r.operands()) > 1:
            rhs, lhs = gr_operand(r, l)
        else:
            lhs = l
            rhs = r
        return lhs, rhs

    # transform into dtft
    fft_system = dtft(eq_system)

    
    outvec = fft_system[-1].numpy()
    
    # caclulate the chec cohomology using the fft_system as a linear system
    # fft_system is linear over polynomials of n 
    cocycle = np.sum(outvec * (-np.ones_like(outvec)**np.mgrid[0:len(outvec)]), axis=len(outvec.shape)-1)

    #cocycle check in floating point
    if cocycle <= 1 and cocycle > -1:
        raise("no sheaf avialable for NN check probablity scores")

    # simplex in this case is the direct sum
    simplex = outvec * (-np.ones_like(outvec)**np.mgrid[1:len(outvec)+1])

    # get line bundle via mul
    linebundle = simplex * -np.flip(simplex)
    
    # inverse dtft
    inverse = sum(linebundle * (e ** I * np.ones_like(linebundle) * (2*pi)))
    save("out.tex",latex(inverse))

"""
# evaluate irfftn using cauchy residue theorem
def evaluate_system(shapes, eq_system, out, tex_save):
    # inverse of fourier transform is anaglogous to convergence of fourier series
    from sympy import fourier_series, solve, latex, sympify 

    # set as a power series
    eq_poly = sum(eq_system[-1])

    # calculate inverse fourier of output side
    from scipy.fft import irfft
    fft_inverse = irfft(out, n=len(out))

    # calculate the limit
    from sympy.series.limitseq import limit_seq

    def get_len(shapes):
        l = shapes[0][0] * shapes[0][1] 
        for shape in shapes[1:]:
            l += shape[0][0] * shape[0][1]
        return l
    
    inf_poly = limit_seq(eq_poly, n=get_len(shapes))
    inf_poly = inf_poly.cancel()

    from sympy import srepr
    save("nn_poly", srepr(inf_poly))

    solved = syp.solve(inf_poly, fft_inverse, INPUT_SYMBOL).doit(deep=True)
    tex_save = latex(solved)
    print("eq solved")
"""

def output_aggregator(model, fft_layers, data):
    # load and get unique features
    [dataset, test] = tdfs.load(data, download=False, split=['train', 'test'])
    
    # get label and value
    value, *features = list(list(dataset.take(1).as_numpy_iterator())[0].keys())

   
    # get types of labels
    # dataset.unique() doesn't work with uint8
    # so we remove the offending key and use it
    def rm_val(d, val) :
        for v in val:
            if v in d:
                del d[v]
        return d
    
    # filter through to find duplicates
    values_removed = dataset.map(lambda i: rm_val(i, [value]))
                                 
    # call unqiue on features
    need_extract = values_removed.unique() 

    @tf.function
    def label_extract(label_extract, features):
        labels = []
        for feature in features:
            for label in label_extract: 
                if hasattr(label[feature], 'numpy') and callable(
                    getattr(label[feature], 'numpy')):
                        labels.append(label[feature].numpy())
        return labels
    
    labels = label_extract(need_extract, features)

    # have to update cardinality each time
    # remember labels is a list due to extract
    @tf.function
    def condense(v):
        b = True 
        for i in v:
            if not i: 
                b = False 
        return b
    
    # condense doesn't work as complains about python bool 
    auto_condense = tf.autograph.to_graph(condense.python_function)

    # bucketize each feature in each label, return complete datapoints 
    # bucketizing is failing at the moment because the labels are consumed
    sets = [dataset.filter(lambda i: 
            auto_condense([i[feature] == label for feature in features])) 
            for label in labels]

    # normalize function for images
    @tf.function
    def normalize_img(image):
      return tf.cast(image, tf.float32) / 255.

    # numpy array of predictions
    sumtensors = []
    for (i, dataset) in enumerate(sets):
        tensor = []
        # get the images
        batch = dataset.padded_batch(BATCH_SIZE, drop_remainder=True)
        # samples not normalized
        normalized = batch.map(lambda x: normalize_img(x['image']))
        for sample in normalized:
            prediction = model.predict(sample)
            # divide by 10 because output is in form n instead of 1\n
            tensor.append(np.sum(prediction, axis=0) / prediction.shape[0] / 10)
        # take avg and append
        sumtensors.append(np.sum(tensor, axis=0) / len(tensor))

    return sumtensors 

    
def model_create_equation(model_dir, tex_save, training_data):
    # check optional args
    from io import StringIO
    activ_obj = pandas.read_csv(StringIO(ACTIVATION_LIST), delimiter=';')

    # create prequesties
    model = tf.keras.models.load_model(model_dir)
    if model is not None:
        # calculate fft + shape
        from sage.misc.sage_eval import sage_eval
        shapes = []
        fft_layers = []
        
        def product(x):
            out = 1
            for y in x:
                out *= y
            return out 

        # append input shape remove None type
        shapes.append([product(model.input_shape[1:])])

        # main wb extraction loop
        for [weights, baises, act, shape] in [
                (layer.weights[0].numpy(), 
                    layer.weights[1].numpy(), 
                        layer.activation, layer.kernel.shape.as_list())
                          for layer in model.layers if len(layer.weights) > 1]:
            # if no activation assume linear
            if act is None:
                act = sage_eval('x',locals={"x": INPUT_SYMBOL})
            fft_layers.append([weights, baises, act])
            shapes.append([shape, weights.shape, baises.shape])
        
        targets = [1] * len(shapes)
        # add output target
        targets[-1] = model.output_shape[-1] 
        shapes = complete_bias(shapes, targets) 
        # fft calculation goes through here
        # inv_layers = output_aggregator(model, fft_layers, training_data)
        peq_system = get_poly_eqs_subst(shapes, activ_obj, fft_layers)
        evaluate_system(shapes, peq_system, tex_save, None)

if __name__=="__main__":
    if len(sys.argv) > 2:
        path = os.path.dirname(__file__)
        model = os.path.join(path, sys.argv[1]) 
        try:
            print(os.path.abspath(model))
            if len(sys.argv) > 3:
                model_create_equation(os.path.abspath(model), sys.argv[3], sys.argv[2])
            else:
                print("""not enough commands, please give a filename of a model to extract, it's training dataset (which may be altered at future date)
                output for a tex file, and a csv file containing each type of acitvation used delimitered by ; (optional)""")
        except FileNotFoundError:
            print("""file not found, 
                        please give filename of model to extract equation from""")
    else:
        print("""not enough commands, please give a filename of a model to extract, it's training dataset (which may be altered at future date)
output for a tex file, and a csv file containing each type of acitvation used delimitered by ; (optional)""")