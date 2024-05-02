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

def trig_eulerlfy(x):
    from sage.all import I, cos, sin, log
    csh = (x).apply_map(cos)
    ssh = (I*(x)).apply_map(sin)
    return (csh + ssh).apply_map(log) / I
 
def get_poly_eqs_subst(shapes, activ_obj, fft_layers):
    from numpy import empty, ndindex
    from sage.matrix.constructor import matrix
    from sage.all import vector
    from sage.all import SR
       
    # pre generates a matrix with symbols in
    # USES ONLY NUMPY ARRAYS as coeff and prev_input
    def calc_expr(coeff, prev_input, ops, exp):
        # nabbed from sympy source and edited for use case
        arr = empty(coeff.shape, dtype=object)
        arr = ops(coeff, prev_input.sum())
        # convert into trigonomentric terms so we can have linear dfft

        # if not werid tuple shaping
        if len(list(coeff.shape)) != 1:
            return matrix(SR, coeff.shape[0], coeff.shape[1], arr)
        else:
            return vector(list(arr))

    x_input = empty(shapes[0], dtype=object)
    for n, index in enumerate(ndindex(shapes[0])):
        x_input[index] = var("x_" + str(n))
    # to system 
    eq_system = [vector(x_input.flatten())]

    # summate equations to get output
    for i in range(1, len(shapes)):
        layer = fft_layers[i-1]

        # get our matrix exprs 
        # calculates dot product in creation
        from operator import __mul__, __add__
        from scipy.fft import fftn
        # because we cannot calculate coeffecients of expressions neatly we calcualate dfft here
        weight = calc_expr(fftn(layer[0], layer[0].shape), eq_system[-1].numpy(), __mul__, i)
        # power is always 1 here because bias is linear
        bias = calc_expr(fftn(layer[1], layer[1].shape), weight.numpy(), __add__, 1)
        
        # lookup the activation function 
        lookup = activation_fn_lookup(layer[2], activ_obj)
        
        # if at end of list don't apply activation fn
        if len(shapes) == i:
            eq_system.append(bias)
        else:
            eq_system.append((bias).apply_map(lambda x: lookup(x=x)))
       
    return eq_system
    
def evaluate_system(shapes, eq_system, tex_save):
    # set as a power series
    from sage.symbolic.relation import solve
    from sage.all import latex
    from sage.misc.flatten import flatten
    
    from functools import reduce
    import itertools
    # find intersects of permutations
    cmatrix = trig_eulerlfy(eq_system[-1])
    permutes = list(itertools.permutations(cmatrix))
    diffs = list(map(lambda ineq: reduce(lambda xs,x: xs - x, ineq), flatten(permutes)))

    sols = solve(flatten(list(set(diffs))), *eq_system[0])
 
    for sol in sols:
        save("out.tex",latex(sol))

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

    
def model_create_equation(model_dir, tex_save, training_data, csv):
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
        evaluate_system(shapes, peq_system, tex_save)

if __name__=="__main__":
    if len(sys.argv) > 2:
        path = os.path.dirname(__file__)
        model = os.path.join(path, sys.argv[1]) 
        try:
            print(os.path.abspath(model))
            if len(sys.argv) > 4:
                model_create_equation(os.path.abspath(model), sys.argv[3], sys.argv[2], sys.argv[4])
            else:
                model_create_equation(os.path.abspath(model), sys.argv[3], sys.argv[2], None)
        except FileNotFoundError:
            print("""file not found, 
please give filename of model to extract equation from""")
    else:
        print("""not enough commands, please give a filename of a model to extract, it's training dataset (which may be altered at future date)
output for a tex file, and a csv file containing each type of acitvation used delimitered by ; (optional)""")