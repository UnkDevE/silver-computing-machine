import tensorflow as tf
from tensorflow import keras 
import tensorflow_datasets as tdfs 

import numpy as np
import scipy as sci
import sympy as syp
import pandas

import sys
import os
import inspect

BATCH_SIZE = 1024

tf.config.run_functions_eagerly(True)
# do not remove forces tf.data to run eagerly
# tf.data.experimental.enable_debug_mode()

# hang fire with convolutions
# helper 
def complete_bias(shapes, targets):
    tup_complete = lambda tup, tar: (tup[0], tar) if len(tup) < 2 else tup

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
    from sympy import Lambda
    in_sym = syp.Symbol("x")
    sourcestr = inspect.getsource(activ_src)
    for (i, acc) in enumerate(csv['activation']):
        if acc in sourcestr:
            return Lambda(in_sym, csv['function'][i])
    return Lambda(in_sym, "x=x") 

# setup symbols for computation
def get_poly_eqs_subst(shapes, activ_obj, fft_layers):
    # add start
    activation = syp.MatrixSymbol("x", shapes[0][0], shapes[0][1])
    activation_fn = syp.Function("s")

    # create neruonal activation equations
    eq_system = [activation]

    # summate equations to get output
    from sympy.matrices import expressions
    
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

    from sympy.matrices.expressions import hadamard_product
    from sympy.matrices.expressions import FunctionMatrix 
    for i in range(1, len(shapes)):
        layer = fft_layers[i-1]
        shape = shapes[i]
        # 0 idx is IN shape
        bias = syp.MatrixSymbol("b_"+ str(i), shape[2][0], shape[2][1])
        # print(bias.shape)
        weight = syp.MatrixSymbol("w_" + str(i), shape[1][0], shape[1][1])

        inner = vecmul(eq_system[-1], shape[0][1], False)

        # because we've made inner the same size dot doesn't work here
        # use hadamard if same size and then summate 
        # to get inner product
        # must have doit in here as to access columns maybe as_explicit works?
        # it does :D
        
        alpha = weight.transpose() @ eq_system[-1] 
        Afn_matrix = FunctionMatrix(alpha.shape[0], alpha.shape[1], activation_fn)
        # set transpose to true for biases
        if len(shapes) == i:
            eq_system.append((alpha + bias))
        else:
            eq_system.append((alpha + bias).func(Afn_matrix))
    
        lookup = activation_fn_lookup(layer[2], activ_obj)
        Afn_matrix.subs(activation_fn, lookup)
        eq_system[-1].subs(weight, syp.Matrix(layer[0]))
        eq_system[-1].subs(bias, syp.Matrix(layer[1]))
        
    return eq_system

# evaluate irfftn using cauchy residue theorem
def evaluate_system(eq_system, fft_inverse, tex_save):
   # inverse of fourier transform is anaglogous to convergence of fourier series
   from sympy import fourier_series, solve, latex, sympify 
   
   # set as polynomials
   # needs to be the same 
   inverse_series = syp.Matrix(np.array(fft_inverse))
   inverse_poly = inverse_series.charpoly()

   # this gives us N output neruons and N output equations in a system 
   # however we need one equation so how do we do this?
   eq_poly = eq_system[-1].as_explicit()

   equate = syp.Eq(eq_poly, inverse_poly)
   solved = syp.solve(equate).doit(deep=True)

   tex_save = latex(solved)
   file = open("out.tex", "xt")
   file.write(tex_save)
   file.close()

   print("eq solved")


def output_aggregator(model, fft_layers, data):
    from functools import reduce
    # load and get unique features
    [dataset, test] = tdfs.load(data, download=False, split=['train', 'test'])
    
    # get label and value
    value, *features = list(list(dataset.take(1).as_numpy_iterator())[0].keys())

    # this is our optimized len fn
    @tf.function
    def len_ds(ds):
        length_np = 0
        for _ in ds.map(lambda x: 1, 
                num_parallel_calls=tf.data.AUTOTUNE, deterministic=False):
            length_np += 1
        return tf.cast(length_np, tf.int64)
    
    len_ds_auto = tf.autograph.to_graph(len_ds.python_function)

    db_len = len(dataset)

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
    length = len(labels)

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

    len_db = [len_ds(data) for data in sets]  
    
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
            tensor.append(prediction)
        sumtensors.append(tensor)

    # helper function to convolve vstack
    def convolute_signal(tensors, mode):
        # use valid so that they overlap completely and our shape is scaled up 
        conv = sci.ndimage.convolve(tensors[0], tensors[1], mode=mode)
        for (i, tensor) in enumerate(tensors[2:]):
            if i % 2 == 0:
                conv = sci.ndimage.convolve(tensor, conv, mode=mode)
            else:
                conv = sci.ndimage.convolve(conv, tensor, mode=mode)
        return conv       

    # convolve the features
    supertensor = np.vstack(sumtensors)
    super_conv = convolute_signal(supertensor.transpose(), 'mirror')

    # then inverse fourier transform
    ifftsor = sci.fft.irfft(super_conv, n=len(super_conv))
    return ifftsor 

    
def model_create_equation(model_dir, tex_save, training_data, csv):
    # check optional args
    if csv == None:
        path = os.path.dirname(__file__)
        csv = os.path.join(path, "./activationlist.csv")

    activ_obj = pandas.read_csv(csv, delimiter=';')

    # create prequesties
    model = tf.keras.models.load_model(model_dir)
    if model != None:
        # calculate fft + shape
        from keras import backend as K
        from sympy.parsing.sympy_parser import parse_expr
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
                        layer.activation, layer.kernel.shape)
                          for layer in model.layers if len(layer.weights) > 1]:
            # if no activation assume linear
            if act == None:
                act = parse_expr("x=x") 
            fft_layers.append([weights, baises, act])
            shapes.append([shape, weights.shape, baises.shape])
        
        targets = [1] * len(shapes)
        # add output target
        targets[-1] = model.output_shape[-1] 
        shapes = complete_bias(shapes, targets) 
        # fft calculation goes through here
        inv_layers = output_aggregator(model, fft_layers, training_data)
        peq_system = get_poly_eqs_subst(shapes, activ_obj, fft_layers)
        evaluate_system(peq_system, inv_layers, tex_save)

if __name__=="__main__":
    if len(sys.argv) > 2:
        path = os.path.dirname(__file__)
        try:
            model = os.path.join(path, sys.argv[1])
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