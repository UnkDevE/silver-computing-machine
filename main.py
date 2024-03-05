import tensorflow as tf
from tensorflow import keras
import tensorflow_datasets as tdfs

import numpy as np
import sympy as syp
import pandas

import sys
import os
import inspect

BATCH_SIZE = 1024

tf.config.run_functions_eagerly(True)
# do not remove forces tf.data to run eagerly
# tf.data.experimental.enable_debug_mode()

# helper 
def complete_bias(shapes):
    shapes_new = []
    for (i, shape) in enumerate(shapes):
        shapes_new.append([shape[0]])
        for j in range(1, len(shape)):
            tup = shape[j]
            if len(tup) < 2:
                tup = (tup[0], 1)
            shapes_new[i].append(tup)

    return shapes_new

# setup symbols for computation
def get_poly_eqs(shapes):
    activation = syp.MatrixSymbol("x", shapes[0][0][0], shapes[0][0][1])
    activation_fn = syp.Function("sigma")

    # create neruonal activation equations
    eq_system = [activation]

    # summate equations to get output
    from sympy.matrices import expressions
    from sympy.vector import matrix_to_vector
    
    for i in range(1, len(shapes)):
        shape = shapes[i]
        bias = syp.MatrixSymbol("b_"+ str(i), shape[2][0], shape[2][1])
        weight = syp.MatrixSymbol("w_" + str(i), shape[1][0], shape[1][1])

        # expand vector to matrix cirumnavigating the vector module
        # which is not useful here
        alpha = eq_system[-1] * weight
        # rows are default for each list so we have to transpose
        biasmatrix = syp.Matrix(
            np.repeat(bias, repeats=alpha.rows, axis=1)).transpose()

        eq_system.append(activation_fn(alpha + biasmatrix))

    return eq_system

def activation_fn_lookup(activ_src, csv):
    sourcestr = inspect.getsource(activ_src)
    for (i, acc) in enumerate(csv['activation']):
        if acc in sourcestr:
            return csv['function'][i]
        else:
            pass
    return None

def subst_into_system(fft_layers, eq_system, activ_obj, shapes):
    input_symbol = syp.MatrixSymbol("x", shapes[0][0][0], shapes[0][0][1]) 
    activation_fn = syp.Function("sigma")

    for (i, layer) in enumerate(fft_layers):
        shape = shapes[i]
        # for each layer create seperate symbol
        weight_symbol = syp.MatrixSymbol("w_" + str(i), 
           shape[1][0], shape[1][1])
        bias_symbol = syp.MatrixSymbol("b_" + str(i), 
            shape[2][0], shape[2][1])

        for system in eq_system[1:]:
            system.subs(weight_symbol, syp.Matrix(layer[0]))
            system.subs(bias_symbol, 
                syp.Matrix(np.repeat(layer[1], repeats=shape[0][0], axis=0)))
            system.subs(activation_fn, activation_fn_lookup(layer[2],
                                 activ_obj))
    
    return eq_system

# evaluate irfftn using cauchy residue theorem
def evaluate_system(eq_system, fft_inverse, tex_save):
   # inverse of fourier transform is anaglogous to convergence of fourier series
   from sympy import fourier_series, solve, latex, sympify 
   
   # feed backwards the output
   equations = []
   # convert from numpy
   inverse = syp.Matrix(fft_inverse)
   # the equation is then solved backwards
   equate = syp.Eq(eq_system[-1], inverse)
   solved = syp.solve(equate)

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
                        labels.append(label[feature])
        return labels
    
    labels = label_extract(need_extract, features)

    # have to update cardinality each time
    # remember labels is a list due to extract
    length = len(labels)

    @tf.function
    def condense(v):
        b = False
        for i in v:
            if not i: 
                b = True
        return b
    
    # condense doesn't work as complains about python bool 
    auto_condense = tf.autograph.to_graph(condense.python_function)

    # bucketize each feature in each label, return complete datapoints 
    sets = [dataset.filter(lambda i: 
            auto_condense([i[feature] == label for feature in features])) 
            for label in labels]

    # normalize function for images
    @tf.function
    def normalize_img(image):
      return tf.cast(image, tf.float32) / 255.

    # numpy array of predictions
    sumtensors = []
    for dataset in sets:
        # get the images
        batch = dataset.padded_batch(BATCH_SIZE, drop_remainder=True)
        # samples not normalized
        normalized = batch.map(lambda x: normalize_img(x['image']))
        for sample in normalized:
            prediction = model.predict(sample)
            sumtensors.append(prediction)
        
    # normalize sumtensor, use whole training data so len(dataset)
    sumtensor = np.sum(sumtensors, axis=(1)) / db_len 
    # get only real counterpart (::2) because no complex parts here
    return np.fft.ifftn(sumtensor)[::2]
    
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
        from sympy import parse_expr
        shapes = []
        fft_layers = []
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

        shapes = complete_bias(shapes) 
        peq_system = get_poly_eqs(shapes)
        # fft calculation goes through here
        inv_layers = output_aggregator(model, fft_layers, training_data)
        peq_system = subst_into_system(fft_layers, peq_system, activ_obj, shapes)
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


