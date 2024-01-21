import tensorflow as tf
from tensorflow import keras
import tensorflow.experimental.numpy as tnp
import tensorflow.compat.v1 as tf1
import tensorflow_datasets as tdfs

import numpy as np
import sympy as syp
import pandas

import sys
import os
import inspect

# load numpy and tensorflow interop
tnp.experimental_enable_numpy_behavior(dtype_conversion_mode="all")

# setup symbols for computation
weight= syp.Symbol("w_0")
activation = syp.Symbol("a")
bias = syp.Symbol("b_0")
output = syp.Symbol("y")
activation_fn = syp.Function("sigma")

def get_poly_eqs(layers_total):
    # create neruonal activation equations
    staring_eq = activation_fn((weight * activation) + bias)
    eq_system = [staring_eq]

    # summate equations to get output
    for i in range(1, layers_total):
        eq_system.append(activation_fn(syp.Symbol("w_" + str(i)) * eq_system[-1]) + bias)

    return eq_system

def activation_fn_lookup(activ_src, csv):
    sourcestr = inspect.getsource(activ_src)
    for (i, acc) in enumerate(csv['activation']):
        if acc in sourcestr:
            return csv['function'][i]
        else:
            pass
    return None

def subst_into_system(fft_layers, eq_system, activ_obj):
    # skip input
    for i in range(0, len(fft_layers)):
        for wb in range(0, len(fft_layers[i][0])):
            print(fft_layers[i][0][wb])
            weight_symbol = syp.Symbol("w_" + str(wb))
            bias_symbol = syp.Symbol("b_" + str(wb))

            for system in eq_system:
                # fft_layers[n][0] is weights whilst 1 is baises
                system.subs(weight_symbol, fft_layers[i][0][0][0][wb])
                system.subs(bias_symbol, fft_layers[i][0][1][wb])
                system.subs(activation_fn, activation_fn_lookup(fft_layers[i][1],
                                                                 activ_obj))
    
    return np.array(eq_system)

# evaluate irfftn using cauchy residue theorem
def evaluate_system(eq_system, fft_inverse, tex_save):
   feedfoward_system = eq_system[len(eq_system) - 1] # last eq is the output
   result = None
   # inverse of fourier transform is anaglogous to convergence of fourier series
   from sympy import fourier_series, solve, latex
   series = fourier_series(feedfoward_system, limits=(weight, 0, 1), finite=True).doit(deep=True)

   # normalize sum
   equation = syp.Eq(series, fft_inverse.sum() / len(fft_inverse))
   print("eq evaluated")

   file = open(tex_save, "xt")
   file.write(latex(solve(equation)))
   file.close()

   print("eq solved")
   return result

#naughty monoid hack
def condense_bool_list(list):
    for x in list:
        if x is False:
            return False
    return True

def output_aggregator(model, fft_layers, data):
    from functools import reduce
    # load and get unique features
    [dataset, test] = tdfs.load(data, download=False, split=['train', 'test'])
    # unbatch_ds = dataset.unbatch()
    
    # get label and value
    value, *features = list(list(dataset.take(1).as_numpy_iterator())[0].keys())
    
    # get types of labels
    # reconstruct featuresets from testset
    # don't forget returns tuple of state and filter
    feature_sets = dataset.map(lambda elem: 
        dataset.filter(lambda f:
            condense_bool_list(
                [not tf.math.equal(f[l], elem[l]) for l in features])))

    
    sumtensors = [] 
    # for each label sample in dataset, take 1 each from sample
    samples = tf.data.Dataset.sample_from_datasets(feature_sets).batch(1)
    # convert into numpy so we can manipulate array
    for sample in samples:
        sumtensors.append(model.predict(sample[value]))

    #normalize sumtensor
    sumtensor = sum(sumtensors) / len(feature_sets)
    return np.fft.ifftn(sumtensor, sumtensor.shape())
    
def model_create_equation(model_dir, tex_save, training_data, csv):
    # check optional args
    if csv == None:
        path = os.path.dirname(__file__)
        csv = os.path.join(path, "./activationlist.csv")

    activ_obj = pandas.read_csv(csv, delimiter=';')

    # create prequesties
    model = tf.keras.models.load_model(model_dir)
    if model != None:
        peq_system = get_poly_eqs(len(model.layers))
        
        # calculate fft
        from keras import backend as K
        from sympy import parse_expr
        fft_layers = []
        for [wb_layer, act] in [
                ([layer.weights[0], layer.weights[1]], layer.activation)
                          for layer in model.layers if len(layer.weights) > 1]:
            # if no activation assume linear
            if act == None:
                act = parse_expr("x=x") 
            fft_layers.append([wb_layer, act])

        # fft calculation goes through here
        inv_layers = output_aggregator(model, fft_layers, training_data)
        peq_system = subst_into_system(fft_layers, peq_system, activ_obj)
        result = evaluate_system(peq_system, inv_layers, tex_save)

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


