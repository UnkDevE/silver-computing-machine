import tensorflow as tf
from tensorflow import keras
import numpy as np
import sympy as syp
import sympy
import pandas

import sys
import os
import inspect

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

def act_lookup_read(activ_file):
    csv = pandas.read_csv(activ_file, delimiter=';')
    csv[0] 
    pass

def activation_fn_lookup(activ_obj):
    sourcestr = inspect.getsource(activ_obj)
    
    pass

def subst_into_system(fft_layers, eq_system):
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
                system.subs(activation_fn, activation_fn_lookup(fft_layers[i][1]))
    
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

def model_create_equation(model_dir, tex_save, csv):
    # check optional args
    if csv == None:
        path = os.path.dirname(__file__)
        csv = os.path.join(path, "./activationlist.csv")

    # create prequesties
    model = tf.keras.models.load_model(model_dir)
    if model != None:
        peq_system = get_poly_eqs(len(model.layers))
        
        # calculate fft
        from scipy import fft as fft
        from keras import backend as K
        from sympy import parse_expr
        fft_layers = []
        nparr = []
        for [wb_layer, act] in [
                ([layer.weights[0], layer.weights[1]], layer.activation)
                          for layer in model.layers if len(layer.weights) > 1]:
            # fft for interpolation or finding the polynomial
            # fft == inverse laplace.
            nparr.append(wb_layer)
            if act != None:
                fft_layers.append([wb_layer, act])
            else: 
                fft_layers.append([wb_layer, parse_expr("x=x")])

        peq_system = subst_into_system(fft_layers, peq_system)
        fft_inverse = fft.irfftn(np.array(nparr))
        result = evaluate_system(peq_system, fft_inverse, tex_save)

if __name__=="__main__":
    if len(sys.argv) > 2:
        path = os.path.dirname(__file__)
        try:
            model = os.path.join(path, sys.argv[1])
            if len(sys.argv) > 3:
                model_create_equation(os.path.abspath(model), sys.argv[2], sys.argv[3])
            else:
                model_create_equation(os.path.abspath(model), sys.argv[2], None)
        except FileNotFoundError:
            print("""file not found, 
please give filename of model to extract equation from""")
    else:
        print("""not enough commands, please give a filename of a model to extract, 
output for a tex file, and a csv file containing each type of acitvation used delimitered by ; (optional)""")


