import tensorflow as tf
from tensorflow import keras
import tensorflow_datasets as tdfs

import numpy as np
import sympy as syp
import pandas

import sys
import os
import inspect


tf.config.run_functions_eagerly(True)
tf.data.experimental.enable_debug_mode()

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
    values_removed = dataset.map(lambda i: rm_val(i, [value]),
        num_parallel_calls=tf.data.AUTOTUNE, deterministic=False)

    # call unqiue on features
    need_extract = values_removed.unique() 

    @tf.function
    def label_extract(label_extract):
        labels = []
        for label in label_extract: 
            if hasattr(label['label'], 'numpy') and callable(
                getattr(label['label'], 'numpy')):
                    labels.append(label["label"])
        return labels
    
    extract_auto = tf.autograph.to_graph(label_extract.python_function)
    labels = extract_auto(need_extract)

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


    # assert length of features 
    # TODO: this throws error -> fix!
    len_db = [len_ds_auto(data) for data in sets]
     
    # set the outer as length
    # convert to np float32 (single) using .numpy causes int32 - not good

    # numpy array of predictions
    # sum_len fraction
    sumtensors = []
    for (d_len, dataset) in zip(len_db, sets):
        # get the images
        samples = dataset.batch(tf.cast(d_len / length, tf.int64))
        for sample in samples:
            prediction = model.predict(sample['image'])
            #  correct = [0.0 for _ in range(length)]
            #  correct[samples['label'].numpy()] = 1.0
            sumtensors.append(prediction)
        
    # normalize sumtensor, use whole training data so len(dataset)
    sumtensor = np.sum(sumtensors) / db_len
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


