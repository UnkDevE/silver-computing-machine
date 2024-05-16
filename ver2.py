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
from sage.all import heaviside, var

import numpy as np
import pandas

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

# looks up each activation from csv and then defines a function to it 
def activation_fn_lookup(activ_src, csv):
    if activ_src is None:
        return csv['function']['linear']
    sourcefnc = activ_src.__name__
    for (i, acc) in enumerate(csv['activation']):
        if acc == sourcefnc:
            return (csv['function'][i])
    return csv['function']['linear']

def output_aggregator(model, data):
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
    inputimages = []
    tensors = []
    for (i, dataset) in enumerate(sets):
        # get the images
        batch = dataset.padded_batch(BATCH_SIZE, drop_remainder=True)
        # samples not normalized
        normalized = batch.map(lambda x: normalize_img(x['image']))
        for sample in normalized:
            inputimages.append(sample)
            prediction = model.predict(sample)
            # divide by 10 because output is in form n instead of 1\n
            tensors.append(np.sum(prediction, axis=0) / prediction.shape[0] / 10)

    return list(zip(inputimages, tensors)) 

# kernel in linear algebra
def kernel(rref):
    # now column reduced echeolon
    cref = rref.transpose()
    cstack = []
    # we flip here as we want the bottom half of the matrix up to the zero 
    # row reducing compute
    for row in np.flip(cref):
        # split by zero row
        if row == np.zeros_like(cref)[0]:
            return np.hstack(cstack)
        cstack.append(row)
    return np.hstack(cstack) 

# calculates the chec differential
def chec_diff(x, start):
    return np.cumsum(x * (-np.ones_like(x)**np.mgrid[start:len()]))
    
# start finding chec cohomology from sheafs
# simplex / cocycle in this case is the direct sum
# compute kernels, images and chec cohomologies
# gets the kth layers kth Chomology class
def chec_chomology(layer):
    from scipy import linalg

    diff = chec_diff(layer, 0)
    cocycle = chec_diff(layer, 1)
    image = linalg.solve(cocycle, np.zeros_like(cocycle)[0])

    # LDU we want U (reduced row echelon form)
    _, _, rref = linalg.lu(linalg.solve(diff[-1], np.zeros_like(layer)))
    ker = kernel(rref)

    # calculate chomologies
    return ker * linalg.inv(image) 

 
def solve_system(shapes, acitvations, layers, solutions):
    # first dtft makes our system linear, however it's already in this form 
    # so we can ignore our transform until we finish computation

    # now we modify the differential to apply for matricies
    # so for this we simply mutliple 

    # This for loop below caclualtes all the terms lienarly so we want to add 
    # in the non linear function compositions in a liner manner
    # we do this by finding the taylor expansion of the function
    def taylor2mat(taylor):
        cfs = taylor.coefficients()
        # fill in gaps of coefficients
        mat = []
        for (cff, ipow), i in enumerate(cfs):
            zcf = []
            if ipow != i:
                zcf = [0 for _ in ipow - i]
            mat.append(*zcf)
            mat.append(cff)

        return np.array(mat)
 
    tayloract = [taylor2mat(act.taylor(INPUT_SYMBOL, 0, len(layers))) for act in acitvations]

    cech = []
    lacts = []    
    for layer,act in zip(layers, tayloract):
        # create the multiplicants of powers in taylor series
        lactpow = np.stack([layer * xn for xn in act])
        # add each power to right power bukcket
        # creates a line for input so we can use it for chec
        # use pythonic sum to sum matricies
        lacts.append(sum([layer[i] + lpow for i, lpow in enumerate(lactpow)]))

        # find chomologies
        cohol = []
        for funclayer in lacts:
            cohol.append(chec_chomology(funclayer))

        # append direct sum of power matricies
        cech.append(np.sum(cohol, axis=len(cohol)-1))
    
    # TODO: compose chomologies of R* 
    # TODO: get out of fourier domain using inverse transform
    # TODO: UNIT tests

    # next part is to direct sum all comolohogies 
    # direct_sum = map(np.sum, cohomologies) 
    pass


def model_create_equation(model_dir, tex_save, training_data):
    # check optional args
    from io import StringIO
    activ_obj = pandas.read_csv(StringIO(ACTIVATION_LIST), delimiter=';')

    # create prequesties
    model = tf.keras.models.load_model(model_dir)
    if model is not None:
        # calculate fft + shape
        shapes = []
        layers = []
        
        def product(x):
            out = 1
            for y in x:
                out *= y
            return out 
        
        from sage.misc.sage_eval import sage_eval

        # append input shape remove None type
        shapes.append([product(model.input_shape[1:])])

        # main wb extraction loop
        for [weights, baises, act, shape] in [
                (layer.weights[0].numpy(), 
                    layer.weights[1].numpy(), 
                        layer.activation, layer.kernel.shape.as_list())
                          for layer in model.layers if len(layer.weights) > 1]:

            # if no activation assume linear
            act = sage_eval(activation_fn_lookup(act), locals={'x': INPUT_SYMBOL})

            layers.append([weights, baises, act])
            shapes.append([shape, weights.shape, baises.shape])
        
        targets = [1] * len(shapes)
        # add output target
        targets[-1] = model.output_shape[-1] 
        shapes = complete_bias(shapes, targets) 
        # fft calculation goes through here
        solutions = output_aggregator(model, training_data)
        solved_system = solve_system(shapes, activ_obj, layers, solutions)
        # evaluate_system(shapes, peq_system, tex_save, None)

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