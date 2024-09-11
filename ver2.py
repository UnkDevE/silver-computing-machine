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


from sympy import symbols
from scipy import linalg 
import numpy as np
# import pandas

"""
# list of activation functions
ACTIVATION_LIST = activation;function;
linear;x;
step;Heaviside(x)
logistic;1/(1+E^-x);
tanh;E^x - E^-x/E^x+E^-x;
smht;E^ax - E^-bx/E^cx + E^-dx;
relu;maximum(0, x);
softplus;ln(1+E^x);
"""

INPUT_SYMBOL = symbols("x")
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
    [dataset, test] = tdfs.load(data, download=False, split=['train[:100%]', 'test'])
    
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
    for dataset in sets:
        # get the images
        batch = dataset.padded_batch(BATCH_SIZE, drop_remainder=True)
        # samples not normalized
        normalized = batch.map(lambda x: normalize_img(x['image']))
        for sample in normalized:
            inputimages.append(sample)
            prediction = model.predict(sample)
            # divide by 10 because output is in form n instead of 1\n
            tensors.append(np.sum(prediction, axis=0) / prediction.shape[0] / 10)

    return list(zip(labels, zip(inputimages, tensors)))

# image in linear algebra
def image(rref):
    # now column reduced echeolon
    cref = rref.transpose()
    cstack = []
    # we flip here as we want the bottom half of the matrix up to the zero 
    # row reducing compute
    for row in np.flip(cref):
        # split by zero row
        if np.all(row == np.zeros_like(cref[0]).astype(np.float64)):
            return np.stack(cstack)
        cstack.append(row)
    return np.stack(cstack) 

# calculates the chec differential
def _chec_diff(x, start):
    shape = np.roll(np.reshape(np.meshgrid(x)[0], x.shape), start)
    return np.reshape(np.cumsum(x * (-np.ones_like(x)**shape)), x.shape)

#rolls a 1D array
def shifts(x, start):
    y = []
    for i in range(-start, x.shape[0] - start):
        y.append(np.roll(x, i))
    return y

    
def chec_diff(x, start):
    permutes = shifts(x, start)

    diffs = []
    for i, perm in enumerate(permutes):
        diffs.append(_chec_diff(perm, i))

    return np.array(diffs)

"""
def pad_coeff_output(coeff):
    last = 0
    arr = []
    for [x, i] in coeff:
        if i-1 >= last:
            [arr.append(0) for _ in range(i-1)]
        arr.append(x)
        last = i

    return np.array(arr)
"""
# start finding chec cohomology from sheafs
# simplex / cocycle in this case is the direct sum
# compute kernels, images and chec cohomologies
# gets the kth layers kth Chomology class
def chec_chomology(layer):
    diff = chec_diff(layer, 0)
    cocycle = chec_diff(layer, 1)
    _, _, rrefco = linalg.lu(cocycle)
    coboundary = image(rrefco)

    # LDU we want U (reduced row echelon form)
    _, _, rref = linalg.lu(diff)
    simplex = image(rref.transpose())

    # calculate chomologies
    return (simplex, coboundary) 


def sortshape(a):
    last = (0,0)
    shapes = list(a.shape)
    for (i, sh) in enumerate(shapes):
        if sh >= last[1]:
            a = np.swapaxes(a, -i, last[0])
            last = (i, sh)
    return a
    
# where space is the space in matrix format
# plus the subset of spaces
def quot_space(subset, space):
    # find commonality shape and set them
    subset = sortshape(subset)

    # set solve subset for zero 
    zerosub = []
    for subs in subset:
        zerosub.append(linalg.solve(subs, np.ones(subs.shape[0])))
    zerosub = np.array(zerosub)

    # sheafify with irfftn to find quotient
    quot = []
    zSl, zs, ZSr = linalg.svd(zerosub)
    diag = linalg.diagsvd(zs, zerosub.shape[0], zerosub.shape[1])

    for sp in space:
        # use svd here
        SL, s, SR = linalg.svd(sp)

        # compose both matricies 
        new_diag = linalg.diagsvd(s, sp.shape[0], sp.shape[1]) @ diag
        inputbasis = (zSl * SR) @ diag
        orthsout = SL @ new_diag @ ZSr 

        quot.append(inputbasis @ orthsout.T)
    
    quot = np.sum(np.array(quot), axis = 0)
    return quot

def cohomologies(layers):
    # find chomologies
    cohol = []
    kerims = []

    # layer is normally nonsquare so we vectorize
    for funclayer in layers:
        kerims.append(chec_chomology(funclayer))
        if len(kerims) >= 2:
           cohol.append(quot_space(kerims[-1][0], kerims[-2][1]))

    # append R space
    start = [quot_space(kerims[0][1], np.ones_like(kerims[0][1]))]
    # don't forget append in start in reverse!
    [start.append(c) for c in cohol]

    return start

def solve_system(activations, layers):
    # first dtft makes our system linear, however it's already in this form 
    # so we can ignore our transform until we finish computation

    # now we modify the differential to apply for matricies
    # so for this we simply mutliple 

    # This for loop below caclualtes all the terms lienarly so we want to add 
    # in the non linear function compositions in a liner manner
    # we want to *not* use inputs as it gums up the works
    zetas = []
    baises = []
    for i, (layer,act) in enumerate(zip(layers, activations)):
        # add in nonlinear operator 
        # take power to remove nonlinears
        # weight = np.float_power(layer[0], float(i+1))
        # bias = np.float_power(layer[1], float(i+1))
        weight = layer[0]
        bias = layer[1] # still gives zero
        # TODO: recovery of inputs through activaiton?
        # vectorize activation
        inv = np.vectorize(lambda x: act(x=x))

        # recreate zeta
        if len(zetas) < 1:
            zetas.append(weight)
        else:
            zetas.append(inv(zetas[-1] + baises[-1]).astype(np.float64) @ weight)
        baises.append(bias)

    zetas.append(zetas[-1] + baises[-1])
    # run cohomologies
    return cohomologies(zetas)

def create_sols_from_system(solved_system):
    solutions = []
    for system in solved_system:
        outdim = system.shape[-1]

        # create output template to be rolled
        template = np.zeros(outdim)
        template[0] = 1 
        inv = linalg.pinv(system)

        SL, s, _ = linalg.svd(inv)

        # solve backwards
        sols = []
        for shift in shifts(template, 0):
            # use svd with solve 
            sols.append(linalg.solve(SL, shift) * s @ inv)

        solutions.append(np.array(sols))

    # what next sheafify outputs?
    return solutions

def model_create_equation(model_dir, tex_save, training_data):
    # check optional args
    # from io import StringIO
    # activ_obj = pandas.read_csv(StringIO(ACTIVATION_LIST), delimiter=';')

    # create prequesties
    model = tf.keras.models.load_model(model_dir)
    if model is not None:
        # calculate fft + shap
        shapes = []
        layers = []
        
        def product(x):
            out = 1
            for y in x:
                out *= y
            return out 
        
        # append input shape remove None type
        shapes.append([product(model.input_shape[1:])])
        activations = []

        # main wb extraction loop
        for [weights, baises, act, shape] in [
                (layer.weights[0].numpy(), 
                    layer.weights[1].numpy(), 
                        layer.activation, layer.kernel.shape.as_list())
                          for layer in model.layers if len(layer.weights) > 1]:

            # if no activation assume linear
            activations.append(lambda x: x if act is None else act(x))
            layers.append([weights, baises]) 
            shapes.append([shape, weights.shape, baises.shape])
        
        targets = [1] * len(shapes)
        # add output target
        targets[-1] = model.output_shape[-1] 
        shapes = complete_bias(shapes, targets) 


        # fft calculation goes through here
        solved_system = solve_system(activations, layers)
        # lets add the input vector
        # create solutions to output
        sols = create_sols_from_system(solved_system)
        # convert from matrix

        # sheafify 
        solution = sols[0] 
        outward = sols[0]
        for sheaf in sols[1:]:
            if sheaf.shape == solution.shape:
                solution = solution * sheaf
                outward = solution
            else:
                solution = solution @ sheaf.T

        sort_avg = sorted(
            output_aggregator(model, training_data), key= lambda tup: tup[0])

        from scipy.fft import irfftn, rfftn
        # sheafifed = irfftn(solution, shapes[0]) 
        sheafifed = np.imag(rfftn(solution, shapes[0]))

        tester(model, shapes[-1], sheafifed, outward, sort_avg)

def tester(model, outshape, sheafout, sheafs, sort_avg):
    model_shape = [1 if x is None else x for x in model.input_shape]
    out = np.reshape(sheafout, model_shape) 
    final_test = model(out)

    prelim_shape = model_shape
    prelim_shape[0] *= sheafs.shape[0]
    prelimin = np.reshape(sheafs, prelim_shape)
    prelims = []
    prelims.append(model.predict(prelimin))
    # aggregate outputs
    # and sort by feature
    avg_outs = [avg for (_, (_, avg)) in sort_avg]

    template = np.reshape(np.arange(0, len(avg_outs)), outshape[-1])
    import matplotlib.pyplot as plt
    # plt.plot(prelims) 
    [plt.plot(template, np.transpose(prelim), "g-") for prelim in prelims]
    [plt.plot(template, np.transpose(avo), "bo-") for avo in avg_outs]
    plt.plot(template, np.transpose(final_test.numpy()), "ro--")

    plt.savefig("out.png")

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