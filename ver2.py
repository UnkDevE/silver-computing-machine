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
import keras as K

from scipy.fft import rfftn 
from scipy import linalg 
import numpy as np
import matplotlib.pyplot as plt

#TUNE THESE INPUT PARAMS
MAX_ITER=2048
BATCH_SIZE = 1024
TRAIN_SIZE=16
RBF_BOUND_MIN=1e-5
RBF_BOUND_MAX=1e15
NEIGHBOURS_DOT=3
GP_SCALE=0.1
GP_VAR=0.01

# tf.data.experimental.enable_debug_mode()
# do not remove forces tf.data to run eagerly
tf.config.run_functions_eagerly(True)

def product(x):
    out = 1
    for y in x:
        out *= y
    return out 

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
    reshape = np.reshape(np.arange(product(shape.shape)), shape.shape)
    pow = np.float_power(-np.ones_like(x), reshape)
    s=x * pow
    #memory leak here
    sum = np.cumsum(s)
    return np.reshape(sum, x.shape)

#rolls a 1D array
def shifts(x, start):
    y = np.zeros([x.shape[0], *x.shape])
    for i in range(-start, x.shape[0] - start):
        y[i] = np.roll(x, i)
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

        outtemplate = shifts(template, 0)
        # solve backwards
        sols = []
        for shift in outtemplate:
            # use svd with solve 
            sols.append(linalg.solve(SL, shift) * s @ inv)

        solutions.append((np.array(sols), outtemplate))

    # what next sheafify outputs?
    return solutions

def ceildiv(a, b):
    return -(a // -b)

def graph_model(model, training_data, activations, shapes, layers):
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
    solution = sols[0][0]
    outward = sols[0][0]
    for sheaf in sols[1:]:
        sheaf = sheaf[0]
        if sheaf.shape == solution.shape:
            solution = solution * sheaf
            outward = solution
        else:
            solution = solution @ sheaf.T

    sort_avg = sorted(
        output_aggregator(model, training_data), key= lambda tup: tup[0])

    # sheafifed = irfftn(solution, shapes[0]) 
    sheafifed = np.imag(rfftn(solution, shapes[0]))
    
    ret = [sheafifed, sols, outward, sort_avg]
    return ret


def interpolate_model_train(sols, model, train):
    # hang on about this 
    ins = np.array(sols[0])
    # out = np.array(sols[1])
    outshape = len(sols[1])
    out = sols[1]

    # need this for later
    model_shape = [1 if x is None else x for x in model.input_shape]
    Tout = model.predict(train.reshape([product(train.shape) // product(model_shape), *model_shape[1:]]))
    
    from itertools import batched

    quart_out = list(batched(Tout, n=(outshape * 4)))
    quart_in = list(batched(train, n=(outshape * 4)))

    first_quarts = (np.array(quart_in[0]), np.array(quart_out[0]))
    out_with_quart = np.concatenate([out, first_quarts[1]])

    from rbf.gproc import gpiso, gppoly 

    # use Gaussian basis
    priors = gpiso('ga', eps=0.5, var=1.0) 
    priors += gppoly(2)
    inout = np.stack([ins, out])
    gp_post = priors.condition(inout, out_with_quart[0]) 
    for i_in, outs in enumerate(out_with_quart[1:], start=1):
        priors += gppoly(2)
        gp_post += priors.condition(ins, outs) 

    sample = [gp_post(s_out) for sout in quart_out[1:]] 
    # inverse one hot the outputs
    onehottmp = np.reshape(np.tile(np.arange(outshape), out.shape[0]), out.shape)
    onehotout = np.reshape(onehottmp[np.max(sampleout)], out.shape[0]).reshape(-1, 1)
    # allocating a sample from classification is not possible atm
    # so we allocate a image and then classify it?
    model.fit(sample, onehotout)
    return model

def bucketize(prelims):
    arr = []
    for ps in prelims:
        for i, p in enumerate(ps):
            if len(arr) <= i:
                arr.append([])
            arr[i].append(p)
    return np.array(arr)

def tester(model, sheafout, sheafs, sort_avg):
    model_shape = [1 if x is None else x for x in model.input_shape]
    out = np.reshape(sheafout, model_shape) 
    final_test = model(out)

    prelim_shape = model_shape
    prelim_shape[0] *= sheafs.shape[0]
    prelimin = np.reshape(sheafs, prelim_shape)
    prelims = model.predict(prelimin)
    
    avg_outs = [avg for (_, (_, avg)) in sort_avg]
    return [avg_outs, bucketize(prelims), final_test.numpy()]

def plot_test(starttest, endtest, outshape, name):
    tests = [starttest, endtest] 
    plt.xlabel("features")
    plt.ylabel("activation")

    colours = ["ro--", "bo--"]

    for i, [avg_outs, prelims, final_test] in enumerate(tests):
        template = np.reshape(np.arange(1, len(avg_outs)+1), outshape[-1])
        # plot our test
        plt.violinplot(np.transpose(avg_outs), showmeans=True)
        plt.violinplot(np.transpose(prelims), showmeans=True) 
        plt.plot(template, np.transpose(final_test), colours[i])

    plt.savefig(name)
    # clear figures and axes
    plt.cla()
    plt.clf()

def model_create_equation(model_dir, training_data):
    # check optional args
    # from io import StringIO
    # activ_obj = pandas.read_csv(StringIO(ACTIVATION_LIST), delimiter=';')

    # create prequesties
    model = tf.keras.models.load_model(model_dir)
    if model is not None:
        # calculate fft + shap
        shapes = []
        layers = []
        
        # append input shape remove None typp/e
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

        [sheaf, sols, outward, sort_avg] = graph_model(model, training_data, activations, shapes, layers)        
        control = tester(model, sheaf, outward, sort_avg)

        for i in range(TRAIN_SIZE):
            model = interpolate_model_train(sols[-1], model, outward)
            [sheaf, sols, outward, sort_avg] = graph_model(model, training_data, activations, shapes, layers)        
            test = tester(model, sheaf, outward, sort_avg)
            plot_test(control, test, shapes[-1], "out-epoch-"+str(i)+".png")


if __name__=="__main__":
    if len(sys.argv) > 2:
        path = os.path.dirname(__file__)
        model = os.path.join(path, sys.argv[1]) 
        try:
            print(os.path.abspath(model))
            if len(sys.argv) > 2:
                model_create_equation(os.path.abspath(model), sys.argv[2])
            else:
                print("""not enough commands, please give a filename of a model to extract, it's training dataset (which may be altered at future date)
                output for a tex file, and a csv file containing each type of acitvation used delimitered by ; (optional)""")
        except FileNotFoundError:
            print("""file not found, 
                        please give filename of model to extract equation from""")
    else:
        print("""not enough commands, please give a filename of a model to extract, it's training dataset (which may be altered at future date)
output for a tex file, and a csv file containing each type of acitvation used delimitered by ; (optional)""")
