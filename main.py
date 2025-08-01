#!/bin/python
"""
    SILVER-COMPUTING-MACHINE converts Nerual nets into human readable code
    or maths
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

from scipy.fft import rfftn
from scipy import linalg
import numpy as np
import matplotlib.pyplot as plt

# TUNE THESE INPUT PARAMS
MAX_ITER = 2048
BATCH_SIZE = 1024
TEST_ROUNDS = 1
TRAIN_SIZE = 1
GP_SCALE = 0.1
GP_VAR = 0.01

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

    if os.path.isfile(filename):
        os.remove(filename)

    with open(filename, "xt") as file:
        file.write(write)


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


# normalize function for images
@tf.function
def normalize_img(image):
    return tf.cast(image, tf.float32) / 255.


@tf.function
def label_extract(label_extract, features):
    labels = []
    for feature in features:
        for label in label_extract:
            if hasattr(label[feature], 'numpy') and callable(getattr(
                    label[feature], 'numpy')):
                labels.append(label[feature].numpy())
    return labels


def output_aggregator(model, dataset):
    # get label and value
    value, *features = list(list(
        dataset.take(1).as_numpy_iterator())[0].keys())

    # get types of labels
    # dataset.unique() doesn't work with uint8
    # so we remove the offending key and use it
    def rm_val(d, val):
        for v in val:
            if v in d:
                del d[v]
        return d

    # filter through to find duplicates
    values_removed = dataset.map(lambda i: rm_val(i, [value]))

    # call unique on features
    need_extract = values_removed.unique()

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
            tensors.append(np.sum(prediction,
                                  axis=0) / prediction.shape[0] / 10)

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
    s = x * pow
    # memory leak here
    sum = np.cumsum(s)
    return np.reshape(sum, x.shape)


# rolls a 1D array
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
    last = (0, 0)
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

        # compose both matrices
        new_diag = linalg.diagsvd(s, sp.shape[0], sp.shape[1]) @ diag
        inputbasis = (zSl * SR) @ diag
        orthsout = SL @ new_diag @ ZSr

        quot.append(inputbasis @ orthsout.T)

    quot = np.sum(np.array(quot), axis=0)
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

    # now we modify the differential to apply for matrices
    # so for this we simply multiple

    # This for loop below calculates all the terms linearly so we want to add
    # in the non linear function compositions in a liner manner
    # we want to *not* use inputs as it gums up the works
    zetas = []
    biases = []
    for i, (layer, act) in enumerate(zip(layers, activations)):
        # add in nonlinear operator
        # take power to remove nonlinears
        # weight = np.float_power(layer[0], float(i+1))
        # bias = np.float_power(layer[1], float(i+1))
        weight = layer[0]
        bias = layer[1]  # still gives zero
        # TODO: recovery of inputs through activaiton?
        # vectorize activation
        inv = np.vectorize(lambda x: act(x=x))

        # recreate zeta
        if len(zetas) < 1:
            zetas.append(weight)
        else:
            zetas.append(inv(zetas[-1] + biases[-1])
                         .astype(np.float64) @ weight)

        biases.append(bias)

    zetas.append(zetas[-1] + biases[-1])
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
        output_aggregator(model, training_data), key=lambda tup: tup[0])

    # sheafifed = irfftn(solution, shapes[0])
    sheafifed = np.imag(rfftn(solution, shapes[0]))

    ret = [sheafifed, sols, outward, sort_avg, (solution.T @ outward).T]
    return ret


def get_features(features, value):
    xs = []
    for f in features:
        xs.append(value[f])

    return xs


def get_ds(dataset):
    # shuffle dataset so each sample is a bit different
    dataset.shuffle(BATCH_SIZE)
    # predict training batch, normalize images by 255
    value, *features = list(list(dataset.take(1).as_numpy_iterator())[0]
                            .keys())

    [images, labels] = list(tf.data.Dataset.get_single_element(
        dataset.batch(len(dataset))).values())
    images = normalize_img(images)
    return [images, labels]


def save_interpol_video(trainset, interset, step):
    import matplotlib.animation as animation

    fig = plt.figure()
    img1 = plt.imshow(trainset[0], cmap='gray',
                      interpolation=None,
                      animated=True)
    img2 = plt.imshow(interset[0], cmap='coolwarm', alpha=0.5,
                      interpolation=None, animated=True)

    def update(i):
        # offset for imagery
        img1.set_array(trainset[i])
        img2.set_array(interset[i])
        return [img1]

    ani = animation.FuncAnimation(
        fig=fig,
        func=update,
        interval=20,
        save_count=200,
        blit=True)

    ani.save("images" + str(step) + ".mp4")

    # clear figures and axes
    plt.cla()
    plt.clf()


def interpolate_model_train(sols, model, train, step):
    # get shapes
    outshape = len(sols[1])
    model_shape = [1 if x is None else x for x in model.input_shape]
    # sensible names
    ins = np.array(sols[0])
    # out = np.array(sols[1])
    from scipy.interpolate import make_splprep
    # out is already a diagonalized matrix of 1s
    # so therefore the standard basis becomes 0
    _, _, std_basis = linalg.svd(ins)
    # solve for the new std_basis
    new_basis = linalg.solve(std_basis, np.zeros(std_basis.shape[0]))
    # create LU Decomposition towards new_basis
    lu_decomp = linalg.lu(np.vstack([ins, new_basis]).T)

    # multiply out the final answer column so it is at an equal outputs
    # we can't use this on LU decomposition as it would come out as zero.
    def reduce_basis(decomp):
        solved_decomp = []
        for sample in decomp:
            tomul = sample[:-1]
            mul = sample[-1]
            solved_sheafs = tomul * mul
            solved_decomp.append(solved_sheafs)
        return np.array(solved_decomp)

    # get dataset
    [images, labels] = get_ds(train)

    # interpolate
    [spline, u] = make_splprep(lu_decomp[1].T, k=outshape + 1)
    mask_samples = reduce_basis(np.array(spline(images).swapaxes(0, 1)))
    mask_samples = np.reshape(mask_samples, [images.shape[0] * outshape,
                                             *model_shape[1:]])

    solved_samples = np.repeat(images, outshape, axis=0)
    solved_samples = np.reshape(solved_samples, solved_samples.shape[:-1])

    # check model, reshape inputs
    mask = mask_samples > solved_samples
    import numpy.ma as ma
    masked_samples = ma.array(solved_samples, mask=mask, fill_value=0)

    save_interpol_video(solved_samples, mask_samples, step)

    rep_labels = np.repeat(labels, outshape)
    model.fit(masked_samples, rep_labels, batch_size=BATCH_SIZE, epochs=5)
    return [model, lu_decomp[1], spline, u]


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
        template = np.reshape(np.arange(1, len(avg_outs) + 1), outshape[-1])
        # plot our test
        plt.violinplot(np.transpose(avg_outs), showmeans=True)
        plt.violinplot(np.transpose(prelims), showmeans=True)
        plt.plot(template, np.transpose(final_test), colours[i])

    plt.savefig(name)
    # clear figures and axes
    plt.cla()
    plt.clf()


# returns mutliplicants then powers
def get_unzip_coeffs(ndspline, max_inputs):
    muls = []
    pows = []
    for spline in ndspline:
        t_muls = []
        t_pows = []
        for eq in spline:
            expand = eq.expand()
            zipcoeffs = expand.coefficients(sparse=False)

            mul = zipcoeffs[0].n()
            # if linear
            pow = 0 if len(zipcoeffs) < 2 else zipcoeffs[1].n()

            t_muls.append(mul)
            t_pows.append(pow)

        t_muls = np.array(t_muls)
        t_pows = np.array(t_pows)

        # where multiplicant zero change to zero in power
        t_pows[t_muls == 0.0] = 0.0

        # if less than max inputs pad with zero
        if len(t_muls) < max_inputs:
            t_muls = np.pad(t_muls, (0, max_inputs), t_muls.shape[0],
                            constant_value=0)

        if len(t_pows) < max_inputs:
            t_pows = np.pad(t_pows, (0, max_inputs), t_pows.shape[0],
                            constant_value=0)

        muls.append(t_muls)
        pows.append(t_pows)

    # pad values so we have same sizes
    # this is ordered in symbolic x1 -> xn as inputs
    return np.dstack([np.array(muls), np.array(pows)])


# returns coeffecients in matrix form with [mutliplicants, powers]
def generate_bernstien_polys(params, lu_system):
    from scipy.special import binom
    from sage.all import var

    # use De Casteljau's algorithm
    # get the length of knots so we don't do too many iterations
    knots = params.shape[-1]
    # get output dim
    outdim = lu_system.shape[-1]

    # create vector x type of symbols representing input dimension
    syms = np.array([var("x" + str(i)) for i in range(knots)], dtype="object")

    coeffs = []
    for d in range(outdim):
        coeffs.append((1 - syms) * params[d] +
                      syms * params[d + 1 % outdim])

    coeffs = np.array(coeffs)
    bernoli = np.array([binom(n, v)
                        for v, n in enumerate(reversed(range(knots)))])

    eq = bernoli * coeffs

    # remove symbolics and just use coefficients
    coeffs = get_unzip_coeffs(eq, knots)

    return [coeffs, syms, eq]


def generate_readable_eqs(sol_system, bspline, name):
    # init symbol system bspline has two args
    coeffs, syms, eq = generate_bernstien_polys(bspline[1], sol_system)

    # this now solves for polynomial space
    # now solve each simultaneous equation of tensor output dim * 2
    # first coefficients are all zero due to linearity
    new_shape = list(coeffs.shape)
    new_shape[0] -= 1

    coeffs = np.delete(coeffs, 0, 0)
    coeffs = np.reshape(coeffs, new_shape)
    coeffs = coeffs.T

    # solve to singular constant value
    def svd_lu(lu):
        # use svd to get components
        r_basis, nul, l_scale = linalg.svd(lu)

        # solve components scaling for new basis
        # premute is set to true so no rounding errs occur
        r_factor = linalg.lu(r_basis, permute_l=True, p_indices=True)
        l_factor = linalg.lu(l_scale, permute_l=True, p_indices=True)

        # rhs == 0 here due to factor decomposition i.e. solve(n, 0) == 0
        # therefore r_basis is our inverse
        new_basis = r_factor[0] @ r_factor[1] @ r_basis
        new_scale = l_factor[0] @ l_factor[1] @ l_scale

        # recreate diagonal so we can use to create same shape
        mix = linalg.diagsvd(nul, *lu.shape)

        return new_basis @ mix @ new_scale

    # now we need to solve this via svd and LU
    # no need for pivot
    mul_svd = svd_lu(coeffs[0])
    pow_svd = svd_lu(coeffs[1])

    from sage.all import PolynomialRing, QQ, latex
    from sage.rings.polynomial import polynomial_ring
    mat_ring = PolynomialRing(QQ, names=list(syms))

    # this gets round an outdated library using deprecated functions
    def is_ring(ring):
        return isinstance(ring, polynomial_ring.PolynomialRing_generic)

    setattr(polynomial_ring, 'is_PolynomialRing', is_ring)
    # this import HAS to be after setattr for it to work
    from rec_sequences.FunctionalEquation import FunctionalEquation
    breakpoint()
    mat_syms = np.reshape(np.repeat(syms, mul_svd.shape[-1]), mul_svd.shape)
    mat_eq = np.dstack([mul_svd, mat_syms, pow_svd])

    # find reoccurance relations for mat_eq
    algebras = FunctionalEquation(mat_ring, mat_eq)

    save("EQ.tex", latex(algebras))

    return algebras


def model_create_equation(model_dir, training_data):
    # check optional args
    # create prerequisites
    model = tf.keras.models.load_model(model_dir)
    if model is not None:
        # load dataset for training
        [train_dataset, test_dataset] = tdfs.load(
            training_data,
            download=False,
            split=['train[:80%]', 'test'])
        # calculate fft + shape
        shapes = []
        layers = []

        # append input shape remove None type
        shapes.append([product(model.input_shape[1:])])
        activations = []

        # main wb extraction loop
        for [weights, biases, act, shape] in [
            (layer.weights[0].numpy(), layer.weights[1].numpy(),
                layer.activation, layer.kernel.shape.as_list())
                for layer in model.layers if len(layer.weights) > 1]:
            # weird indentation here
            # if no activation assume linear
            activations.append(
                lambda x: x if act is None else act(x)
            )

            layers.append([weights, biases])
            shapes.append([shape, weights.shape, biases.shape])

        [sheaf, sols, outward, sort_avg, sheafsol] = graph_model(
            model,
            train_dataset,
            activations,
            shapes,
            layers)

        control = tester(model, sheaf, outward, sort_avg)

        # copy out cfgs
        loss_fn_cfg = model.loss.get_config()
        loss_fn = model.loss.__class__.from_config(loss_fn_cfg)
        optcfg = model.optimizer.get_config()
        optimizer = model.optimizer.__class__.from_config(optcfg)
        # copy over user metrics
        metrics = [model.metrics[-1].__dict__['_user_metrics'][0].__class__()]

        train_dataset.cache()
        for t in range(TEST_ROUNDS):
            # should we wipe the model every i in TRAIN_SIZE or leave it?
            test_model = tf.keras.models.clone_model(model)

            test_model.compile(optimizer=optimizer,
                               loss=loss_fn, metrics=metrics)

            # using sols[0] shape as a template for input
            # this would be input, output shape of neural
            # nets e.g. 784,10 for mnist
            systems = []

            train_dataset.shuffle(BATCH_SIZE)
            bsplines = []
            for i in range(TRAIN_SIZE):
                # find variance in solved systems
                [test_model, solved_system,
                    bspline, u] = interpolate_model_train(
                        sols[-1],
                        test_model,
                        train_dataset,
                        i)
                bsplines.append([bspline, u])
                systems.append(solved_system)
                # and testing
                test = tester(test_model, sheaf, outward, sort_avg)
                plot_test(control, test, shapes[-1],
                          "out-epoch-" + str(i) + ".png")

                test_dataset = get_ds(test_dataset)
                print("EVALUATION:")
                test_model.evaluate(test_dataset[0], test_dataset[1],
                                    verbose=2)
                print("CONTROL:")
                model.evaluate(test_dataset[0], test_dataset[1], verbose=2)

            test_model.save("MNIST_only_interpolant.keras")

            # generate the human readable eq
            generate_readable_eqs(systems[-1],
                                  bsplines[-1], "EQUATION_OUTPUT.latex")


if __name__ == "__main__":
    if len(sys.argv) > 2:
        path = os.path.dirname(__file__)
        model = os.path.join(path, sys.argv[1])
        try:
            print(os.path.abspath(model))
            if len(sys.argv) > 2:
                model_create_equation(os.path.abspath(model), sys.argv[2])
            else:
                print("""not enough commands, please give a filename of a model
                      to extract, it's training dataset (which may be altered
                      at future date)
                output for a tex file, and a csv file containing each type of
                      acitvation used delimitered by ; (optional)""")
        except FileNotFoundError:
            print("""file not found,
                    please give filename of model to extract equation from""")
    else:
        print("""not enough commands, please give a filename of a
              model to extract, it's training dataset (which may be altered at
    find the relations will probably result in error
                 future date)""")
