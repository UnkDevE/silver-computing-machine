#!/bin/python
"""
    SILVER-COMPUTING-MACHINE converts Nerual nets into human readable code
    or maths
    Copyright (C) 2024-2025 Ethan Riley

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


    Jax module - does heavy lifting of topology analysis
"""


# jax for custom code
import jax
import jax.numpy as jnp
import torch
# torch for tensor LU
import torch.linalg as t_linalg
import jax.scipy.linalg as j_linalg

from jax.numpy.fft import rfftn
import numpy as np

from matplotlib import pyplot as plt


def jax_to_tensor(jax_arr):
    return torch.from_numpy(np.asarray(jax_arr).copy())


def product(x):
    out = 1
    for y in x:
        out *= y
    return out


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


# image in linear algebra
def image(rref):
    # now column reduced echeolon
    cref = rref.transpose()
    cstack = []
    # we flip here as we want the bottom half of the matrix up to the zero
    # row reducing compute
    for row in jnp.flip(cref):
        # split by zero row
        if jnp.all(row == jnp.zeros_like(cref[0]).astype(jnp.float64)):
            return jnp.stack(cstack)
        cstack.append(row)
    return jnp.stack(cstack)


# calculates the chec differential
def _chec_diff(x, start):
    # axis=1 from oldcode to fix bottoming out on final feature
    # this could be sped up with jax but is quick op
    mesh = np.meshgrid(x)[0]

    shape = jnp.roll(jnp.reshape(mesh, x.shape), start, axis=1)
    reshape = jnp.reshape(jnp.arange(product(shape.shape)), shape.shape)
    pow = jnp.float_power(-jnp.ones_like(x), reshape)
    s = x * pow
    # memory leak here
    sums = jnp.cumsum(s)
    return jnp.reshape(sums, x.shape)


# rolls a 1D array
def shifts(x, start):
    y = np.zeros([x.shape[0], *x.shape])
    for i in range(-start, x.shape[0] - start):
        y[i] = np.roll(x, i)
    return jnp.array(y)


def chec_diff(x, start):
    permutes = shifts(x, start)

    diffs = []
    for i, perm in enumerate(permutes):
        diffs.append(_chec_diff(perm, i))

    return jnp.array(diffs)


"""
def pad_coeff_output(coeff):
    last = 0
    arr = []
    for [x, i] in coeff:
        if i-1 >= last:
            [arr.append(0) for _ in range(i-1)]
        arr.append(x)
        last = i

    return jnp.array(arr)
"""


# start finding chec cohomology from sheafs
# simplex / cocycle in this case is the direct sum
# compute kernels, images and chec cohomologies
# gets the kth layers kth Chomology class
def chec_chomology(layer):
    diff = chec_diff(layer, 0)
    cocycle = chec_diff(layer, 1)
    rrefco, _, _, = t_linalg.lu_factor_ex(jax_to_tensor(cocycle))
    coboundary = image(rrefco.numpy())

    # LDU we want U (reduced row echelon form)
    rref, _, _, = t_linalg.lu_factor_ex(jax_to_tensor(diff))
    simplex = image(rref.numpy().transpose())

    # calculate chomologies
    return (simplex, coboundary)


def sortshape(a):
    last = (0, 0)
    shapes = list(a.shape)
    for (i, sh) in enumerate(shapes):
        if sh >= last[1]:
            a = jnp.swapaxes(a, -i, last[0])
            last = (i, sh)
    return a


# takes in two shapes to matmul if same mutliplies
# sorts axes in greatest to least
def maybematmul(a, b):
    shapes = [list(a.shape), list(b.shape)]
    # sort in gt to lt
    shapes_sorted = [np.flip(np.argsort(s)) for s in shapes]

    if shapes[0] != shapes[1]:
        a = jax.lax.transpose(a, shapes_sorted[0])
        b = jax.lax.transpose(b, shapes_sorted[1])

    fs = shapes_sorted[0]
    bools = [si == fs for si in shapes_sorted if sum(si) == sum(fs)]
    ors = [len(si) > len(fs) for si in shapes_sorted]

    if not np.any(ors) and np.all(bools):
        c = b * a
    else:
        if np.any(ors):
            # this isn't quite right as the sum returns something incorrect
            if sum(shapes[0]) <= sum(shapes[1]):
                c = a.T * b.T
            else:
                c = a @ b
        elif sum(shapes[0]) > sum(shapes[1]):
            c = a @ b
        else:
            c = b @ a

    return jax.lax.transpose(c, np.flip(np.argsort(c.shape)))


# where space is the space in matrix format
# plus the subset of spaces
def quot_space(subset, space):
    # find commonality shape and set them
    subset = sortshape(subset)

    # set solve subset for zero
    zerosub = []
    for subs in subset:
        # make square by jigging axes order by greatest first because set
        unqaxes = list(set(subs.shape))
        jsubs = jnp.swapaxes(subs, len(unqaxes) - 1, 0)
        jsolve = j_linalg.solve(jsubs, jnp.ones(jsubs.shape[-1]))
        zerosub.append(jsolve)

    zerosub = jnp.array(zerosub)

    # sheafify with irfftn to find quotient
    quot = []
    zSl, zs, ZSr = j_linalg.svd(zerosub)
    diag = j_linalg.svd(zerosub, full_matrices=True, compute_uv=False)

    for sp in space:
        # use svd here
        SL, s, SR = j_linalg.svd(sp)
        # differing approaches if even or odd dim
        # compose both matrices
        shapes = [list(s.shape), list(zs.shape)]
        [s.sort() for s in shapes]

        if shapes[0] == shapes[1]:
            new_diag = s @ zs
        else:
            new_diag = (s.T @ zs).T

        if set(shapes[0]) != set(shapes[1]) or ZSr.shape == SL.shape:
            inputbasis = maybematmul(ZSr, SL) @ diag
            orthsout = zSl @ diag @ new_diag @ SR
        else:
            inputbasis = maybematmul(ZSr, SR) @ new_diag
            orthsout = maybematmul(SL @ diag @ new_diag, SR)
            inputbasis = jnp.swapaxes(inputbasis,
                                      len(inputbasis.shape) // 2, 0)

        quot.append(maybematmul(inputbasis, orthsout))

    quot = jnp.sum(jnp.array(quot), axis=0)
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
    start = [quot_space(kerims[0][1], jnp.ones_like(kerims[0][1]))]
    # don't forget append in start in reverse!
    [start.append(c) for c in cohol]

    return jnp.array(start)


def create_sols_from_system(solved_system):
    solutions = []
    for system in solved_system:
        outdim = system.shape[-1]

        # create output template to be rolled
        template = np.zeros(outdim)
        template[0] = 1
        template = jnp.array(template)
        inv = t_linalg.pinv(jax_to_tensor(system)).numpy()

        SL, s, _ = j_linalg.svd(inv)

        outtemplate = shifts(template, 0)
        # solve backwards
        sols = []
        for shift in outtemplate:
            # use svd with solve
            sols.append(maybematmul(j_linalg.solve(SL, shift) * s, inv))

        solutions.append((jnp.array(sols), outtemplate))

    # what next sheafify outputs?
    return solutions


def ceildiv(a, b):
    return -(a // -b)


def graph_model(model, shapes, layers):
    targets = [1] * len(shapes)
    # add output target
    targets[-1] = shapes[-1]

    shapes = complete_bias(shapes, targets)

    # fft calculation goes through here
    # zeroth index of layer is weights
    solved_system = [cohomologies(layer[0]) for layer in layers]
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

    # sheafifed = irfftn(solution, shapes[0])
    sheafifed = jnp.imag(rfftn(solution, shapes[0]))

    ret = [sheafifed, sols, outward, (solution.T @ outward).T]
    return ret


# saver function helper
def save(filename, write):
    import os.path

    if os.path.isfile(filename):
        os.remove(filename)

    with open(filename, "xt") as file:
        file.write(write)


def save_interpol_video(model_name, trainset, interset, step):
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

    ani.save(model_name + str(step) + ".mp4")

    # clear figures and axes
    plt.cla()
    plt.clf()


def interpolate_model_train(sols, model, train, step, shapes, vid_out=None):
    # get shapes
    outshape = len(sols[1])
    model_shape = [1 if x is None else x for x in shapes[0]]
    # sensible names
    ins = jnp.array(sols[0])
    # out = jnp.array(sols[1])
    from scipy.interpolate import make_splprep
    # out is already a diagonalized matrix of 1s
    # so therefore the standard basis becomes 0
    _, _, std_basis = j_linalg.svd(ins)
    # solve for the new std_basis
    new_basis = j_linalg.solve(std_basis, jnp.zeros(std_basis.shape[0]))
    # create LU Decomposition towards new_basis
    lu_decomp = t_linalg.lu_factor_ex(jax_to_tensor(jnp.vstack([ins,
                                                    new_basis]).T))

    # multiply out the final answer column so it is at an equal outputs
    # we can't use this on LU decomposition as it would come out as zero.
    def reduce_basis(decomp):
        solved_decomp = []
        for sample in decomp:
            tomul = sample[:-1]
            mul = sample[-1]
            solved_sheafs = tomul * mul
            solved_decomp.append(solved_sheafs)
        return jnp.array(solved_decomp)

    # get dataset
    import model_extractor as me
    [images, labels] = me.get_ds(train)

    # interpolate
    [spline, u] = make_splprep(lu_decomp[1].T.numpy(), k=outshape + 1)
    mask_samples = reduce_basis(jnp.array(spline(images).swapaxes(0, 1)))
    mask_samples = jnp.reshape(mask_samples, [images.shape[0] * outshape,
                               *model_shape[1:]])

    solved_samples = jnp.repeat(images, outshape, axis=0)
    solved_samples = jnp.reshape(solved_samples, solved_samples.shape[:-1])

    # check model, reshape jnputs
    mask = mask_samples > solved_samples
    import numpy.ma as ma
    masked_samples = ma.array(solved_samples, mask=mask, fill_value=0)

    # this is internal testing and so must be baked in?
    # save video output as vid_out directory
    if vid_out is not None:
        save_interpol_video(str(vid_out), solved_samples, mask_samples, step)

    rep_labels = jnp.repeat(labels, outshape)
    model.fit(masked_samples, rep_labels, epochs=5)
    return [model, lu_decomp[1].numpy(), spline, u]
