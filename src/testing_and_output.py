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
"""

from random import randint
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import torch
from torchvision.models import get_model

import src.cech_algorithm as ca
import src.model_extractor as me

# TUNE THESE INPUT PARAMS
TEST_ROUNDS = 1
TRAIN_SIZE = 1

# for reproduciblity purposes
GENERATOR_SEED = randint(0, sys.maxsize)
print("REPRODUCUBLE RANDOM SEED IS:" + str(GENERATOR_SEED))

sd = Path("seeds")
sd.touch()
with open("seeds", "a") as f:
    f.write(str(GENERATOR_SEED) + "\n")

# needs to be _global_ here otherwise generation of seed will start at 0
# multiple times
torch.manual_seed(GENERATOR_SEED)
GENERATOR = generator1 = torch.Generator().manual_seed(GENERATOR_SEED)


def bucketize(prelims):
    arr = []
    for ps in prelims:
        for i, p in enumerate(ps):
            if len(arr) <= i:
                arr.append([])
            arr[i].append(p)
    return np.array(arr)


def tester(model, shapes, sheafout, sheafs, sort_avg):
    model_shape = list(shapes[0])  # in shape from pytorch
    out = np.reshape(sheafout, [-1, *model_shape])
    out = out[np.newaxis, :]

    # needs batching
    torch_out = ca.jax_to_tensor(out)

    model.eval()
    final_test = model(torch_out)

    print(sort_avg)
    return [sort_avg, final_test.detach().numpy()]


def plot_test(starttest, endtest, outshape, name):
    tests = [starttest, endtest]
    plt.xlabel("features")
    plt.ylabel("activation")

    colours = ["ro--", "bo--"]

    for i, [avg_outs, final_test] in enumerate(tests):
        template = np.reshape(np.arange(1, len(avg_outs) + 1), outshape[-1])
        # plot our test
        plt.violinplot(np.transpose(avg_outs), showmeans=True)
        plt.plot(template, np.transpose(final_test), colours[i])

    plt.savefig(name)
    # clear figures and axes
    plt.cla()
    plt.clf()


def get_activations(model):
    hooks = {}
    for name, _ in model.named_modules():
        attr = getattr(model, name, None)
        if attr is not None:
            hooks[name] = attr

    return hooks


def model_create_equation(model, names, dataset, in_shape):
    # check optional args
    # create prerequisites
    if model is not None:
        # load dataset for training

        from torch.utils.data import random_split
        [train_dataset, test_dataset] = random_split(dataset, [0.7, 0.3],
                                                     generator=GENERATOR)

        # calculate fft + shape
        layers = []

        # if no activation assume linear
        activations = get_activations(model)

        # extract correct input size
        shapes = [[in_shape, in_shape]]

        # main wb extraction loop
        from itertools import batched
        for [[weights, biases], act] in zip(batched(model.parameters(), 2),
                                            activations):

            shapes.append([weights.size(), biases.size()])
            # make a copy of weights and biases to work off of
            layers.append([np.copy(weights.detach().numpy()),
                           np.copy(biases.detach().numpy()),
                           act, shapes[-1]])

        [sheaf, sols, outward, sort_avg] = ca.graph_model(
            model,
            shapes,
            layers)

        control = tester(model, shapes, sheaf, outward, sort_avg)

        for _ in range(TEST_ROUNDS):
            # should we wipe the model every i in TRAIN_SIZE or leave it?

            import copy
            test_model = copy.copy(model)

            # using sols[0] shape as a template for input
            # this would be input, output shape of neural
            # nets e.g. 784,10 for mnist
            systems = []

            bsplines = []
            for i in range(TRAIN_SIZE):
                # find variance in solved systems
                [test_model, solved_system,
                    bspline, u] = ca.interpolate_model_train(
                        sols[-1],
                        test_model,
                        train_dataset,
                        i,
                        shapes,
                        names,
                        vid_out="{name}_hotspots.mp4".format(
                            name="".join(names)))

                bsplines.append([bspline, u])
                systems.append(solved_system)
                # and testing
                test = tester(test_model, shapes, sheaf, outward, sort_avg)
                plot_test(control, test, shapes[-1],
                          "{name}-out-epoch-{i}.png"
                          .format(name="".join(names), i=i))

                labels = me.get_labels(names)
                print("EVALUATION:")
                test_model.evaluate(test_dataset, labels,
                                    verbose=2)
                print("CONTROL:")
                model.evaluate(test_dataset, labels, verbose=2)


def model_test_batch(root, res, names, download=True):
    datasets = me.download_data(root, res, download=download)

    for ds in datasets:
        model = get_model(names[0], weights=names[1])
        model.eval()
        model_create_equation(model, names, ds, res)
