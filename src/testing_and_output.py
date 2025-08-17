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

import numpy as np
import matplotlib.pyplot as plt

import torch
from torchvision.models import vgg11

import src.cech_algorithm as ca
import src.model_extractor as me

# TUNE THESE INPUT PARAMS
TEST_ROUNDS = 1
TRAIN_SIZE = 1


# for reproduciblity purposes
GENERATOR_SEED = randint(0, sys.maxsize)
print("REPRODUCUBLE RANDOM SEED IS:" + str(GENERATOR_SEED))
# needs to be _global_ here otherwise generation of seed will start at 0
# multiple times
GENERATOR = generator1 = torch.Generator().manual_seed(GENERATOR_SEED)


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


def model_create_equation(model, model_name, dataset):
    # check optional args
    # create prerequisites
    if model is not None:
        # load dataset for training

        from torch.utils.data import random_split
        [train_dataset, test_dataset] = random_split(dataset, [7, 3],
                                                     generator=GENERATOR)

        # calculate fft + shape
        shapes = []
        layers = []

        # append input shape remove None type
        shapes.append([ca.product(model.input_shape[1:])])
        activations = []

        # main wb extraction loop
        for layer in model:
            weights, biases, act, shape = [layer.weights, layer.baises,
                                           layer.act, layer.size()]

            # if no activation assume linear
            activations.append(
                lambda x: x if act is None else act(x)
            )

            layers.append([weights, biases])
            shapes.append([shape, weights.numpy().shape,
                           biases.numpy().shape])

        [sheaf, sols, outward, sort_avg, _] = ca.graph_model(
            model,
            shapes,
            layers)

        control = tester(model, sheaf, outward, sort_avg)

        for _ in range(TEST_ROUNDS):
            # should we wipe the model every i in TRAIN_SIZE or leave it?
            test_model = model.clone()

            # using sols[0] shape as a template for input
            # this would be input, output shape of neural
            # nets e.g. 784,10 for mnist
            systems = []

            bsplines = []
            for i in range(TRAIN_SIZE):
                # find variance in solved systems
                [test_model, solved_system,
                    bspline, u] = ca.interpolate_model_train(
                        model_name,
                        sols[-1],
                        test_model,
                        train_dataset,
                        i)
                bsplines.append([bspline, u])
                systems.append(solved_system)
                # and testing
                test = tester(test_model, sheaf, outward, sort_avg)
                plot_test(control, test, shapes[-1],
                          model_name + "-out-epoch-" + str(i) + ".png")

                test_dataset = me.get_ds(test_dataset)
                print("EVALUATION:")
                test_model.evaluate(test_dataset[0], test_dataset[1],
                                    verbose=2)
                print("CONTROL:")
                model.evaluate(test_dataset[0], test_dataset[1], verbose=2)

            test_model.save(model_name + "_only_interpolant")


def model_test_batch(root):
    datasets = me.download_data(root)

    for ds in datasets:
        vgg11_model = vgg11(ds)
        model_create_equation(vgg11_model, "vgg11_" + str(type(ds)), ds)
