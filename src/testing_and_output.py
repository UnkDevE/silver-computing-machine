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

import torch

import numpy as np
import matplotlib.pyplot as plt

import cech_algorithm as ca
import model_extractor as me

# TUNE THESE INPUT PARAMS
TEST_ROUNDS = 1
TRAIN_SIZE = 1


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


# helper for dict extraction
def extract(param, layer):
    ds = layer.__dict__()
    if param in ds:
        return ds[param]


def model_create_equation(model_dir, model_name, ds_pair):
    # check optional args
    # create prerequisites
    model = torch.load(model_dir)
    if model is not None:
        # load dataset for training
        [train_dataset, test_dataset] = ds_pair

        # calculate fft + shape
        shapes = []
        layers = []

        # append input shape remove None type
        shapes.append([ca.product(model.input_shape[1:])])
        activations = []

        params_to_ex = ["weights", "biases", "act", "shape"]
        # main wb extraction loop
        for layer in model:
            weights, biases, act, shape = [extract(p, layer)
                                           for p in params_to_ex]

            # if no activation assume linear
            activations.append(
                lambda x: x if act is None else act(x)
            )

            layers.append([weights, biases])
            shapes.append([shape, weights.shape, biases.shape])

        [sheaf, sols, outward, sort_avg, _] = ca.graph_model(
            model,
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
            test_model = model.clone()

            test_model.compile(optimizer=optimizer,
                               loss=loss_fn, metrics=metrics)

            # using sols[0] shape as a template for input
            # this would be input, output shape of neural
            # nets e.g. 784,10 for mnist
            systems = []

            train_dataset.shuffle(me.BATCH_SIZE)
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
