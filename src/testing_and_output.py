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
import numpy as np
import matplotlib.pyplot as plt

from torchvision.models import get_model
from torchvision.transforms import v2

from scipy.stats import chisquare

import src.cech_algorithm as ca
import src.model_extractor as me
import src.training as tr

from copy import copy


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
    torch_out = torch_out.to(ca.TORCH_DEVICE)

    model.eval()
    final_test = model(torch_out)

    # print(sort_avg)
    return [sort_avg, final_test.cpu().detach().numpy()]


def plot_test(starttest, endtest, outshape, name):
    tests = [starttest, endtest]
    plt.xlabel("features")
    plt.ylabel("activation")

    colours = ["ro--", "bo--"]

    for i, [avg_outs, final_test] in enumerate(tests):
        template = np.reshape(avg_outs, [ca.product(list(avg_outs.shape[1:])),
                              avg_outs.shape[0]])
        # plot our test
        plt.violinplot(template, showmeans=True)

        # interpolate spline with nearest round up power of two
        p2_shape = 1
        while p2_shape < final_test.shape[-1]:
            p2_shape *= 2

        from scipy.interpolate import make_interp_spline
        final_test = np.reshape(final_test, final_test.shape[-1])
        interp = make_interp_spline(np.linspace(0.0, 1.0, final_test.shape[0]),
                                    final_test)

        yaxis = interp(np.linspace(0, p2_shape))

        plt.plot(yaxis.T, colours[i])

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


def model_create_equation(model, names, dataset, in_shape, test_rounds):
    # check optional args
    # create prerequisites
    if model is not None:
        # works for IMAGENET ONLY
        if "imagenet" in names[1].lower():
            dataset.target_transform = tr.ClassLabelWrapper()

        from torch.utils.data import random_split
        [train_dataset, test_dataset] = random_split(dataset, [0.7, 0.3],
                                                     generator=ca.GENERATOR)

        train_dataset.dataset = copy(dataset)

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
            layers.append([np.copy(weights.cpu().detach().numpy()),
                           np.copy(biases.cpu().detach().numpy()),
                           act, shapes[-1]])

        [sheaf, sols, outward, sort_avg] = ca.graph_model(
            model,
            shapes,
            layers)

        control = tester(model, shapes, sheaf, outward, sort_avg)

        [bspline, _, _] = tr.make_spline(sols[-1])

        from src.meterns import HDRMaskTransform
        train_dataset.dataset.transforms = v2.Compose([
            train_dataset.dataset.transforms,
            HDRMaskTransform])

        for _ in range(test_rounds):
            # should we wipe the model every i in TRAIN_SIZE or leave it?

            test_model = copy(model)

            # using sols[0] shape as a template for input
            # this would be input, output shape of neural
            # nets e.g. 784,10 for mnist

            for i in range(test_rounds):
                # find variance in solved systems

                test_model = tr.interpolate_model_train(
                    test_model,
                    train_dataset, i,
                    names)

                # and testing
                test = tester(test_model, shapes, sheaf, outward, sort_avg)
                plot_test(control, test, shapes[-1],
                          "{name}-out-epoch-{i}.png"
                          .format(name="".join(names), i=i))

                # onehots labels
                chis = []
                from torch.utils.data import DataLoader
                test_loader = DataLoader(test_dataset,
                                         batch_size=me.BATCH_SIZE)

                for [data, actual] in test_loader:
                    ctrl = model(data).detach().numpy()
                    test = test_model(data).detach().numpy()

                    test_chi = chisquare(test, f_exp=actual)
                    c_chi = chisquare(ctrl, f_exp=actual)
                    diff_chi = chisquare(test, f_exp=ctrl)
                    chis.append(np.array([test_chi, c_chi, diff_chi]))

                chis = np.array(chis).T
                breakpoint()
                test_acc = np.mean(chis)
                ctrl_acc = np.mean(chis)
                diff_acc = np.mean(chis)
                diff_mean = test_acc - ctrl_acc

                print("CONTROL:")
                print(ctrl_acc)
                print("EVALUATION:")
                print(test_acc)
                print("CHI DIFF")
                print(diff_acc)
                print("MEAN DIFF")
                print(diff_mean)


def model_test_batch(root, res, rounds, names, download=True):
    datasets = me.download_data(root, res, download=download)

    for ds in datasets:
        model = get_model(names[0], weights=names[1])
        model.to(ca.TORCH_DEVICE)
        model.eval()
        model_create_equation(model, names, ds, res, rounds)
