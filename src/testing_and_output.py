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

import torch
from torchvision.models import get_model
from torchvision.transforms import v2

from scipy import stats

import src.cech_algorithm as ca
import src.model_extractor as me
import src.training as tr

from copy import copy
import os


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
        template = np.reshape(avg_outs,
                              [ca.product(list(avg_outs.shape[1:])),
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
    tests = []
    if model is not None:
        # works for IMAGENET ONLY
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

        for i in range(test_rounds):
            # should we wipe the model every i in TRAIN_SIZE or leave it?

            from copy import deepcopy
            # MUST use deepcopy here otherwise model is used twice
            test_model = deepcopy(model)

            # using sols[0] shape as a template for input
            # this would be input, output shape of neural
            # nets e.g. 784,10 for mnist

            # find variance in solved systems

            test_model = tr.interpolate_model_train(
                test_model,
                train_dataset, i,
                names)

            # and testing
            test = tester(test_model, shapes, sheaf, outward, sort_avg)
            plot_test(control, test, shapes[-1],
                      "{name}-out-epoch-{i}.png".format(name="".join(names),
                                                        i=i))

            # onehots labels
            from torch.utils.data import DataLoader
            test_loader = DataLoader(test_dataset, batch_size=me.BATCH_SIZE)

            # safety code so no training happens
            model.eval()
            test_model.eval()

            print("TESTING...")
            accs = []
            for [data, actual] in test_loader:
                data = data.float().to(ca.TORCH_DEVICE, non_blocking=True)
                actual = actual.float().cpu().numpy()

                ctrl = model(data).cpu().detach().numpy()
                test = test_model(data).cpu().detach().numpy()

                # find mean over batch
                ctrl_t = stats.ttest_ind(ctrl, actual)
                test_t = stats.ttest_ind(test, actual)
                diff = stats.ttest_ind(test, ctrl)
                accs.append([ctrl_t.pvalue, test_t.pvalue, diff.pvalue])

            accs = np.array(accs).T
            m1 = stats.combine_pvalues(accs[0]).pvalue
            m2 = stats.combine_pvalues(accs[1]).pvalue
            diff = m1 - m2
            tvsctrl = stats.combine_pvalues(accs[2]).pvalue
            print("INDEPENDENT EVAL VS ACT PVALUE:")
            print(m1)
            print("INDEPENDENT TEST VS ACT PVALUE:")
            print(m2)
            print("PVALUE DIFF:")
            print(diff)
            print("TTEST TEST VS CTRL DIFF:")
            print(tvsctrl)

            tests.append({'eval': float(m1),
                          'test': float(m2),
                          'diff': float(diff),
                          'testvsctrl': float(tvsctrl),
                          })

        # clean up
        model.to('cpu')
        del model
        import gc
        gc.collect()
        torch.cuda.empty_cache()

        import shutil
        # remove runs directory that contains caches of models
        # as it affects model weight preformance and changes control
        # file is in src so go up one to dir and remove runs
        cache_path = os.path.join(os.path.join(
            os.path.realpath(__file__), '..'), "runs")
        if os.path.exists(cache_path):
            shutil.rmtree(cache_path)

        return tests


def model_test_batch(root, res, rounds, names, download=True):
    datasets = me.download_data(root, res, download=download)
    tests = []

    for ds in datasets:
        print("USING {} DATSET".format(ds.__class__.__name__))
        model = get_model(names[0], weights=names[1])
        model.to(ca.TORCH_DEVICE)
        model.eval()
        test = None
        try:
            out = model_create_equation(model, names, ds, res, rounds)
            test = {
                'dataset': ds.__class__.__name__,
                'test_output': out}
        except Exception as e:
            print(e)
            test = {'dataset': ds.__class__.__name__,
                    'test_output': 'failure err {}'.format(e)}
        finally:
            tests.append(test)

        # just in case
        model = None

    import json
    with open("test_output.json", "a+") as f:
        json.dump(tests, f, indent=4)
