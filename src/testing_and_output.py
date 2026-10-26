#!/bin/python
"""
    SILVER-COMPUTING-MACHINE converts Nerual nets into human readable code
    or maths
    Copyright (C) 2024-2026 Ethan Riley

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
from torch.utils.data import DataLoader
from scipy import stats

import src.cech_algorithm as ca
import src.model_extractor as me
import src.training as tr
from src.training import BATCH_SIZE

import os
import json

PVALUE_SMALL = 0.005
PVALUE_MID = 0.01
PVALUE_LARGE = 0.05
EQUIVALENCE_BOUND = 0.05


def get_activations(model):
    hooks = {}
    for name, _ in model.named_modules():
        attr = getattr(model, name, None)
        if attr is not None:
            hooks[name] = attr

    return hooks


def reset_model_weights(layer):
    if hasattr(layer, 'reset_parameters'):
        layer.reset_parameters()
    else:
        if hasattr(layer, 'children'):
            for child in layer.children():
                reset_model_weights(child)


def ttost_ind(x, y, delta):
    xystat = (stats.ttest_ind(x, y - delta, alternative='greater', axis=None),
              stats.ttest_ind(x, y + delta, alternative='less', axis=None))

    pval = xystat[0].pvalue
    if pval.max() < xystat[1].pvalue.max():
        pval = xystat[1].pvalue

    return stats.combine_pvalues(pval)


# rolls a 1D array
def shifts(x, start):
    y = np.zeros([x.shape[0], *x.shape])
    for i in range(-start, x.shape[0] - start):
        y[i] = np.roll(x, i)
    return np.array(y)


def plot_test_spline(bspline, dataset, percent_samples, name, fast=False):
    fast_evaluator = bspline['gpu']
    slow_evaluator = bspline['scipy']
    # cheat to get class_to_idx as there is no standard of getting labels
    f_img, labels = dataset[0]
    no_classes = len(labels.unsqueeze(0))
    ds_len = len(dataset)

    class_outs = []
    tests = []
    zero_weight = np.zeros_like(labels)
    from torch.utils.data import WeightedRandomSampler
    for i in range(no_classes):
        class_weight = zero_weight
        class_weight[i] = 1.
        sampler = WeightedRandomSampler(class_weight,
                                        ds_len // percent_samples,
                                        replacement=True,
                                        generator=ca.GENERATOR)
        loader = DataLoader(dataset, batch_size=BATCH_SIZE, sampler=sampler)

        outs = []
        if not fast:
            for img, k in loader:
                gpu_img = img.to(ca.TORCH_DEVICE, non_blocking=True)
                fast = fast_evaluator(gpu_img).cpu().numpy()
                slow = slow_evaluator(img.numpy())
                test = None
                if np.any(stats.kstest(fast, slow).pvalue > PVALUE_SMALL):
                    test = stats.mannwhitneyu(fast, slow, axis=None)
                else:
                    test = ttost_ind(fast, slow, EQUIVALENCE_BOUND)

                tests.append(test)
                outs.append(fast)
        else:
            for img, k in loader:
                gpu_img = img.to(ca.TORCH_DEVICE, non_blocking=True)
                fast = fast_evaluator(gpu_img).cpu().numpy()
                outs.append(fast)

        class_outs.append((i, outs))

    if tests != []:
        pval = stats.combine_pvalues(tests)
        with open("spline_acc.json", "a+") as spf:
            json.dump(pval, spf, cls=numpyencoder)

    # TODO: plot the data
    ctrl_x = bspline['scipy'].c[:, 0]
    ctrl_y = bspline['scipy'].c[:, 1]

    plt.plot(ctrl_x, ctrl_y,
             'k--', label='Control Points', marker='o', markerfacecolor='red')
    # linear space per pixel
    linspace = np.linspace(0, 1, num=ca.product(f_img.shape)
                           ).reshape(f_img.shape)
    # find permutations of linspace
    lp_space = shifts(linspace, start=0)

    shapes = bspline['scipy'](lp_space)
    plot_shape = bspline['scipy'](
            np.linspace(0, 1, num=ca.product(
                f_img.shape[1:])).reshape(*f_img.shape[1:]))
    plt.plot(plot_shape[0], plot_shape[1], 'b', linewidth=2.0,
             label='B-spline curve')

    # too many fors!
    for cl, outs in class_outs:
        # move batch to iter access
        for batch in outs:
            # move batch from to first position instead of piecewise
            batch = np.moveaxis(batch, 1, 0)
            for outpiecewise, shape in zip(batch, shapes):
                for out in outpiecewise:
                    distance = []
                    for sp in shape:
                        # norm comes out 1d dunno why
                        distance.append(np.linalg.norm(out - sp))
                    avgs = np.average(distance, axis=(0, 1))
                    plt.contour(avgs, cl // no_classes, levels=no_classes)

    plt.axis([min(ctrl_x)-1, max(ctrl_x)+1, min(ctrl_y)-1, max(ctrl_y)+1])
    plt.savefig(name)
    # clear figures and axes
    plt.cla()
    plt.clf()


def model_create_equation(model, names, dataset, in_shape, test_rounds,
                          root='./datasets', imagenet_ds=False,
                          alldefault=False):
    # check optional args
    # create prerequisites
    base_transform = [v2.ToImage(),
                      v2.ToDtype(torch.float32,
                                 scale=True),
                      v2.Resize([in_shape, in_shape]),
                      v2.RGB()]

    dataset_train = dataset(root, transform=v2.Compose(base_transform),
                            target_transform=tr.ClassLabelWrapper())
    tests = []
    if model is not None:
        # works for IMAGENET MODEL ONLY

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

        if not alldefault:
            bspline = tr.make_spline(sols[-1])
            print("MAKING VISUAL AND CORRELATION MATRIX")
            spline_acc = "Y" != input("TEST SPLINE ACC Y/N:")
            plot_test_spline(bspline, dataset_train, 5,
                             '{}.png'.format(''.join(names)), fast=spline_acc)

            vidout_yn = "Y" == input("VIDEO OUTPUT OF DS PRE-MERTENS Y/N:")

            edit_yn = input("EDIT Y/N")
            if edit_yn == "Y":
                print("GENERATING FEATURE SET")
                pass
        else:
            bspline = tr.make_spline(sols[-1])
            plot_test_spline(bspline, dataset_train, 5,
                             '{}.png'.format(''.join(names)), fast=True)

        for i in range(test_rounds):
            # should we wipe the model every i in TRAIN_SIZE or le
            # plot_test(control, test, shapes[-1],
            #           "{name}-out-epoch-{i}.png".format(name="".join(names),
            #                                            i=i))

            from copy import deepcopy
            # MUST use deepcopy here otherwise model is used twice
            test_model = deepcopy(model)

            # using sols[0] shape as a template for input
            # this would be input, output shape of neural
            # nets e.g. 784,10 for mnist

            # find variance in solved systems
            # keep bspline in scope to allow for editing
            test_model = tr.interpolate_model_train(test_model,
                                                    dataset_train,
                                                    i, bspline['gpu'],
                                                    names, save_vid=vidout_yn)

            # and testing
            # onehots labels
            test_loader = DataLoader(dataset_train, generator=ca.GENERATOR,
                                     batch_size=BATCH_SIZE)

            # safety code so no training happens
            model.eval()
            test_model.eval()

            if not imagenet_ds:
                print("TESTING...")
                ctrls, t_tests, cvst = [[], [], []]
                for data, actual in test_loader:
                    data = data.float().to(ca.TORCH_DEVICE, non_blocking=True)

                    # find mean over batch
                    actual = actual.float().cpu().numpy()
                    ctrl = model(data).cpu().detach().numpy()
                    test = test_model(data).cpu().detach().numpy()

                    ctrl_t, test_t, tvsctrl = [None, None, None]
                    if np.any(stats.kstest(test, ctrl).pvalue > PVALUE_SMALL):
                        ctrl_t = stats.mannwhitneyu(ctrl, actual, axis=None)
                        test_t = stats.mannwhitneyu(test, actual, axis=None)
                        tvsctrl = stats.mannwhitneyu(test, ctrl, axis=None)
                    else:
                        ctrl_t = ttost_ind(ctrl, actual, EQUIVALENCE_BOUND)
                        test_t = ttost_ind(test, actual, EQUIVALENCE_BOUND)
                        tvsctrl = ttost_ind(test, ctrl, EQUIVALENCE_BOUND)

                    ctrls.append(ctrl_t)
                    t_tests.append(test_t)
                    cvst.append(tvsctrl)

                ctrl_t = stats.combine_pvalues(ctrls)
                test_t = stats.combine_pvalues(tests)
                tvsctrl = stats.combine_pvalues(tvsctrl)
                print("EVAL VS ACT EQUIV:")
                print('SUCCESS : {}, PVAL: {}'.format(*ctrl_t))
                print("TEST VS ACT EQUIV:")
                print('SUCCESS : {}, PVAL: {}'.format(*test_t))
                print("TEST VS CTRL DIFF EQUIV:")
                print('SUCCESS : {}, PVAL: {}'.format(*tvsctrl))

                tests.append({
                              'eval_pval': '{}'.format(ctrl_t[1]),
                              'test_pval': '{}'.format(test_t[1]),
                              'testvsctrl_pvalue': '{}'.format(tvsctrl[1]),
                              'randomseed': int(torch.initial_seed())
                              })
            # clean up
            model.to('cpu')
            del model
            import gc
            gc.collect()
            torch.cuda.empty_cache()

        return tests


class numpyencoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(numpyencoder, self).default(obj)


def model_test_batch(root, res, rounds, names, download=True, seed=0):
    datasets = me.download_data(root, res, download=download)
    tests = []

    with open("test_output.json", "a+") as f:
        json.dump({'ran_with_parameters': names}, f, indent=4)
        try:
            ALLflag = False
            for i, [ds_name, ds] in enumerate(datasets):
                yesno = "Y"
                if not ALLflag:
                    yesno = input(
                            "DATASET {} of {} START? ALL/Y/N/STOP: "
                            .format(i, len(datasets)))
                    if yesno == "ALL":
                        ALLflag = True
                if yesno == "Y":
                    print("USING {} DATASET, LEN {}".format(ds_name,
                                                            len(ds(root))))
                    model = get_model(names[0], weights=names[1])
                    model.to(ca.TORCH_DEVICE)
                    model.eval()
                    test = None
                    names.append(ds_name)
                    out = model_create_equation(model, names, ds, res, rounds,
                                                root=root, alldefault=ALLflag)
                    test = {
                        'dataset': ds_name,
                        'ds_len': len(ds(root)),
                        'test_output': out}
                    reset_model_weights(model)

                    import shutil
                    from pathlib import Path
                    # remove runs directory that contains caches of models
                    # as it affects model weight performance and changes
                    # control
                    # file is in src so go up one to dir and remove runs
                    cache_path = Path(__file__).parent.resolve() / "runs"

                    if os.path.exists(cache_path):
                        shutil.rmtree(cache_path)

                    names.pop()
                    tests.append(test)
                    # just in case
                    model = None
                elif yesno == "STOP":
                    break

                if tests != []:
                    json.dump(tests, f, indent=4, cls=numpyencoder)
                    tests.pop()
        finally:
            if tests != []:
                json.dump(tests, f, indent=4, cls=numpyencoder)
                tests.pop()


def imagenet_test_batch(root, res, rounds, names, seed=0):
    from torchvision import datasets
    from pathlib import Path
    tests = []
    print("USING IMAGENET")

    imagenet_train_ds = datasets.ImageNet(Path(root) + "manual", split='train')

    model = get_model(names[0], weights=names[1])
    model.to(ca.TORCH_DEVICE)
    model.eval()

    out = model_create_equation(model, names, imagenet_train_ds, res, rounds)

    test = {
        'dataset': "IMAGENET2012",
        'ds_len': len(imagenet_train_ds),
        'test_output': out
    }
    tests.append(test)
    with open("imagenet_test_output.json", "a+") as f:
        dct = {'ran_with_parameters': names}
        if tests != []:
            json.dump([dct, tests], f, indent=4, cls=numpyencoder)
        else:
            json.dump(dct, f, indent=4, cls=numpyencoder)
