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


def test_pval(control, predict):
    if np.any(stats.kstest(control, predict).pvalue > PVALUE_SMALL):
        test = stats.mannwhitneyu(control, predict, axis=None)
    else:
        test = ttost_ind(control, predict, EQUIVALENCE_BOUND)

    return test


@torch.compile()
def test_spline(bspline, dataset, samples_percent,
                check_spline=False):

    ds_len = len(dataset)
    fast_evaluator = bspline['cpu']
    slow_evaluator = None
    if check_spline:
        slow_evaluator = bspline['scipy']

    # cheat to get class_to_idx as there is no standard of getting labels
    f_img, labels = dataset[0]
    no_classes = len(labels)

    class_outs = []
    tests = []
    zero_weight = np.zeros_like(labels)

    from torch.utils.data import WeightedRandomSampler, RandomSampler
    if check_spline:
        sampler = RandomSampler(dataset,
                                num_samples=(ds_len // 100) * samples_percent,
                                generator=ca.GENERATOR)
        loader = DataLoader(dataset, batch_size=BATCH_SIZE, sampler=sampler
                            )
        for img, k in loader:
            fast = fast_evaluator(img)
            slow = slow_evaluator(img.numpy())
            tests.append(test_pval(slow, fast))

        if tests != []:
            pval = stats.combine_pvalues(tests)
            with open("spline_acc.json", "a+") as spf:
                json.dump(pval, spf, cls=numpyencoder)

    fast_evaluator = bspline['gpu']
    for i in range(no_classes):
        class_weight = zero_weight
        class_weight[i] = 1.
        sampler = WeightedRandomSampler(class_weight,
                                        samples_percent,
                                        replacement=False,
                                        generator=ca.GENERATOR
                                        )
        loader = DataLoader(dataset, batch_size=samples_percent,
                            sampler=sampler)

        outs = []
        for img, k in loader:
            img_gpu = img.to(ca.TORCH_DEVICE, non_blocking=True)
            fast = fast_evaluator(img_gpu)
            outs.append(fast)

        outs_cpu = [o.cpu() for o in outs]
        class_outs.append((i, outs_cpu))

    return tests, class_outs


def draw_spline(name, bspline, dataset, class_outs):
    f_img, labels = dataset[0]
    no_classes = len(torch.unique(labels))

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

    plt.savefig("space_"+name)
    # clear figures and axes
    plt.cla()
    plt.clf()

    cl_avg = []
    for cl, outs in class_outs:
        # move batch to iter access
        for batch in outs:
            # move batch from to first position instead of piecewise
            avg = []
            batch = np.moveaxis(batch, 1, 0)
            for outpiecewise, shape in zip(batch, shapes):
                for out in outpiecewise:
                    distance = []
                    for sp in shape:
                        # norm comes out 1d
                        distance.append((np.linalg.norm(out - sp),
                                         cl // no_classes))
                    avg.append(np.average(distance, axis=0))
        cl_avg.append(np.average(avg))

    plt.violinplot(cl_avg)

    plt.savefig(name)
    # clear figures and axes
    plt.cla()
    plt.clf()


def product(xs):
    out = 1
    for x in xs:
        out *= x
    return out


def spline_features(dataset, bspline, test=False):
    # we want class_outs here as output to use for feature extraction
    _, class_outs = test_spline(bspline, dataset, 20, check_spline=False)

    from sklearn.cluster import MiniBatchKMeans
    kmeans = MiniBatchKMeans(n_clusters=len(class_outs))
    bikmeans = MiniBatchKMeans(n_clusters=len(class_outs))
    tests = []

    def testtrain(test=False):
        for feature, spline_out in class_outs:
            # inverse the spline function using point matrix as
            # a masked transform
            from src.training import pointmatrix
            # remove last from stack as has changed shape
            if len(spline_out) != 1:
                spline_out = np.stack(spline_out[:-1])
            else:
                spline_out = spline_out[0]

            sp_out_last = spline_out[-1].reshape(-1, *spline_out[-1].shape[2:])
            spline_out = spline_out.reshape(product(spline_out.shape[:3]),
                                            *spline_out.shape[3:])
            spline_out = np.vstack([spline_out, sp_out_last])
            out = torch.Tensor(np.array(spline_out))
            uk_cpu = bspline['gpu'].uniq_knots.clone().detach().cpu()
            coef_cpu = bspline['gpu'].coeffs.clone().detach().cpu()
            # batch up outs
            from torch.utils.data import TensorDataset

            outs = TensorDataset(out)
            out_loader = DataLoader(outs, batch_size=BATCH_SIZE)

            out_pMxs = []
            # work on CPU for this one
            for [xout] in out_loader:
                out_pMxs = pointmatrix(
                        uk_cpu,
                        xout,
                        coef_cpu, device='cpu').numpy()

                # stack batched output

                # must start at 1 as we are using multiplication as the
                # coherent bicluster
                # in fact lets try it without this and test with the bicluster
                x = out_pMxs.reshape(-1, *out_pMxs.shape[-2:])
                fit = (feature + 1) * x
                if not test:
                    kmeans.partial_fit(x)
                    bikmeans.partial_fit(fit)
                else:
                    testa = kmeans.predict(x)
                    ctrl = kmeans.predict(fit)
                    # reverse feature bicluster
                    ctrl = ctrl // (feature + 1)
                    if np.any(stats.kstest(testa,
                                           ctrl).pvalue > PVALUE_SMALL):
                        tests.append(stats.mannwhitneyu(testa, ctrl,
                                                        axis=None))
                    else:
                        tests.append(ttost_ind(testa, ctrl,
                                               EQUIVALENCE_BOUND))

        testtrain()
        if test:
            testtrain(test=test)

        # save for inspection purposes
        with open("featureset_test_kmeans.json", "a+") as f:
            json.dump(tests, f, indent=4, cls=numpyencoder)

    return kmeans, tests


def model_create_equation(model, names, dataset, in_shape, test_rounds,
                          root='./datasets', alldefault=False):
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

        # set a label for equiv testing to ignore if edits go through
        label_eqvtest_ignore = -1

        if not alldefault:
            bspline = tr.make_spline(sols[-1])
            print("MAKING VISUAL AND CORRELATION MATRIX")
            check_spl = True if "Y" == input(
                    "TEST SPLINE2SPLINE ACC (V.SLOW) Y/N:") else False

            _, class_out = test_spline(bspline, dataset_train, 5,
                                       check_spline=check_spl)

            # draw_spline('{}.png'.format(''.join(names)), bspline,
            #            dataset_train, class_out)
            vidout_yn = ("Y" == input("VIDEO OUTPUT OF DS PRE-MERTENS Y/N:"))
            edit_yn = input("EDIT Y/N:")
            if edit_yn == "Y":
                testacc_yn = ("Y" == input(
                    "GENERATING FEATURE SET, TEST ACC Y/N:"))
                pred, _ = spline_features(dataset_train, bspline,
                                          test=testacc_yn)
                print("FEATURES NO: {}".format(len(class_out)))
                ft_label = int(input("ENTER NTH LABEL WEIGHT EDIT: "))
                spline_out = class_out[ft_label][1][0].numpy()
                breakpoint()
                spline_out = spline_out.reshape(-1, spline_out.shape[-2:])
                breakpoint()
                ft_cluster = pred.predict(spline_out)
                redist = float(input("REDISTRIBUTION WEIGHT FLOAT: "))
                bspline.coeffs = ft_cluster[bspline.coeffs] * redist
        else:
            bspline = tr.make_spline(sols[-1])

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

            print("TESTING...")
            ctrls, t_tests, cvst = [[], [], []]
            ectrls, et_tests, ecvst = [[], [], []]
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

                if np.squeeze(actual) != label_eqvtest_ignore:
                    ctrls.append(ctrl_t)
                    t_tests.append(test_t)
                    cvst.append(tvsctrl)
                else:
                    ectrls.append(ctrl_t)
                    et_tests.append(test_t)
                    ecvst.append(tvsctrl)

            ctrl_t = stats.combine_pvalues(ctrls)
            test_t = stats.combine_pvalues(t_tests)
            tvsctrl = stats.combine_pvalues(cvst)

            print("EVAL VS ACT EQUIV:")
            print('{}'.format(ctrl_t))
            print("TEST VS ACT EQUIV:")
            print('{}'.format(test_t))
            print("TEST VS CTRL DIFF EQUIV:")
            print('{}'.format(tvsctrl))

            test_dict = {
                        'eval_pval': '{}'.format(ctrl_t[1]),
                        'test_pval': '{}'.format(test_t[1]),
                        'testvsctrl_pvalue': '{}'.format(tvsctrl[1]),
                        'randomseed': int(torch.initial_seed())
                }

            if label_eqvtest_ignore != -1:
                ectrl_t = stats.combine_pvalues(ectrls)
                etest_t = stats.combine_pvalues(et_tests)
                etvsctrl = stats.combine_pvalues(ecvst)

                diff_ct = abs(ectrl_t.pvalue - ctrl_t.pvalue)
                diff_et = abs(etest_t.pvalue - test_t.pvalue)
                diff_tve = abs(etvsctrl.pvalue - tvsctrl.pvalue)

                print("EDIT EVAL VS ACT EQUIV DIFF:")
                print('{}'.format(diff_ct))
                print("EDIT TEST VS ACT EQUIV DIFF:")
                print('{}'.format(diff_et))
                print("EDIT TEST VS CTRL DIFF EQUIV DIFF:")
                print('{}'.format(diff_tve))

                test_dict['diff_control_avg_vs_actual_diff'] = diff_ct
                test_dict['diff_test_vs_edit_diff'] = diff_ct
                test_dict['diff_ctrl_avg_vs_edit_diff'] = diff_ct

            tests.append(test_dict)
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
