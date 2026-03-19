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


    pytorch module - trains models
"""

import os
import random
import importlib

import torch

# torch for tensor LU
from torch.utils.data import DataLoader
import torch.linalg as t_linalg

import jax.scipy.linalg as j_linalg
import jax.numpy as jnp

import numpy as np
import torchcurves as tc

import src.cech_algorithm as ca

DL_WORKERS = os.cpu_count() // 2


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


# import class e.g. loss or opt and return it
def get_class(module, classname):
    return getattr(importlib.import_module(module), classname)


def epoch(model, epochs, names, train, test):
    # init writer loss and opt
    loss_fn = get_class("torch.nn", names[2])()
    opt = get_class("torch.optim", names[3])(model.parameters())
    best_vloss = 1_000_000.

    def train_one_epoch(epoch_num):
        running_loss = 0.
        last_loss = 0.

        # Here, we use enumerate(train) instead of
        # iter(train) so that we can track the batch
        # index and do some intra-epoch reporting
        for i, data in enumerate(train):
            # Every data instance is an input + label pair
            if len(data) < 2:
                continue

            inputs, labels = data
            inputs = inputs.to(ca.TORCH_DEVICE, non_blocking=True)
            labels = labels.to(ca.TORCH_DEVICE, non_blocking=True)

            # Zero your gradients for every batch!
            opt.zero_grad()

            # Make predictions for this batch
            outputs = model(inputs)

            # Compute the loss and its gradients
            loss = loss_fn(outputs, labels)
            loss.backward()

            # Adjust learning weights
            opt.step()

            # Gather data and report
            running_loss += loss.item()
            if i % 1000 == 999:
                last_loss = running_loss / 1000  # loss per batch
                print('  batch {} loss: {}'.format(i + 1, last_loss))
                running_loss = 0.

        return last_loss

    for epoch_number, epoch in enumerate(range(epochs)):
        print('EPOCH {}:'.format(epoch_number))

        # Make sure gradient tracking is on, and do a pass over the data
        model.train(True)
        avg_loss = train_one_epoch(epoch_number)

        running_vloss = 0.0
        # Set the model to evaluation mode, disabling dropout and
        # using population statistics for batch normalization.
        model.eval()

        # Disable gradient computation and reduce memory consumption.
        with torch.no_grad():
            for i, vdata in enumerate(test):
                vinputs, vlabels = vdata
                vinputs = vinputs.to(ca.TORCH_DEVICE, non_blocking=True)
                vlabels = vlabels.to(ca.TORCH_DEVICE, non_blocking=True)
                voutputs = model(vinputs)
                vloss = loss_fn(voutputs, vlabels)
                running_vloss += vloss

        avg_vloss = running_vloss / (epoch_number + 1)
        print('LOSS train {} valid {}'.format(avg_loss, avg_vloss))

        # Track best performance
        if avg_vloss < best_vloss:
            best_vloss = avg_vloss

        if ca.TORCH_DEVICE == "cuda":
            torch.cuda.empty_cache()

    return model


# refuse to reload samples as it will re randomize the output
class ClassLabelWrapper(object):
    def __init__(self):
        with open("imagenet1000_clsidx_to_labels.json") as classes:
            import json
            self.targets = json.load(classes)
            # reverse dict for ease of use
            self.targets = {v: i for i, v in self.targets.items()}

    def __call__(self, ysub):
        if ysub is not list:
            ysub = [ysub]

        y = []
        if type(ysub[0]) is not str:
            y = ysub
        else:
            y = np.array([self.targets[v] for v in
                          self.targets.keys() if v in ysub])

        one_hot = np.zeros(len(self.targets) + 1)
        if np.size(y) != 0:
            one_hot[int(y[0])] = 1.0

        hot_y = torch.from_numpy(one_hot)

        return hot_y


def collate_fn(batch):
    pairs = [(b[0], b[-1]) for b in batch if len(b) > 1]
    return [torch.stack(t) for t in list(zip(*pairs))]


# PYTORCH CODE ONLY
def interpolate_model_train(model, train, step, names):
    print("STEP {}".format(step))
    # setup for training loop
    # works for IMAGENET ONLY
    from torch.utils.data import random_split
    # random split for training
    [train_s, test_s] = random_split(train, [0.7, 0.3], generator=ca.GENERATOR)

    from src.model_extractor import BATCH_SIZE
    # collate fn None raises errors so we use default collate to force
    # collation
    train_s = DataLoader(train_s,
                         pin_memory=True, persistent_workers=True,
                         batch_size=BATCH_SIZE, num_workers=DL_WORKERS,
                         worker_init_fn=seed_worker, collate_fn=collate_fn)

    test_s = DataLoader(test_s, pin_memory=True, persistent_workers=True,
                        batch_size=BATCH_SIZE, num_workers=DL_WORKERS,
                        worker_init_fn=seed_worker, collate_fn=collate_fn)

    # training loop
    epoch(model, 5, names, train_s, test_s)

    return model


@torch.compile
def product(xs):
    y = xs[0]
    for x in xs[1:]:
        y *= x
    return y


def make_spline(sols, names, train_s, test_s):
    ins = jnp.array(sols[0])
    # out = jnp.array(sols[1])
    # out is already a diagonalized matrix of 1s
    # so therefore the standard basis becomes 0
    _, _, std_basis = j_linalg.svd(ins)
    # solve for the new std_basis
    new_basis = j_linalg.inv(std_basis)
    # create LU Decomposition towards new_basis
    jaxt = ca.jax_to_tensor(jnp.outer(new_basis, ins))
    lu_decomp = t_linalg.lu_factor_ex(jaxt)

    # get shapes
    input_dim = train_s[0][0].shape
    print(input_dim)
    intermediate_dim = lu_decomp[0].T.shape
    print(intermediate_dim)
    knots = product(intermediate_dim) + 1
    breakpoint()

    # interpolate
    kan = torch.nn.Sequential(
        # layer 1
        tc.BSplineCurve(len(intermediate_dim),
                        dim=len(input_dim), knots_config=knots,
                        normalize_fn='rational'),
        # layer 2
        tc.BSplineCurve(len(input_dim),
                        dim=len(input_dim)+1, degree=len(input_dim)+1,
                        knots_config=knots, normalize_fn='rational'),
        # layer 3
        tc.BSplineCurve(len(input_dim),
                        dim=len(input_dim), knots_config=knots,
                        normalize_fn='rational'))

    return interpolate_model_train(kan, train_s, 0, names)
