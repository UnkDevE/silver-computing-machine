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
import importlib
from datetime import datetime

# jax for custom code
import jax.numpy as jnp
import torch

# torch for tensor LU
import torch.linalg as t_linalg
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, Dataset

from torchvision.transforms.v2 import ToDtype, Compose

import jax.scipy.linalg as j_linalg
import numpy as np

import src.cech_algorithm as ca
from src.meterns import HDRMaskTransform

DL_WORKERS = os.cpu_count() - 2


# import class e.g. loss or opt and return it
def get_class(module, classname):
    return getattr(importlib.import_module(module), classname)


def make_spline(sols):
    # get shapes
    interpol_shape = sols[-1].shape
    # sensible names
    ins = jnp.array(sols[0])
    # out = jnp.array(sols[1])
    from scipy.interpolate import make_splprep
    # out is already a diagonalized matrix of 1s
    # so therefore the standard basis becomes 0
    _, _, std_basis = j_linalg.svd(ins)
    # solve for the new std_basis
    new_basis = j_linalg.inv(std_basis)
    # create LU Decomposition towards new_basis
    jaxt = ca.jax_to_tensor(jnp.outer(new_basis, ins))

    lu_decomp = t_linalg.lu_factor_ex(jaxt)
    # interpolate
    lu_decomp = [decomp.detach().numpy() for decomp in lu_decomp]
    # spline shaping err
    [spline, u] = make_splprep(lu_decomp[0].T, k=sum(interpol_shape) + 1)

    return [spline, u, lu_decomp]


def epoch(model, epochs, names, train, test):
    # init writer loss and opt
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    writer = SummaryWriter('runs/imagenet_{}'.format(timestamp))
    loss_fn = get_class("torch.nn", names[2])()
    opt = get_class("torch.optim", names[3])(model.parameters())
    best_vloss = 1_000_000.

    def train_one_epoch(epoch_num, tb_writer):
        running_loss = 0.
        last_loss = 0.

        # Here, we use enumerate(train) instead of
        # iter(train) so that we can track the batch
        # index and do some intra-epoch reporting
        for i, data in enumerate(train):
            # Every data instance is an input + label pair
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
                tb_x = epoch_num * len(train) + i + 1
                tb_writer.add_scalar('Loss/train', last_loss, tb_x)
                running_loss = 0.

        return last_loss

    for epoch_number, epoch in enumerate(range(epochs)):
        print('EPOCH {}:'.format(epoch_number))

        # Make sure gradient tracking is on, and do a pass over the data
        model.train(True)
        avg_loss = train_one_epoch(epoch_number, writer)

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

        # Log the running loss averaged per batch
        # for both training and validation
        writer.add_scalars('Training vs. Validation Loss',
                           {'Training': avg_loss, 'Validation': avg_vloss},
                           epoch_number + 1)
        writer.flush()

        # Track best performance, and save the model's state
        if avg_vloss < best_vloss:
            best_vloss = avg_vloss
            model_path = '{}_{}_{}'.format("".join(names), timestamp,
                                           epoch_number)
            torch.save(model.state_dict(), model_path)

    return model


# refuse to reload samples as it will re randomize the output
class TransformDatasetWrapper(Dataset):
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform

        with open("imagenet1000_clsidx_to_labels.json") as classes:
            import json
            self.targets = json.load(classes)
            # reverse dict for ease of use
            self.targets = {v: i for i, v in self.targets.items()}

    def __getitem__(self, index):
        x, ysub = self.subset[index]
        if self.transform:
            x = self.transform(x)

        y = np.array([int(self.targets[v]) for v in
                     self.targets.keys() if v in ysub])

        one_hot = np.zeros(len(self.targets) + 1)
        # classifier is not multiclass
        if np.size(y) != 0:
            one_hot[y[0]] = 1.0

        hot_y = torch.from_numpy(one_hot)

        return x, hot_y

    def __len__(self):
        return len(self.subset)


# PYTORCH CODE ONLY
def interpolate_model_train(spline, model, train, step, names):
    print("STEP {}".format(step))
    # setup for training loop
    # re-transform dataset with spline & HDR resample
    tds = TransformDatasetWrapper(train,
                                  transform=Compose([
                                      ToDtype(torch.float32, scale=True),
                                      HDRMaskTransform(spline)]))

    from torch.utils.data import random_split
    # random split for training
    [train_s, test_s] = random_split(tds, [0.7, 0.3], generator=ca.GENERATOR)

    from src.model_extractor import BATCH_SIZE
    train_s = DataLoader(train_s,
                         batch_size=BATCH_SIZE, num_workers=DL_WORKERS)

    test_s = DataLoader(test_s, batch_size=BATCH_SIZE, num_workers=DL_WORKERS)

    # training loop
    epoch(model, 5, names, train_s, test_s)

    return model
