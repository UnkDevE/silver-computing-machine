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


   Torch dataset wrangler
"""

import torch
from torchvision.transforms import v2
from torchvision import datasets

import numpy as np

# something sane here
# 512 just over 24 GB of VRAM
BATCH_SIZE = 512 


# looks up each activation from csv and then defines a function to it
def activation_fn_lookup(activ_src, csv):
    if activ_src is None:
        return csv['function']['linear']
    sourcefnc = activ_src.__name__
    for (i, acc) in enumerate(csv['activation']):
        if acc == sourcefnc:
            return (csv['function'][i])
    return csv['function']['linear']


def label_extract(label_extract, features):
    labels = []
    for feature in features:
        for label in label_extract:
            if hasattr(label[feature], 'numpy') and callable(getattr(
                    label[feature], 'numpy')):
                labels.append(label[feature].numpy())
    return labels


def bucketize_features(model, dataset):
    # get label and value
    features = dataset.classes
    # filter through to find duplicates
    values_removed = dataset.unique()

    # call unique on features
    need_extract = values_removed.unique()

    labels = label_extract(need_extract, features)

    # have to update cardinality each time
    # remember labels is a list due to extract

    def condense(v):
        b = True
        for i in v:
            if not i:
                b = False
        return b

    # condense doesn't work as complains about python bool

    # bucketize each feature in each label, return complete datapoints
    # bucketizing is failing at the moment because the labels are consumed
    sets = [dataset.filter(lambda i:
            condense([i[feature] == label for feature in features]))
            for label in labels]

    # numpy array of predictions
    inputimages = []
    tensors = []
    for dataset in sets:
        # get the images
        batch = dataset.padded_batch(BATCH_SIZE, drop_remainder=True)
        # samples not normalized
        for sample in batch:
            inputimages.append(sample)
            prediction = model.predict(sample)
            # divide by 10 because output is in form n instead of 1\n
            tensors.append(np.sum(prediction,
                                  axis=0) / prediction.shape[0] / 10)

    return list(zip(labels, zip(inputimages, tensors)))


def get_labels(names):
    from torchvision.models import get_weight
    weights = get_weight("{m}_Weights.{w}"
                         .format(m=names[0].upper(), w=names[1]))
    categories = weights.meta["categories"]
    return categories


# download all inbuilt datasets and construct them
def download_data(dataset_root, res, download=True):
    import inspect
    datasets_names = [ds for ds in datasets.__dict__.keys()
                      if inspect.isclass(datasets.__dict__[ds])]
    # multiple filters have to be done in seq because and is not short
    # circuiting
    ds_list = []
    for ds_name in datasets_names:
        ds = datasets.__dict__[ds_name]
        sig = inspect.signature(ds.__init__)
        kwargs_ds = [k for k, param in sig.parameters.items()
                     if param.kind != param.POSITIONAL_ONLY]
        args_ds = [p for p in sig.parameters.keys() if p not in kwargs_ds]

        if sig.parameters.get("download") is not None:
            try:
                dataset = None
                if "download" in args_ds:
                    dataset = datasets.__dict__[ds_name](
                        dataset_root, download)
                elif "download" in kwargs_ds:
                    dataset = datasets.__dict__[ds_name](
                        dataset_root, download=download)

                dataset.transform = v2.Compose([v2.ToImage(),
                                                v2.ToDtype(torch.float32,
                                                           scale=True),
                                                v2.Resize([res, res])])
                ds_list.append(dataset)
            except Exception as e:
                print("dataset download did not work not appending...")
                print("err: " + str(e))

    return ds_list
