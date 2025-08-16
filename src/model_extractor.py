import sys
import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor


from scipy.fft import rfftn
import numpy as np
from scipy import linalg
import matplotlib.pyplot as plt

# something sane here
BATCH_SIZE=1024

def product(x):
    out = 1
    for y in x:
        out *= y
    return out

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
    value, *features = dataset.numpy()

    # get types of labels
    # dataset.unique() doesn't work with uint8
    # so we remove the offending key and use it
    def rm_val(d, val):
        for v in val:
            if v in d:
                del d[v]
        return d

    # filter through to find duplicates
    values_removed = dataset.map(lambda i: rm_val(i, [value]))

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
        normalized = batch.map(lambda x: normalize_img(x['image']))
        for sample in normalized:
            inputimages.append(sample)
            prediction = model.predict(sample)
            # divide by 10 because output is in form n instead of 1\n
            tensors.append(np.sum(prediction,
                                  axis=0) / prediction.shape[0] / 10)

    return list(zip(labels, zip(inputimages, tensors)))


def get_features(features, value):
    xs = []
    for f in features:
        xs.append(value[f])

    return xs


def get_ds(dataset):
    # shuffle dataset so each sample is a bit different
    dataset.shuffle(BATCH_SIZE)
    # predict training batch, normalize images by 255
    value, *features = list(list(dataset.take(1).as_numpy_iterator())[0]
                            .keys())

    [images, labels] =
    return [images, labels]

def model_read_create(model, modelname):
    path = os.path.dirname(__file__)
    model = os.path.join(path, model)
    model_create_equation(os.path.abspath(model), modelname)

if __name__ == "__main__":
    try:
        if len(sys.argv) > 1:
            test_main(sys.argv[1])
    except FileNotFoundError:
        print("""file not found,
                  please give list of datasets to test from""")
