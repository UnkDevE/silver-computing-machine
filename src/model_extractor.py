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

from torchvision import datasets


# looks up each activation from csv and then defines a function to it
def activation_fn_lookup(activ_src, csv):
    if activ_src is None:
        return csv['function']['linear']
    sourcefnc = activ_src.__name__
    for (i, acc) in enumerate(csv['activation']):
        if acc == sourcefnc:
            return (csv['function'][i])
    return csv['function']['linear']


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
    download_ds = download
    for ds_name in datasets_names:
        ds = datasets.__dict__[ds_name]
        sig = inspect.signature(ds.__init__)
        kwargs_ds = [k for k, param in sig.parameters.items()
                     if param.kind != param.POSITIONAL_ONLY]
        args_ds = [p for p in sig.parameters.keys() if p not in kwargs_ds]

        if sig.parameters.get("download") is not None:
            if download:
                yesno = input("DOWNLOADING {} Y/N/STOP: ".format(ds_name))
                if yesno == "N":
                    download_ds = False
                elif yesno == "STOP":
                    download = False
                    download_ds = False

            try:
                if "download" in args_ds:
                    datasets.__dict__[ds_name](
                        dataset_root, download_ds)
                elif "download" in kwargs_ds:
                    datasets.__dict__[ds_name](
                        dataset_root, download=download_ds)

                ds_list.append([ds_name, datasets.__dict__[ds_name]])
            except Exception as e:
                if download:
                    print("DATASET: {} download err {}".format(ds_name, e))
                else:
                    pass  # do not throw for not downloaded datasets

    return ds_list
