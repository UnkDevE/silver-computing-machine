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


    pytorch module - HDR preprocessing of spline outputs
"""
import torch
import torch.nn.functional as F
from torch.func import hessian, vmap, functional_call

from torchvision.transforms.v2 import Grayscale, GaussianBlur

import numpy as np

from src import cech_algorithm as ca

# defined from merge meterns paper
# https://onlinelibrary.wiley.com/doi/abs/10.1111/j.1467-8659.2008.01171.x
SIGMA = 0.2


def round_up_to_odd(f):
    return np.ceil(f) // 2 * 2 + 1


class HDRMaskTransform(object):
    """Hdr resample the splined solved sample

    Args:
        spline (bspline object): spline object to call when using saved sample
    """

    # define laplace via hessian
    def get_laplace(self, model):
        # params (need to be an input for torch.func to work)
        params = dict(model.named_parameters())

        # functionalize version (params are now an input to the model)
        def fcall(params, x):
            return functional_call(model, params, x)

        def compute_laplacian(params, img):
            # forward-over-reverse hessian calc.
            hessian_ = hessian(fcall, argnums=1)(params, img)
            # use relative dims for vmap
            # (function doesn't see the batch dim of the input)
            _laplacian = hessian_.diagonal(0, -2, -1)
            return _laplacian

        # We define a laplacian func for a single function,
        # then vectorize over the batch.
        from functools import partial
        laplacian = partial(vmap(compute_laplacian, in_dims=(None, 0))(params))

        return laplacian

    # QUALITY MEASURES
    def quality(self, img):
        Grays = Grayscale()
        gray = Grays(img)
        # convert from numpy
        gray = torch.from_numpy(gray)
        # use padding to keep size
        contrast = self.laplace(gray)
        saturation = torch.std(img, keepdim=True)
        # exposure algorithm is how close exp is to 0.5 in Guass curve
        exposure = torch.sqrt((torch.log(img)) * 2 * (SIGMA ** 2)) + 0.5
        return contrast * saturation * exposure

    def laplace_pyramid(self, imgs, dims, Guass):
        blurs = [Guass(imgs)]
        laplaces = []

        for _ in range(dims - 1):
            blurs.append(Guass(blurs[-1]))
            # upsample current blur to last size
            upsample = F.upsample(blurs[-1], blurs[-2].size)
            laplaces.append(blurs[-2] - upsample)

        return laplaces

    def meterns(self, imgs, dims):
        Guass = GaussianBlur(kernel_size=dims)
        qs = F.normalize(self.quality(imgs))

        # compute blurs and laplace pyramid
        blurs = [Guass(qs)]
        for _ in range(dims - 1):
            blurs.append(Guass(blurs[-1]))

        laplaces = self.laplace_pyramid(imgs, dims, Guass)

        # create partials
        partials = [sum(laplace * blur) for laplace,
                    blur in zip(laplaces, blurs)]
        partials.reverse()

        image = None
        for i in range(len(partials), 1):
            upsample = F.upsample(partials[i - 1], partials[i].size())
            image = partials[i] + upsample

        return image

    def __init__(self, spline, model):
        self.spline = spline
        self.laplace = self.get_laplace(model)

    def __call__(self, sample):
        # WE HAVE TO USE NUMPY HERE SO THAT TORCH DOES NOT FORK JAX
        sample = sample.numpy().squeeze().T
        mask_samples = self.spline(sample)

        rep_shape = ca.product(mask_samples.shape[:(
            len(mask_samples.shape) - len(sample.shape))])

        solved_samples = np.repeat(
            sample,
            rep_shape).reshape(mask_samples.shape)

        # we want in full colour but dunno how to do that
        # check model, reshape inputs
        mask = mask_samples < solved_samples
        applied_samples = np.where(mask, solved_samples, 0)

        # hdr code here
        # no need for opencv as meterns is quite simple
        imgs = np.asarray(applied_samples)

        # kernel has to be odd for guass to work
        hdr = self.meterns(imgs, round_up_to_odd(imgs.shape[0]))
        # no need for exposure times
        return hdr
