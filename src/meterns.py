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

from torchvision.transforms.v2 import Grayscale, GaussianBlur

import numpy as np

# defined from merge meterns paper
# https://onlinelibrary.wiley.com/doi/abs/10.1111/j.1467-8659.2008.01171.x
SIGMA = 0.2


def round_up_to_odd(f):
    return np.ceil(f) // 2 * 2 + 1


def next_odd_if_even(x):
    if x % 2 == 0:
        return round_up_to_odd(x)
    return x


def product(xs):
    y = xs[0]
    for x in xs[1:]:
        y *= x
    return y


class HDRMaskTransform(object):
    """Hdr resample the splined solved sample

    Args:
        spline (bspline object): spline object to call when using saved sample
    """

    # QUALITY MEASURES
    def quality(self, img):
        gray = img.detach().clone()
        if len(img.size()) >= 3:
            gray = Grayscale(num_output_channels=3)(gray)

        # use calculate second order deriviatives (laplacian) by autograd
        contrast = sum(list(torch.gradient(sum(list(torch.gradient(gray))))))
        saturation = torch.std(img)
        # exposure algorithm is how close exp is to 0.5 in Guass curve
        exposure = torch.sqrt((torch.log(img)) * 2 * (SIGMA ** 2)) + 0.5

        return contrast * saturation * exposure

    def laplace_pyramid(self, imgs, dims, Guass):
        blurs = [Guass(imgs)]
        laplaces = []

        for _ in range(dims - 1):
            blurs.append(Guass(blurs[-1]))
            # upsample not needed done already from pytorch
            laplaces.append(blurs[-2] - blurs[-1])

        return laplaces

    def meterns(self, imgs, dims):
        # torch tensor needs to be flipped because it is flipped somewhere
        # this causes transforms to raise an error that there are too many
        # values to unpack this isn't true
        ar_size = list(imgs.size()[1:])
        ar_size.reverse()
        imgs = imgs.reshape([imgs.size()[0], *ar_size])

        Guass = GaussianBlur(kernel_size=dims, sigma=(SIGMA, 0.5))
        qs = [F.normalize(self.quality(img)) for img in imgs]

        # compute blurs and laplace pyramid
        blurs = [Guass(qs)]
        for _ in range(dims - 1):
            blurs.append(Guass(blurs[-1]))
        blurs = [x for xs in blurs for x in xs]

        laplaces = self.laplace_pyramid(imgs, dims, Guass)

        # create partials
        partials = [laplace * blur for (laplace,
                    blur) in list(zip(laplaces, blurs))]
        partials.reverse()

        image = None
        for i in range(len(partials), 1):
            image = partials[i] + partials[i - 1]

        return image

    def __init__(self, spline):
        self.spline = spline

    def __call__(self, sample):
        # WE HAVE TO USE NUMPY HERE SO THAT TORCH DOES NOT FORK JAX
        sample_np = sample.numpy().squeeze().T
        mask_samples = self.spline(sample_np)
        t_mask_samples = torch.tensor(mask_samples)

        rep_shape = product(mask_samples.shape[:(
            len(mask_samples.shape) - len(sample_np.shape))])

        solved_samples = torch.tensor(sample_np.repeat(
            rep_shape).reshape(mask_samples.shape))

        # we want in full colour but dunno how to do that
        # check model, reshape inputs
        mask = t_mask_samples.le(solved_samples)
        imgs = torch.where(mask, solved_samples, 0)

        # hdr code here
        # no need for opencv as meterns is quite simple

        # kernel has to be odd for guass to work
        hdr = self.meterns(imgs, next_odd_if_even(len(imgs.shape)))
        # no need for exposure times
        return hdr
