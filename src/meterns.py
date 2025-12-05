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

from torchvision.transforms.v2 import Grayscale, GaussianBlur

from torch import nn
import torch.nn.functional as F

# Define the 4 - neighbor negative Laplacian kernel
laplacian_kernel = torch.tensor([[[[0, -1, 0],
                                   [-1, 4, -1],
                                   [0, -1, 0]]]], dtype=torch.float32)

# defined from merge meterns paper
# https://onlinelibrary.wiley.com/doi/abs/10.1111/j.1467-8659.2008.01171.x
SIGMA = 0.2


# QUALITY MEASURES
def quality(img):
    Grays = Grayscale()
    gray = Grays(img)
    # convert from numpy
    gray = torch.from_numpy(gray)
    # use padding to keep size
    l_conv = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3,
                       bias=False)
    l_conv.weight = nn.Parameter(laplacian_kernel)

    contrast = l_conv(gray)
    saturation = torch.std(img, keepdim=True)
    # exposure algorithm is how close exp is to 0.5 in Guass curve
    exposure = torch.sqrt((torch.log(img)) * 2 * (SIGMA ** 2)) + 0.5
    return contrast * saturation * exposure


def laplace_pyramid(imgs, dims, Guass):
    blurs = [Guass(imgs)]
    laplaces = []

    for _ in range(dims - 1):
        blurs.append(Guass(blurs[-1]))
        # upsample current blur to last size
        upsample = F.upsample(blurs[-1], blurs[-2].size)
        laplaces.append(blurs[-2] - upsample)

    return laplaces


def meterns(imgs, dims):
    Guass = GaussianBlur(kernel_size=dims)
    qs = F.normalize(quality(imgs))

    # compute blurs and laplace pyramid
    blurs = [Guass(qs)]
    for _ in range(dims - 1):
        blurs.append(Guass(blurs[-1]))

    laplaces = laplace_pyramid(imgs, dims, Guass)

    # create partials
    partials = [sum(laplace * blur) for laplace, blur in zip(laplaces, blurs)]
    partials.reverse()

    image = None
    for i in range(len(partials), 1):
        upsample = F.upsample(partials[i - 1], partials[i].size())
        image = partials[i] + upsample

    return image
