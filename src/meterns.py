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

from torchvision.transforms.v2 import Grayscale, GuassianBlur

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
    gray = Grayscale.forwards(img)
    # use padding to keep size
    contrast = F.conv2d(gray, laplacian_kernel, padding=1)
    saturation = torch.std(img, keepdim=True)
    # exposure algorithm is how close exp is to 0.5 in Guass curve
    exposure = torch.sqrt((torch.log(img)) * 2 * (SIGMA ** 2)) + 0.5
    return contrast * saturation * exposure


def meterns(imgs, dims):
    qs = [F.normalize(quality(img)) for img in imgs]
    # compute blurs and laplace pyramid
    blurs = [GuassianBlur.forwards(imgs)]
    laplaces = []

    for i in range(dims - 1):
        blurs.append(GuassianBlur.forwards(blurs[-1]))



    return qs
