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
from torchvision.transforms.v2 import Grayscale, GaussianBlur, Transform

import numpy as np

# defined from merge meterns paper
# https://onlinelibrary.wiley.com/doi/abs/10.1111/j.1467-8659.2008.01171.x
SIGMA = 0.2

DS_COUNT = 0


def round_up_to_odd(f):
    return np.ceil(f) // 2 * 2 + 1


def next_odd_if_even(x):
    if x % 2 == 0:
        return round_up_to_odd(x)
    return x


@torch.compile
def product(xs):
    y = xs[0]
    for x in xs[1:]:
        y *= x
    return y


class HDRDummyTransform(Transform):
    def __init__(self, spline):
        self.spline = spline

    def __call__(self, sample):
        # WE HAVE TO USE NUMPY HERE SO THAT TORCH DOES NOT FORK JAX
        sample_np = sample.numpy().squeeze().T
        mask_samples = self.spline(sample_np)
        t_mask_samples = torch.tensor(mask_samples)
        return t_mask_samples


def _tovid(imgs, name, w, h):
    import ffmpeg
    vid_p = (
        ffmpeg
        .input('pipe:', format='rawvideo', pix_fmt='rgb24', s='{}x{}'.
               format(w, h))
        .output(name+".mp4", pix_fmt='rgb24')
        .overwrite_output()
        .run_async(pipe_stdin=True, quiet=True)
    )

    for im in imgs:
        img = im.numpy()
        img = img.astype(np.uint8)
        vid_p.stdin.write(img.tobytes())

    vid_p.stdin.close()
    vid_p.wait()


class HDRMaskTransform(Transform):
    """Hdr resample the splined solved sample

    Args:
        spline (bspline object): spline object to call when using saved sample
    """

    # QUALITY MEASURES
    def quality(self, img):
        gray = img.detach().clone()
        if len(img.size()) <= 3:
            gray = Grayscale(num_output_channels=3)(gray)

        # use calculate second order deriviatives (laplacian) by autograd
        contrast = sum(list(torch.gradient(sum(list(torch.gradient(gray))))))
        saturation = torch.std(img)
        # exposure algorithm is how close exp is to 0.5 in Guass curve
        exposure = torch.exp(-((img - 0.5) / (SIGMA ** 2)))

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
        Guass = GaussianBlur(kernel_size=dims, sigma=(SIGMA, 0.5))
        qs = [self.quality(img) for img in imgs]

        # compute blurs and laplace pyramid
        blurs = [Guass(qs)]
        for _ in range(dims - 1):
            blurs.append(Guass(blurs[-1]))
        blurs = [F.normalize(x) for xs in blurs for x in xs]

        laplaces = self.laplace_pyramid(imgs, dims, Guass)

        # create partials
        partials = [laplace * blur for (laplace,
                    blur) in list(zip(laplaces, blurs))]
        partials.reverse()

        image = torch.empty_like(partials[0])
        for i in range(1, len(partials)):
            n = len(partials) - i
            image += partials[n] * partials[n - 1]

        return F.normalize(image.sum(0))

    def __init__(self, save_vid=False, names=[]):
        self.save_vid = save_vid
        self.names = names
        super().__init__()

    # no params needed
    def transform(self, imgs, _):
        if self.save_vid:
            _tovid(imgs, "{}_{}".format("".join(self.names),
                                        DS_COUNT),
                   imgs[0].shape[1], imgs[0].shape[2])

        # kernel has to be odd for guass to work
        hdr = self.meterns(imgs, next_odd_if_even(len(imgs.shape)))
        # no need for exposure times
        isnan = torch.any(torch.isnan(hdr))
        print("HDR NaN: {}".format(isnan))
        return hdr
