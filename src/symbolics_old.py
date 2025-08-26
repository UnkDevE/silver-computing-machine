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
"""


import numpy as np
import scipy.linalg as linalg

import src.model_extractor as me


# returns mutliplicants then powers
def get_unzip_coeffs(ndspline, max_inputs):
    muls = []
    pows = []
    for spline in ndspline:
        t_muls = []
        t_pows = []
        for eq in spline:
            expand = eq.expand()
            zipcoeffs = expand.coefficients(sparse=False)

            mul = zipcoeffs[0].n()
            # if linear
            pow = 0 if len(zipcoeffs) < 2 else zipcoeffs[1].n()

            t_muls.append(mul)
            t_pows.append(pow)

        t_muls = np.array(t_muls)
        t_pows = np.array(t_pows)

        # where multiplicant zero change to zero in power
        t_pows[t_muls == 0.0] = 0.0

        # if less than max inputs pad with zero
        if len(t_muls) < max_inputs:
            t_muls = np.pad(t_muls, (0, max_inputs), t_muls.shape[0],
                            constant_value=0)

        if len(t_pows) < max_inputs:
            t_pows = np.pad(t_pows, (0, max_inputs), t_pows.shape[0],
                            constant_value=0)

        muls.append(t_muls)
        pows.append(t_pows)

    # pad values so we have same sizes
    # this is ordered in symbolic x1 -> xn as inputs
    return np.dstack([np.array(muls), np.array(pows)])


# returns coeffecients in matrix form with [mutliplicants, powers]
def generate_bernstien_polys(params, lu_system):
    from scipy.special import binom
    from sage.all import var

    # use De Casteljau's algorithm
    # get the length of knots so we don't do too many iterations
    knots = params.shape[-1]
    # get output dim
    outdim = lu_system.shape[-1]

    # create vector x type of symbols representing input dimension
    syms = np.array([var("x" + str(i)) for i in range(knots)], dtype="object")

    coeffs = []
    for d in range(outdim):
        coeffs.append((1 - syms) * params[d] +
                      syms * params[d + 1 % outdim])

    coeffs = np.array(coeffs)
    bernoli = np.array([binom(n, v)
                        for v, n in enumerate(reversed(range(knots)))])

    eq = bernoli * coeffs

    # remove symbolics and just use coefficients
    coeffs = get_unzip_coeffs(eq, knots)

    return [coeffs, syms, eq]


def generate_readable_eqs(sol_system, bspline, name):
    # init symbol system bspline has two args
    coeffs, syms, eq = generate_bernstien_polys(bspline[1], sol_system)

    # this now solves for polynomial space
    # now solve each simultaneous equation of tensor output dim * 2
    # first coefficients are all zero due to linearity
    new_shape = list(coeffs.shape)
    new_shape[0] -= 1

    coeffs = np.delete(coeffs, 0, 0)
    coeffs = np.reshape(coeffs, new_shape)
    coeffs = coeffs.T

    # solve to singular constant value
    def svd_lu(lu):
        # use svd to get components
        r_basis, nul, l_scale = linalg.svd(lu)

        # solve components scaling for new basis
        # premute is set to true so no rounding errs occur
        r_factor = linalg.lu(r_basis, permute_l=True, p_indices=True)
        l_factor = linalg.lu(l_scale, permute_l=True, p_indices=True)

        # rhs == 0 here due to factor decomposition i.e. solve(n, 0) == 0
        # therefore r_basis is our inverse
        new_basis = r_factor[0] @ r_factor[1] @ r_basis
        new_scale = l_factor[0] @ l_factor[1] @ l_scale

        # recreate diagonal so we can use to create same shape
        mix = linalg.diagsvd(nul, *lu.shape)

        return new_basis @ mix @ new_scale

    # now we need to solve this via svd and LU
    # no need for pivot
    mul_svd = svd_lu(coeffs[0])
    pow_svd = svd_lu(coeffs[1])

    from sage.all import PolynomialRing, QQ, latex
    from sage.rings.polynomial import polynomial_ring
    mat_ring = PolynomialRing(QQ, names=list(syms))

    # this gets round an outdated library using deprecated functions
    def is_ring(ring):
        return isinstance(ring, polynomial_ring.PolynomialRing_generic)

    setattr(polynomial_ring, 'is_PolynomialRing', is_ring)
    # this import HAS to be after setattr for it to work
    from rec_sequences.FunctionalEquation import FunctionalEquation
    breakpoint()
    mat_syms = np.reshape(np.repeat(syms, mul_svd.shape[-1]), mul_svd.shape)
    mat_eq = np.dstack([mul_svd, mat_syms, pow_svd])

    # find reoccurance relations for mat_eq
    algebras = FunctionalEquation(mat_ring, mat_eq)

    me.save("EQ.tex", latex(algebras))

    return algebras
