#! python3
# -*- coding: utf-8 -*-
#
# problem.py
#
# Define Bloch eigenproblems in FENICSx.
#
# Created:       2023-07-26 13:03:21
# Last modified: 2023-07-29 13:08:49
#
# Copyright (c) 2023 Connor D. Pierce
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""
Bloch eigenproblems.

Classes
-------
BlochProblemType

Functions
---------
mass_form
stiffness_form
strain
stress
to_voigt
"""


__all__ = ["mass_form", "stiffness_form", "strain", "stress", "to_voigt"]


# Imports

import enum
import numpy as np
import typing
import ufl

from dolfinx.fem import Constant
from petsc4py.PETSc import ScalarType
from ufl import dot, dx, inner, sym, grad, Argument, Coefficient, Form


# Classes


class BlochProblemType(enum.IntEnum):
    """Types of Bloch eigenproblems.

    DIRECT - k(ω) formulation
    INDIRECT - ω(k) formulation, with Floquet-periodic $u(x)$ as eigenvector
    INDIRECT_TRANSFORMED - ω(k) formulation, periodic $\\hat{u}(x)$ as eigenvector
    """

    DIRECT: 1
    INDIRECT: 2
    INDIRECT_TRANSFORMED: 3

    # TODO: check if this conforms to the standard nomenclature in phononics literature


# Functions


def to_voigt(tensor):
    """Convert second-order tensor to Voigt notation.

    Parameters
    ----------
    - `tensor` second-order tensor to be converted to Voigt notation

    Returns
    -------
    vector representation of `tensor`
    """

    if tensor.ufl_shape[0] == 1:
        return ufl.as_vector([tensor[0, 0]])
    elif tensor.ufl_shape[1] == 2:
        return ufl.as_vector([tensor[0, 0], tensor[1, 1], 2 * tensor[0, 1]])
    elif tensor.ufl_shape[2] == 3:
        return ufl.as_vector(
            [
                tensor[0, 0],
                tensor[1, 1],
                tensor[2, 2],
                2 * tensor[1, 2],
                2 * tensor[2, 0],
                2 * tensor[0, 1],
            ]
        )


def strain(u: typing.Union[Argument, Coefficient], use_voigt: bool = False):
    """Infinitesimal strain tensor of a form argument or coefficient.

    Parameters
    ----------
    - `u` Function, TrialFunction, or TestFunction from which to calculate the
        (symmetric) infinitesimal strain
    - `use_voigt` return the strain in Voigt notation (as a vector) or as a tensor

    Returns
    -------
    the infinitesimal strain tensor
    """

    if use_voigt:
        return to_voigt(strain(u, False))
    else:
        return sym(grad(u))


def stress(
    u: typing.Union[Argument, Coefficient],
    modulus: Coefficient,
    use_voigt: bool = False,
):
    """Linear elastic stress tensor of a form argument or coefficient.

    Parameters
    ----------
    - `u` Function, TrialFunction, or TestFunction from which to calculate the stress
    - `modulus` stiffness (as a Function). If `use_voigt` is true, `modulus` must be the
        (square) stiffness matrix with appropriate shape for the problem dimension, e.g.
        $3 \\times 3$ for 2D problems. If `use_voigt` is false, `modulus` must be the
        fourth-order stiffness tensor with appropriate shape.
    - `use_voigt` return the stress in Voigt notation (as a vector) or as a tensor

    Returns
    -------
    the linear elastic stress tensor
    """

    if use_voigt:
        eps = strain(u, True)
        return dot(modulus, eps)
    else:
        eps = strain(u, False)
        ii, jj, kk, ll = ufl.i, ufl.j, ufl.k, ufl.l
        return modulus[ii, jj, kk, ll] * eps[kk, ll]


def mass_form(
    problem_type: BlochProblemType,
    u_tr: Argument,
    u_test: Argument,
    density: Coefficient,
    **kwargs,
) -> Form:
    """Create the mass bilinear form for the Bloch wave problem.

    The problem type is detected from the shape of the trial and test functions. Scalar
    trial functions trigger a Helmholtz-type problem to be returned, while vector trial
    functions trigger an elasticity-type problem to be returned. Elasticity problems
    allow for tensor-valued density.

    Parameters
    ----------
    - `problem_type` type of problem formulation
    - `u_tr` trial function of the problem
    - `u_test` test function of the problem
    - `density` scalar or tensor function representing the material density

    Returns
    -------
    `mass_form` - tuple containing of a single sesquilinear form or two bilinear forms
        representing the mass matrix
    """

    # TODO: is the mass matrix the same for all three formulations?

    if problem_type == BlochProblemType.DIRECT:
        if "wave_vec" not in kwargs:
            raise RuntimeError("`wave_vec` is required for direct problems")

        raise NotImplementedError("Direct formulation not implemented yet.")
    elif problem_type == BlochProblemType.INDIRECT:
        raise NotImplementedError(
            "Untransformed indirect formulation not implemented yet."
        )
    elif problem_type == BlochProblemType.INDIRECT_TRANSFORMED:
        if "wave_vec" not in kwargs:
            raise RuntimeError(
                "`wave_vec` is required for transformed indirect problems"
            )

        if len(u_tr.ufl_shape) == 0:
            # Scalar trial function; create Helmholtz problem
            return inner(density * u_tr, u_test) * dx
        elif len(u_tr.ufl_shape) == 1:
            # Vector trial function; create elasticity problem
            return inner(dot(density, u_tr), u_test) * dx
        else:
            raise ValueError(f"Unsupported trial function shape: {u_tr.ufl_shape}")
    else:
        raise ValueError(f"Unrecognized problem type: {problem_type}")


def _stiffness_form_direct(
    u_tr: Argument, u_test: Argument, modulus: Coefficient, wave_vec: Constant
):
    # Return the stiffness form for direct formulations

    raise NotImplementedError("Direct formulation not implemented yet.")


def _stiffness_form_indirect(
    u_tr: Argument, u_test: Argument, modulus: Coefficient, **kwargs
):
    # Return the stiffness form for untransformed indirect formulations

    raise NotImplementedError("Untransformed indirect formulation not implemented yet.")


def _stiffness_form_indirect_transformed(
    u_tr: Argument, u_test: Argument, modulus: Coefficient, wave_vec: Constant
):
    # Return the stiffness form for transformed indirect formulations

    if len(u_tr.ufl_shape) == 0:
        # Scalar trial function; create Helmholtz problem
        if ScalarType == np.complex128:
            return (
                (
                    inner(dot(modulus, grad(u_tr)), grad(u_test))
                    + inner(dot(wave_vec, dot(modulus, wave_vec)) * u_tr, u_test)
                    - 1j * inner(u_tr * dot(modulus, wave_vec), grad(u_test))
                    + 1j * inner(dot(modulus, grad(u_tr)), wave_vec * u_test)
                )
                * dx,
            )
        else:
            real_form = (
                inner(dot(modulus, grad(u_tr)), grad(u_test))
                + inner(u_tr * dot(wave_vec, modulus), wave_vec * u_test)
            ) * dx
            imag_form = (
                inner(dot(modulus, grad(u_tr)), wave_vec * u_test)
                - inner(u_tr * dot(modulus, wave_vec), grad(u_test))
            ) * dx
            return (real_form, imag_form)
    elif len(u_tr.ufl_shape) == 1:
        # Vector trial function; create elasticity problem
        raise NotImplementedError("Elasticity equations not implemented yet")
        # if ScalarType == np.complex128:
        #     return (inner(stress(modulus, u_tr), strain(u_test)) * dx,)
        # else:
        #     real_form = inner(stress(modulus, u_tr), strain(u_test)) * dx
        #     imag_form = 0
        #     return (real_form, imag_form)
    else:
        raise ValueError(f"Unsupported trial function shape: {u_tr.ufl_shape}")


def stiffness_form(
    problem_type: BlochProblemType,
    u_tr: Argument,
    u_test: Argument,
    modulus: Coefficient,
    **kwargs,
) -> typing.Union[tuple[Form], tuple[Form, Form]]:
    """Create the stiffness bilinear form for the Bloch wave problem.

    The problem type is detected from the shape of the trial function. (It is assumed
    that the test function will have the same shape, and will cause an error if it does
    not.) Scalar trial functions trigger a Helmholtz-type problem to be returned, and
    the modulus can be either a scalar or a second-order tensor in this case. Vector
    trial functions trigger an elasticity-type problem to be returned, and the modulus
    must be either a fourth-order tensor or a Voigt stiffness matrix.

    The PETSc scalar type determines the splitting scheme used. If the scalar type is
    complex, a single sesquilinear form is returned that incorporates real and imaginary
    parts. If the scalar type is real, the problem is assumed to be lossless and two
    bilinear forms are returned, one each for the real and imaginary parts of the
    stiffness matrix.

    Additional keyword arguments may be required, depending on `problem_type` (see
    below).

    Parameters
    ----------
    - `u_tr` trial function of the problem
    - `u_test` test function of the problem
    - `modulus` scalar or matrix function representing the material modulus

    Required keyword arguments
    --------------------------
    - `wave_vec: dolfinx.fem.Constant` wavevector (required for
        `problem_type == BlochProblemType.DIRECT` and
        `problem_type == BlochProblemType.INDIRECT_TRANSFORMED)

    Returns
    -------
    a tuple containing either a single sesquilinear form or two bilinear forms
    representing the mass matrix
    """

    # Check problem type and keyword args, then dispatch to the required implementation
    if problem_type == BlochProblemType.DIRECT:
        if "wave_vec" not in kwargs:
            raise RuntimeError("`wave_vec` is required for direct problems")
        else:
            return _stiffness_form_direct(u_tr, u_test, modulus, **kwargs)
    elif problem_type == BlochProblemType.INDIRECT:
        return _stiffness_form_indirect(u_tr, u_test, modulus)
    elif problem_type == BlochProblemType.INDIRECT_TRANSFORMED:
        if "wave_vec" not in kwargs:
            raise RuntimeError(
                "`wave_vec` is required for transformed indirect problems"
            )
        else:
            return _stiffness_form_indirect_transformed(
                u_tr, u_test, modulus, kwargs["wave_vec"]
            )
    else:
        raise ValueError(f"Unrecognized problem type: {problem_type}")
