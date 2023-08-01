#! python3
# -*- coding: utf-8 -*-
#
# solver.py
#
# Eigensolvers for band structure computations.
#
# Created:       2023-07-31 14:37:57
# Last modified: 2023-07-31 14:41:24
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


__all__ = ["get_EPS"]


# Imports

from mpi4py import MPI
from petsc4py import PETSc
from slepc4py import SLEPc


# Functions


def get_EPS(
    stiffness_matrix: PETSc.Mat,
    mass_matrix: PETSc.Mat,
    n_solutions: int,
    target: float = 0,
):
    """Create and configure SLEPc eigensolver for band structure computations."""

    EPS = SLEPc.EPS()
    EPS.create(comm=MPI.COMM_WORLD)
    EPS.setOperators(stiffness_matrix, mass_matrix)
    EPS.setProblemType(SLEPc.EPS.ProblemType.GHEP)

    EPS.setDimensions(nev=n_solutions)

    EPS.setType(SLEPc.EPS.Type.KRYLOVSCHUR)

    EPS.setWhichEigenpairs(SLEPc.EPS.Which.TARGET_MAGNITUDE)
    EPS.setTarget(target)  # sorting

    EPS.setTolerances(tol=1e-5, max_it=12)

    ST = EPS.getST()
    ST.setType(SLEPc.ST.Type.SINVERT)
    ST.setShift(10)

    KSP = ST.getKSP()
    KSP.setType(KSP.Type.NONE)
    PC = KSP.getPC()
    PC.setType(PETSc.PC.Type.LU)
    PC.setFactorSolverType(PETSc.Mat.SolverType.MUMPS)
    KSP.setPC(PC)
    ST.setKSP(KSP)
    EPS.setST(ST)

    # parse command line options
    EPS.setFromOptions()
    return EPS
