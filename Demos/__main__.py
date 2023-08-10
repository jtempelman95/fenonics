#! python3
# -*- coding: utf-8 -*-
#
# __main__.py
#
# Demo code for solving, post-processing, and plotting band structures using fenonics.
#
# Created:       2023-08-02
# Last modified: 2023-08-02
#
# Copyright (c) 2023 Joshua R. Tempelman (jrt7@illinois.edu), Connor D. Pierce
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
#%%


import matplotlib.pyplot as plt
import numpy as np
from dolfinx.io.gmshio import model_to_mesh
import gmsh
from fenonics.FEM_Functions import *
from fenonics.PostProcess import *
from fenonics.MeshFunctions import get_mesh
from mpi4py import MPI

# Meshing parameters
cut = True
a_len = 0.1
r = np.array([1, 0.9, 0.3, 0.8, 0.6]) * a_len * 0.75
offset = 0 * np.pi / 4
design_vec = np.concatenate((r / a_len, [offset]))
Nquads = 5
da = a_len / 15
refinement_level = 4
refinement_dist = a_len / 10
meshalg = 6

# Make the mesh with Gmsh
gmsh.model, xpt, ypt = get_mesh(
    a_len,
    da,
    r,
    Nquads,
    offset,
    meshalg,
    refinement_level,
    refinement_dist,
    isrefined=True,
    cut=cut,
)

# Import to dolfinx
mesh_comm = MPI.COMM_WORLD
model_rank = 0
mesh, ct, _ = model_to_mesh(gmsh.model, mesh_comm, model_rank, gdim=2)

# Plot the design vector and the produced mesh
plt = PlotSpline(gmsh, r, Nquads, a_len, xpt, ypt)
plt.show()
plotmesh(mesh, ct)

# Define material properties
if not cut:
    c = [20, 60]  # if solid inclusion (mutlple materail model)
    rho = [2, 20]  # if solid inclusion (mutlple materail model)
else:
    c = [30]  # if void inclusion  (if iscut)
    rho = [1.2]  # if void inclusion  (if iscut)


# Define the high symmetry points of the lattice
G = np.array([0, 0])
X = np.array([np.pi / a_len, 0])
M = np.array([np.pi / a_len, np.pi / a_len])
Y = np.array([0, np.pi / a_len])
Mp = np.array([-np.pi / a_len, np.pi / a_len])
Xp = np.array([-np.pi / a_len, 0])
HSpts = [G, X, M, G, Y, M, G, Xp, Mp, G, Y, Mp, G]

# Define the number of solutiosn per wavevec and number of wavevecs to solve for
n_solutions = 30
n_wavevector = len(HSpts) * 10

# Solve the dispersion problem
evals_disp, evec_all, mpc, KX, KY = solve_bands(
    HSpts=HSpts,
    n_wavevector=n_wavevector,
    n_solutions=n_solutions,
    a_len=a_len,
    c=c,
    rho=rho,
    fspace="CG",
    mesh=mesh,
    ct=ct,
)

bgnrm, gapwidths, gaps, lowbounds, highbounds = getbands(np.array(evals_disp))
HS_labels = [
    r"$\Gamma$",
    "X",
    "M",
    r"$\Gamma$",
    "Y",
    "M",
    r"$\Gamma$",
    "X*",
    "M*",
    r"$\Gamma$",
    "Y*",
    "M*",
    r"$\Gamma$",
]

plt = plotbands(
    np.array(evals_disp),
    figsize=(5, 5),
    HSpts=HSpts,
    HS_labels=HS_labels,
    a_len=a_len,
    KX=KX,
    KY=KY,
    inset=False,
)
plt.show()
