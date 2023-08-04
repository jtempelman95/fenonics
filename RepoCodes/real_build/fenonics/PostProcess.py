#! python3
# -*- coding: utf-8 -*-
#
# PostProcess.py
#
# Plot band structures and identify band gaps.
#
# Created:       2023-07-24
# Last modified: 2023-07-29
#
# Copyright (c) 2023 Joshua R. Tempelman (jrt7@illinois.edu)
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
Module `PostProcess`

Functions
---------
getbands
GetSpline
plotbands
PlotSpline
plot_splines
plotmesh
plotvecs
"""


__all__ = [
    "getbands",
    "plotbands",
    "plotmesh",
    "plot_splines",
    "plotvecs",
    "GetSpline",
    "PlotSpline",
]


# Imports

import numpy as np
import matplotlib.pyplot as plt
import dolfinx
import dolfinx_mpc
import pyvista
from dolfinx.plot import create_vtk_mesh

pyvista.start_xvfb()
from matplotlib.patches import Rectangle
from dolfinx import plot

#################################################
#             DATA PROCESSING                   #
#################################################


def getbands(bands):
    """Getting the band-gaps from a dispersion diagram

    input:
        bands - an np array storing eigensolutions across wavevectors

    output:
        BG_normalized - normalized band gap widthds
        gapwidths   - spacing between bands (not necessarly band gap)
        gaps        - un-normalized band gap widths
                      (filtered to ensure the gap is not a
                      numerical artificat)
        lowbounds   - frequencies of the lower bounds of band gaps
        highbounds  - frequencies of the higher bounds of band gaps
    """
    nvec = bands.shape[1]
    nK = bands.shape[0]
    ef_vec = bands.reshape((1, nvec * nK))
    evals_all = np.sort(ef_vec).T
    deval = np.diff(evals_all.T).T
    args = np.flip(np.argsort(deval.T)).T

    # Finding the boundaries of the pass bands
    lowb, uppb, lowlim, uplim, bgs = [], [], [], [], []
    for k in range(nvec):
        lowb.append(np.min(bands.T[k]))
        uppb.append(np.max(bands.T[k]))
    for k in range(nvec):
        LowerLim = np.max(bands[:, k])
        if k < nvec - 1:
            UpperLim = np.min(bands[:, k + 1])
        else:
            UpperLim = np.min(bands[:, k])

        # Check if these limits fall in a pass band
        overlap = False
        for j in range(nvec):
            if LowerLim > lowb[j] and LowerLim < uppb[j]:
                overlap = True
            if UpperLim > lowb[j] and UpperLim < uppb[j]:
                overlap = True
        if overlap == False:
            lowlim.append(LowerLim)
            uplim.append(UpperLim)
            bgs.append(UpperLim - LowerLim)

    # Filter band gaps
    maxfq = np.max(bands[:])
    isgap = [i for i, v in enumerate(bgs) if v > np.median(deval)]
    gaps = np.array(bgs)
    lower = np.array(lowlim)
    higher = np.array(uplim)
    gapwidths = gaps[isgap]
    lowbounds = lower[isgap]
    highbounds = higher[isgap]
    BG_normalized = gapwidths / (0.5 * lowbounds + 0.5 * highbounds)

    return BG_normalized, gapwidths, gaps, lowbounds, highbounds


def GetSpline(gmsh, a_len, xpt, ypt):
    """
    Get spline geometry from gmsh mesh

    inputs:
        gmsh - the gmsh object
        a_len - charecterstic length of unit cell

    outputs:
        SpintPtDat - x-y points used for gmsh to generate spline curve
        SplineDat - x-y array of coordinates defining the spline curve
    """
    node_interior = gmsh.model.mesh.getNodesForPhysicalGroup(1, 5)[1]
    x_int = node_interior[0::3]
    y_int = node_interior[1::3]
    x_int = np.concatenate([x_int, [x_int[0]]])
    y_int = np.concatenate([y_int, [y_int[0]]])
    xi = x_int - a_len / 2
    yi = y_int - a_len / 2
    lxi = len(xi)
    lxp = len(xpt)
    xptnd = np.array(xpt) - a_len / 2
    yptnd = np.array(ypt) - a_len / 2
    lxp = len(xptnd)
    xsv = np.empty(xi.shape)
    SplineDat = np.hstack((xi.reshape(lxi, 1), yi.reshape(lxi, 1)))
    SplinePtDat = np.hstack((xptnd.reshape(lxp, 1), yptnd.reshape(lxp, 1)))
    return SplinePtDat, SplineDat


#################################################
#             VISUALIZATION                     #
#################################################


def plotbands(
    bands: np.array = None,
    HSpts: list = None,
    HS_labels: list = None,
    inset: bool = False,
    KX: list = None,
    KY: list = None,
    a_len: float = 1.0,
    figsize: tuple = (5, 4),
):
    """Plotting the dispersion bands

    Paramters
    ---------
        bands - A numpy array containing all eigensolutions at each wavenumber
        HSpts - A list of the numeric values of high-symmetry points solvded fro
        HS_labels - The labels of the high-symmetry points
        inset -  whetehr or not to plot the path along the IBZ in the inset (based on KX, KY)
        KX - The kx-wavevectors visited in the disperison computaiton (a list of points)
        KY - The ky-wavevectors visited in the disperison computaiton (a list of points)
        figsize - A tuple for the figure size. (5,4) is default

    output
    --------
        plt - The matplotlib.pyplot object
    """

    bgnrm, gapwidths, gaps, lowbounds, highbounds = getbands(bands)

    plt.figure(figsize=figsize)

    # Default assumes a G-X-M-G boundary
    if HS_labels == None:
        HS_labels = ["Γ", "X", "M", "Γ"]

    if HSpts == None:
        # Setting the high symmetr points
        P1 = [0, 0]  # Gamma
        P2 = [np.pi / a_len, 0]  # X
        P3 = [np.pi / a_len, np.pi / a_len]  # K
        HSpts = [P1, P2, P3, P1]
        HSstr = ["Γ", "X", "M", "Γ"]

    # Get number of solutions inbetween HS points
    nsol = bands.shape[0]
    nvec = bands.shape[1]
    nvec_per_HS = int(round(nsol / len(HS_labels) + 1))
    xx = np.linspace(0, 1, nsol)

    # PLOT THE DISPERSION BANDS
    for n in range(nvec):
        if n == 0:
            plt.plot(xx, (bands[:, n]), "b.-", markersize=0, label="Bands")
        if n > 0:
            plt.plot(xx, (bands[:, n]), "b.-", markersize=0)

    # Plot the band gaps
    currentAxis = plt.gca()
    for j in range(len(gapwidths)):
        lb = lowbounds[j]
        ub = highbounds[j]
        if j == 0:
            currentAxis.add_patch(
                Rectangle(
                    (np.min(xx), lb),
                    np.max(xx),
                    ub - lb,
                    facecolor="g",
                    ec="none",
                    alpha=0.3,
                    label="bangap",
                )
            )
        else:
            currentAxis.add_patch(
                Rectangle(
                    (np.min(xx), lb),
                    np.max(xx),
                    ub - lb,
                    facecolor="g",
                    ec="none",
                    alpha=0.3,
                )
            )

    # Plot formatting
    plt.grid(color="gray", linestyle="-", linewidth=0.3)
    plt.xticks(np.linspace(0, xx.max(), len(HS_labels)), HS_labels, fontsize=18)
    plt.xlabel(r"Wave Vector ", fontsize=18)
    plt.ylabel("$\omega$ [rad/s]", fontsize=18)
    plt.title("Dispersion Diagram", fontsize=18)
    plt.xlim((0, xx.max()))
    plt.ylim((0, bands.max()))
    plt.legend()

    # Inset for the IBZ
    if inset:
        ax = plt.gca()
        ax2 = ax.inset_axes([0.6, 0.6, 0.4, 0.4])
        ax2.plot(np.array(KX) * a_len / np.pi, np.array(KY) * a_len / np.pi)
        ax2.scatter(
            np.array(HSpts)[:, 0] * a_len / np.pi, np.array(HSpts)[:, 1] * a_len / np.pi
        )
        ax2.text(0.05, 0.05, "$\Gamma$", fontsize=15)
        ax2.text(1.05, 0.05, "X", fontsize=15)
        ax2.text(1.05, 1.05, "M", fontsize=15)
        ax2.text(0.05, 1.05, "Y", fontsize=15)
        ax2.text(-1.05, 0.05, "X*", fontsize=15)
        ax2.text(-1.05, 1.05, "M*", fontsize=15)
        ax2.add_patch(Rectangle((-1, -1), 2, 2, facecolor="gray", ec="none", alpha=0.3))
        ax2.set_xticks([])
        ax2.set_yticks([])
        ax2.axis("square")
        ax2.set_alpha(0.2)
        ax2.set_xlim((-1.5, 1.5))
        ax2.set_ylim((-1.5, 1.5))

    return plt


def PlotSpline(gmsh, r, Nquads, a_len, xpt, ypt):
    '''Plotting the spline curve from the mesh
    
    parameters
    ----------
    gmsh - the gmsh object containg gmsh.mesh
    r  - the design vector fed into get_mesh()
    a_len - chareceterstic unit cell length
    xpt - the x coordinates of the design vector
    ypt - the y coordinates of the design vector
    
    output
    --------
    matplotlib.pyplot object containg spline plot
    '''
    SplinePtDat, SplineDat = GetSpline(gmsh, a_len, xpt, ypt)
    x = SplinePtDat[:, 0]
    y = SplinePtDat[:, 1]
    plt.plot(SplineDat[:, 0], SplineDat[:, 1])
    plt.plot(x[0 : int(len(x) / Nquads)], y[0 : int(len(x) / Nquads)], " .r")
    plt.plot(
        x[int(len(r) / Nquads) : -1], y[int(len(r) / Nquads) : -1], ".", color="gray"
    )
    plt.plot(x[0 : int(len(x) / Nquads)], y[0 : int(len(x) / Nquads)], " .r")
    for j in range(len(SplinePtDat)):
        if j < int(len(x) / Nquads):
            plt.plot([0, x[j]], [0, y[j]], "k")
        else:
            plt.plot([0, x[j]], [0, y[j]], "--", color="gray", linewidth=1)
    ax = plt.gca()
    ax.add_patch(
        Rectangle(
            (-a_len / 2, -a_len / 2),
            a_len,
            a_len,
            facecolor="w",
            ec="r",
            alpha=0.25,
            label="bangap",
        )
    )
    ax.add_patch(
        Rectangle(
            (-a_len / 2, -a_len / 2),
            a_len,
            a_len,
            facecolor="none",
            ec="r",
            alpha=1,
            label="bangap",
        )
    )
    plt.tick_params(
        direction="in",
        length=6,
        width=0.5,
        colors="k",
        grid_color="k",
        grid_alpha=0.5,
        bottom=True,
        top=True,
        left=True,
        right=True,
    )
    ax.grid(color="gray", linestyle="dashed")
    plt.legend(
        ["Bspine", "Design Vector", "Rotated Qudrants"],
        loc="lower right",
        bbox_to_anchor=(0.5, 0.0),
    )
    plt.axis("square")
    return plt


def plot_splines(splines, targets, nrow=6, ncol=6, is_rand=True):
    """
    Function for plotting the spline curves
    """
    # cm      = plt.get_cmap('twilight_shifted', 500)   # 11 discrete colors
    arg = np.argsort(targets)
    cdata = targets
    cdata = pre.MinMaxScaler().fit_transform(cdata.reshape(-1, 1))
    plt.figure(figsize=(5, 5))
    plt.subplots_adjust(
        left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.05, hspace=0.02
    )
    for idx in range(nrow * ncol):
        ax = plt.subplot(nrow, ncol, idx + 1)
        if is_rand:
            idx = int(np.random.rand() * len(features))
        maxarg = arg[-idx]
        ax.plot(
            splines[maxarg][:, 0],
            splines[maxarg][:, 1],
            c=cm.twilight_shifted(cdata[maxarg]),
        )
        ax.axis("equal")
        plt.axis("image")
        ax.set_xlim((-0.05, 0.05))
        ax.set_ylim((-0.05, 0.05))
        ax.set_xticks([])
        ax.set_yticks([])
    return plt


def plotmesh(mesh, ct):
    """ Plotting the mesh written by Gmsh

    parameters
    ---------
    mesh  - a gmsh.mesh object containting mesh info
    ct    - cell tegas
    
    output
    -------
    plotter - a pyvista.plotter object containing the mesh plot
    """

    V = dolfinx.fem.FunctionSpace(mesh, ("CG", 1))
    v = dolfinx.fem.Function(V)
    plotter = pyvista.Plotter()
    grid = pyvista.UnstructuredGrid(*create_vtk_mesh(mesh, mesh.topology.dim))
    num_local_cells = mesh.topology.index_map(mesh.topology.dim).size_local
    grid.cell_data["Marker"] = 1 - ct.values[ct.indices < num_local_cells]
    grid.set_active_scalars("Marker")
    actor = plotter.add_mesh(grid, show_edges=True, line_width=1, edge_color="w")
    # actor = plotter.add_mesh(grid, style="wireframe", color= 'w', line_width=3)
    plotter.set_background("black", top="black")
    plotter.view_xy()
    plotter.screenshot("Mesh.jpeg", window_size=[2000, 1400])
    plotter.add_title(
        "CELLS:" + str(ct.values.shape[0]) + "  | NODES:" + str(v.vector[:].shape[0])
    )
    plotter.show()
    return plotter


def plotvecs(
    plotter: pyvista.Plotter = None,
    V: dolfinx.fem.FunctionSpace = None,
    mpc: dolfinx_mpc.MultiPointConstraint = None,
    evecs: np.array = None,
    eval_number: int = 1,
    wavevec_number: int = 1,
    cmap: str = "bwr",
    nmap: int = 50,
):
    """Plotting the eigenvectors of the dispersion problem

    inputs:
            plotter: Pyvista plotter object (pv.plotter)
            V: Function space (defined over mesh using dolfinx.fem.FunctoinSpace())
            mpc: Multi-point constraint for periodic BVP
            evecs: An numpy of dim [i x j x k]
                    i Number of wavevectors solved for
                    j Number of Dofs in unit cell
                    k Number of evals solved per wavenumber
            eval_number: The eigenvalue number to plot (i)
            wavevec_number: The wavevector index to plot (k)

    outputs:
            plotter:        The plotter object is returned with the eigenvector visual
                            stored on the object. Use plotter.show() after calling function
                            to view.

    """

    # Post-processing the -vecs
    data = np.array(evecs)
    et = data[wavevec_number, :, eval_number]
    vr = dolfinx.fem.Function(V)
    vi = dolfinx.fem.Function(V)
    vr.vector[:] = np.real(et)  # / np.max( np.real(et))
    vi.vector[:] = np.imag(et)  # / np.max( np.real(et))
    vr.x.scatter_forward()
    mpc.backsubstitution(vr.vector)
    vi.x.scatter_forward()
    mpc.backsubstitution(vi.vector)

    # Plotting eigenvectors with pyvista
    mycmap = plt.cm.get_cmap("cmap", nmap)
    u = dolfinx.fem.Function(V)
    cells, types, x = plot.create_vtk_mesh(V)
    grid = pyvista.UnstructuredGrid(cells, types, x)
    u.vector.setArray(vr.vector[:])
    grid.point_data["u"] = u.x.array
    edges = grid.extract_all_edges()
    warped = grid.warp_by_scalar("u", factor=0)
    plotter.add_mesh(
        warped, show_edges=False, show_scalar_bar=True, scalars="u", cmap=mycmap
    )
    plotter.view_xy()
    plotter.camera.tight(padding=0.1)

    return plotter
