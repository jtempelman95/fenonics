#! python3
# -*- coding: utf-8 -*-
#
# MeshFunctions.py
#
# Mesh generation with Gmsh.
#
# Created:       2023-03-10
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
Generate phononic unit cell meshes with Gmsh.

Functions
---------
get_mesh_SquareSpline
GetSpline
"""


__all__ = ["get_mesh", "GetSpline"]


# Imports

import gmsh
import numpy as np


# Functions


def get_mesh(
    a: float = 1.0,
    da: float = None,
    r: np.array = None,
    Nquads: float = 8,
    offset: float = 0,
    meshalg: int = 6,
    refinement_level: float = 5,
    refinement_dist: float = None,
    isrefined: bool = True,
    cut: bool = True,
    symmetry="rotated",
):
    """Generating a mesh with an internal void or inclusion

    arguments
        isrefined           -   decide to refine around interior edges
        cut                 -   Decide of interor geometry is cut-out or inclusion
        Nquad               - number of quadrants to repeat geometry about
        a                   - unit cell lenght
        da                  - nominal mesh spacing
        r                   - vector of radii
        offset              - ofset angle
        meshalg             - wich algorithm to use in gmsh
        refinement_level    - how much denser refined geomtety is

    outputs
        gsmh object
    """

    if da is None:
        da = a / 15
    if refinement_dist is None:
        refinement_dist = a / 8

    gmsh.initialize()
    gmsh.clear()
    # gmsh.option.setNumber("Geometry.Tolerance", 1e-6/4.48401)
    # a - > Charecteristic Length of the unit cell
    # r - > vector of inputs to parameterize the mesh (spline)
    # Nquads - > Number of quadrants to rotate about

    p1 = gmsh.model.occ.addPoint(0, 0, 0, tag=1)
    p2 = gmsh.model.occ.addPoint(a, 0, 0, tag=2)
    p3 = gmsh.model.occ.addPoint(a, a, 0, tag=3)
    p4 = gmsh.model.occ.addPoint(0, a, 0, tag=4)

    # Adding curves defining outter perimeter
    l1 = gmsh.model.occ.addLine(1, 2, tag=1)  # Bottom wall
    l2 = gmsh.model.occ.addLine(2, 3, tag=2)  # Right wall
    l3 = gmsh.model.occ.addLine(3, 4, tag=3)  # Top wall
    l4 = gmsh.model.occ.addLine(4, 1, tag=4)  # Left wall
    gmsh.model.occ.synchronize()

    # Create the outter box
    cell_boundary = gmsh.model.occ.addCurveLoop([l1, l2, l3, l4], tag=1)
    gmsh.model.occ.synchronize()
    # Create a surface from the outter box
    cell_surface = gmsh.model.occ.addPlaneSurface([cell_boundary], tag=1)
    surf = [(2, cell_surface)]
    gmsh.model.occ.synchronize()

    # Add the spline curve
    xspl = []
    yspl = []
    idx = 0
    if Nquads % 2 == 0:
        if symmetry == "mirror":
            shift = 2 * np.pi / len(r) / Nquads / 2
        else:
            shift = 0 * np.pi / len(r) / Nquads / 8
    else:
        shift = 0
    for quad in range(Nquads):
        for k in range(len(r)):
            xc = a / 2 + r[k] / 2 * np.cos(
                2 * k * np.pi / len(r) / Nquads
                + 2 * quad * np.pi / Nquads
                + shift
                + offset
            )
            yc = a / 2 + r[k] / 2 * np.sin(
                2 * k * np.pi / len(r) / Nquads
                + 2 * quad * np.pi / Nquads
                + shift
                + offset
            )
            yspl.append(yc)
            xspl.append(xc)
            idx += 1
            # print(2*k*np.pi/len(r)/Nquads + 2*quad*np.pi/Nquads+shift)
        if Nquads % 2 == 0:
            if symmetry == "mirror":
                r = np.flip(r)

    splinepoints = [
        gmsh.model.occ.addPoint(xspl[v], yspl[v], 0) for v in range(len(xspl))
    ]
    splinepoints.append(splinepoints[0])
    spline1 = gmsh.model.occ.add_bspline(splinepoints, tag=125)

    # Create a surface from the spline
    spline_boundary = gmsh.model.occ.addCurveLoop([spline1], tag=100)
    spline_surf = gmsh.model.occ.addPlaneSurface([spline_boundary], tag=100)
    gmsh.model.occ.synchronize()

    if cut:
        # Fuse the spline surface to the unit cell
        all_surfaces = [(2, spline_surf)]
        out, whole_domain = gmsh.model.occ.cut([(2, cell_surface)], all_surfaces)
        gmsh.model.occ.synchronize()

        other_surfaces = []
        tag = 1
        for domain in whole_domain[0]:
            com = gmsh.model.occ.getCenterOfMass(domain[0], domain[1])
            mass = gmsh.model.occ.getMass(domain[0], domain[1])
            disk1 = gmsh.model.addPhysicalGroup(domain[0], [domain[1]], tag=tag)
            tag += 1
        other_surfaces.append(whole_domain[0][0])

    else:
        # Fuse the spline surface to the unit cell
        all_surfaces = [(2, spline_surf)]
        out, whole_domain = gmsh.model.occ.fragment([(2, cell_surface)], all_surfaces)
        gmsh.model.occ.synchronize()

        other_surfaces = []
        tag = 1
        for domain in whole_domain[0]:
            com = gmsh.model.occ.getCenterOfMass(domain[0], domain[1])
            mass = gmsh.model.occ.getMass(domain[0], domain[1])
            disk1 = gmsh.model.addPhysicalGroup(domain[0], [domain[1]], tag=tag)
            tag += 1
        other_surfaces.append(whole_domain[1][0])

    # set mesh size
    # First remove anythng not in bounding box
    eps = 1e-3
    gmsh.option.setNumber("Geometry.OCCBoundsUseStl", 1)
    vin = gmsh.model.getEntitiesInBoundingBox(-eps, -eps, 0, a + eps, a + eps, 0, 2)

    p = gmsh.model.getBoundary(vin, False, False, True)  # Get all points
    gmsh.model.mesh.setSize(p, da)
    p = gmsh.model.getEntitiesInBoundingBox(2 - eps, -eps, -eps, 2 + eps, eps, eps, 1)
    gmsh.model.mesh.setSize(p, da)

    # Mesh refnement around the inclusion
    if isrefined is True:
        if cut is True:
            gmsh.model.mesh.field.add("Distance", 1)
            edges = gmsh.model.getBoundary(other_surfaces, oriented=True)
            gmsh.model.mesh.field.setNumbers(1, "CurvesList", [5])
            gmsh.model.mesh.field.setNumber(1, "Sampling", 500)
            gmsh.model.mesh.field.add("Threshold", 2)
            gmsh.model.mesh.field.setNumber(2, "IField", 1)
            gmsh.model.mesh.field.setNumber(2, "LcMin", da / refinement_level)
            gmsh.model.mesh.field.setNumber(2, "LcMax", da * 1)
            gmsh.model.mesh.field.setNumber(2, "DistMin", 0.025 * a)
            gmsh.model.mesh.field.setNumber(2, "DistMax", refinement_dist)
            gmsh.model.mesh.field.add("Min", 5)
            gmsh.model.mesh.field.setNumbers(5, "FieldsList", [2])
            gmsh.model.mesh.field.setAsBackgroundMesh(5)
        else:
            gmsh.model.mesh.field.add("Distance", 1)
            edges = gmsh.model.getBoundary(other_surfaces, oriented=True)
            gmsh.model.mesh.field.setNumbers(1, "EdgesList", [e[1] for e in edges])
            gmsh.model.mesh.field.setNumber(1, "Sampling", 500)
            gmsh.model.mesh.field.add("Threshold", 2)
            gmsh.model.mesh.field.setNumber(2, "IField", 1)
            gmsh.model.mesh.field.setNumber(2, "LcMin", da / refinement_level)
            gmsh.model.mesh.field.setNumber(2, "LcMax", da * 1)
            gmsh.model.mesh.field.setNumber(2, "DistMin", 0.025 * a)
            gmsh.model.mesh.field.setNumber(2, "DistMax", refinement_dist)
            gmsh.model.mesh.field.add("Min", 5)
            gmsh.model.mesh.field.setNumbers(5, "FieldsList", [2])
            gmsh.model.mesh.field.setAsBackgroundMesh(5)

    # Add in periodic constraint
    print(gmsh.model.getEntities(dim=1))
    left = []
    right = []
    kj = 0
    for line in gmsh.model.getEntities(dim=1):
        com = gmsh.model.occ.getCenterOfMass(line[0], line[1])

        # Add the exterior and interior edges as a physical groups
        gmsh.model.addPhysicalGroup(1, [line[1]], line[1])

        if np.isclose(com[0], 0):
            left.append(line[1])

        if np.isclose(com[0], a):
            right.append(line[1])

    # gmsh.model.mesh.setSize([(1,5)], da/10)

    # print('found left lines is:'+str(left))
    # print('defined left lines is:'+str(left))
    # print(" --------- ERROR HERE --------------")
    # # wall_ents = gmsh.model.getEntitiesForPhysicalGroup(2, walls)
    # translation = [1, 0, 0, a, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]
    # translationy = [1, 0, 0, 0, 0, 1, 0, a, 0, 0, 1, 0, 0, 0, 0, 1]
    # # gmsh.model.mesh.setPeriodic(1, left, right, translation)
    # gmsh.model.mesh.setPeriodic(1, [l4], [l2], translation)
    # gmsh.model.mesh.setPeriodic(1, [l1], [l3], translationy)
    # gmsh.model.occ.synchronize()
    # print(" --------- ERROR HERE --------------")

    mesh_dim = 2
    gmsh.option.setNumber("Mesh.Algorithm", meshalg)
    gmsh.model.mesh.generate(mesh_dim)

    print("to here")
    return gmsh.model, xspl, yspl


def GetSpline(gmsh):
    """
    # Get spline geometry from gmsh mesh
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
    xptnd = np.array(xpt)
    yptnd = np.array(ypt)
    lxp = len(xptnd)
    xsv = np.empty(xi.shape)
    SplineDat = np.hstack((xi.reshape(lxi, 1), yi.reshape(lxi, 1)))
    SplinePtDat = np.hstack((xptnd.reshape(lxp, 1), yptnd.reshape(lxp, 1)))
    return SplinePtDat, SplineDat


# %% Test
"""
Test the functions out here
"""

if __name__ == "__main__":
    # ################################################## #
    # Testing the mesh functions                #
    # ################################################## #
    from mpi4py import MPI
    import matplotlib.pyplot as plt
    import pyvista

    pyvista.start_xvfb()
    import gmsh
    from dolfinx.fem import FunctionSpace, Function
    from dolfinx.plot import create_vtk_mesh
    from dolfinx.io import XDMFFile
    from dolfinx.io.gmshio import model_to_mesh

    # Inputs to meshing program
    a_len = 0.1
    offset = 0 * np.pi
    Nquads = 8
    da = a_len / 25
    meshalg = 6
    refinement_level = 6
    refinement_dist = a_len / 10
    meshalg = 6

    r = np.array([1, 0.2, 0.9, 0.2]) * a_len * 0.95
    gmsh.model, xpt, ypt = get_mesh_SquareSpline(
        a=a_len, da=da, r=r, Nquads=Nquads, refinement_level=3, symmetry="rotated"
    )

    #################################################################
    #            Import to dolfinx and save as xdmf                 #
    #################################################################
    mesh_comm = MPI.COMM_WORLD
    model_rank = 0
    mesh, ct, _ = model_to_mesh(gmsh.model, mesh_comm, model_rank, gdim=2)
    with XDMFFile(MPI.COMM_WORLD, "mt.xdmf", "w") as xdmf:
        xdmf.write_mesh(mesh)
        xdmf.write_meshtags(ct)

    #################################################################
    #              Plot the mesh                                    #
    #################################################################
    V = FunctionSpace(mesh, ("CG", 1))
    v = Function(V)
    plotter = pyvista.Plotter()
    grid = pyvista.UnstructuredGrid(*create_vtk_mesh(mesh, mesh.topology.dim))
    num_local_cells = mesh.topology.index_map(mesh.topology.dim).size_local
    grid.cell_data["Marker"] = 1 - ct.values[ct.indices < num_local_cells]
    grid.set_active_scalars("Marker")
    actor = plotter.add_mesh(
        grid, show_edges=True, line_width=3, edge_color="k", style="wireframe"
    )
    plotter.set_background("white", top="white")
    plotter.view_xy()
    plotter.camera.tight(padding=0.1)
    plotter.add_title(
        "CELLS:" + str(ct.values.shape[0]) + "  | NODES:" + str(v.vector[:].shape[0]),
        color="r",
    )
    plotter.show()

    # ###########################################################
    # Recover the spline curve used to make the mesh
    # ###########################################################
    node_interior = gmsh.model.mesh.getNodesForPhysicalGroup(1, 5)[1]
    x_int = node_interior[0::3]
    y_int = node_interior[1::3]
    x_int = np.concatenate([x_int, [x_int[0]]])
    y_int = np.concatenate([y_int, [y_int[0]]])
    xi = x_int - a_len / 2
    yi = y_int - a_len / 2
    plt.plot(np.array(xpt) - a_len / 2, np.array(ypt) - a_len / 2, ".")
    plt.plot(xi, yi, color="r")
    plt.grid()
    plt.axis("image")
    plt.show()

    # %%
    """
    # Testing sensitivity to mesh options
    """

    np.random.seed(24)
    r = np.array([1, np.random.rand(1)[0], 0.3]) * a_len * 0.95
    r = np.random.rand(4) * a_len * 0.95

    gmsh.model, xpt, ypt = get_mesh_SquareSpline(
        a_len,
        a_len / 100,
        r,
        Nquads,
        offset,
        meshalg,
        refinement_level,
        refinement_dist,
        isrefined=True,
        cut=True,
    )
    # ###########################################################
    # # Get spline fit from the mesh
    # ###########################################################
    node_interior = gmsh.model.mesh.getNodesForPhysicalGroup(1, 5)[1]
    x_int = node_interior[0::3]
    y_int = node_interior[1::3]
    x_int = np.concatenate([x_int, [x_int[0]]])
    y_int = np.concatenate([y_int, [y_int[0]]])
    xi = x_int - a_len / 2
    yi = y_int - a_len / 2
    Xi = xi
    Yi = yi
    # plt.plot(np.array(xpt)- a_len/2, np.array(ypt)- a_len/2,'.')
    Nquads = 16

    plt.figure(figsize=(10, 10))
    plt.plot(
        xi,
        yi,
        color="k",
        linewidth=2.5,
        label=str(gmsh.model.mesh.getNodes()[0].shape[0]),
    )

    da = a_len / 15
    meshalg = 6
    refinement_level = 4
    refinement_dist = a_len / 25
    meshalg = 6
    for refinement_level in np.linspace(2, 6, 3):
        gmsh.model, xpt, ypt = get_mesh_SquareSpline(
            a_len,
            da,
            r,
            Nquads,
            offset,
            meshalg,
            refinement_level,
            refinement_dist,
            isrefined=True,
            cut=True,
        )
        # ###########################################################
        # # Get spline fit from the mesh
        # ###########################################################
        node_interior = gmsh.model.mesh.getNodesForPhysicalGroup(1, 5)[1]
        x_int = node_interior[0::3]
        y_int = node_interior[1::3]
        x_int = np.concatenate([x_int, [x_int[0]]])
        y_int = np.concatenate([y_int, [y_int[0]]])
        # plt.plot(x_int,y_int,'-')
        xi = x_int - a_len / 2
        yi = y_int - a_len / 2

        # plt.plot(np.array(xpt)- a_len/2, np.array(ypt)- a_len/2,'.')
        plt.plot(
            xi,
            yi,
            "-",
            linewidth=1.5,
            label=str(gmsh.model.mesh.getNodes()[0].shape[0]),
        )
    plt.grid()
    plt.legend()

    ################################################################
    #        Import to dolfinx and save as xdmf                 #
    ################################################################
    mesh_comm = MPI.COMM_WORLD
    model_rank = 0
    mesh, ct, _ = model_to_mesh(gmsh.model, mesh_comm, model_rank, gdim=2)
    with XDMFFile(MPI.COMM_WORLD, "mt.xdmf", "w") as xdmf:
        xdmf.write_mesh(mesh)
        xdmf.write_meshtags(ct)

    #################################################################
    #              Plot the mesh                                    #
    #################################################################
    V = FunctionSpace(mesh, (fspace, 1))
    v = Function(V)
    plotter = pyvista.Plotter()
    grid = pyvista.UnstructuredGrid(*create_vtk_mesh(mesh, mesh.topology.dim))
    num_local_cells = mesh.topology.index_map(mesh.topology.dim).size_local
    grid.cell_data["Marker"] = 1 - ct.values[ct.indices < num_local_cells]
    grid.set_active_scalars("Marker")
    actor = plotter.add_mesh(
        grid, show_edges=True, line_width=3, edge_color="k", style="wireframe"
    )
    plotter.set_background("white", top="white")
    plotter.view_xy()
    plotter.camera.tight(padding=0.1)
    plotter.add_title(
        "CELLS:" + str(ct.values.shape[0]) + "  | NODES:" + str(v.vector[:].shape[0]),
        color="r",
    )
    plotter.show()
