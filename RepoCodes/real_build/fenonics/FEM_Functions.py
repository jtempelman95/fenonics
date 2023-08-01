#! python3
# -*- coding: utf-8 -*-
#
# FEM_Functions.py
#
# Dispersion computation with FENICSx.
#
# Created:       2023-03-10
# Last modified: 2023-07-29
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

"""

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

                DISPERSION COMPUTATION WITH FENICSX

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


    Josh Tempelman
    University of Illinois
    jrt7@illinois.edu

    Originated      MARCH 10, 2023
    Last Modified   MARCH 24, 2023

    ! ! Do not distribute ! !


    About:
        This program formulates a square unit cell mesh with an embedded
        geometry. The dolfinx-fenicsx routine and dolfinx-mpc solve the
        resulting eigenvalue problems over the Γ-X-M-Γ boundaries of the
        IBZ. Post processing identifies band gaps


    Requirements:

        Must have installed
            -python         (v 3.10.6)
            -dolfinx        (v 0.5.1)
            -dolfinx-mpc    (v 0.5.0)
            -gmsh           (v 4.11.0)
            -pyvista        (v 0.3)
            -slepc          (v 3.17.2)
            -petsc          (v 3.17.4)

        Must have on path
            -MeshFunctions.py

        Must operate in linux OS

"""


# General
import numpy as np
import time

# Finite element modeling
import dolfinx
import dolfinx_mpc
from dolfinx import fem
from dolfinx.fem import form, Function, FunctionSpace
from dolfinx.mesh import locate_entities_boundary
from fenonics import problem
from fenonics.problem import BlochProblemType
from mpi4py import MPI
from petsc4py import PETSc
from petsc4py.PETSc import ScalarType
from scipy import sparse
from scipy.sparse.linalg import eigsh
from slepc4py import SLEPc
from ufl import (
    dot,
    dx,
    grad,
    inner,
    Argument,
    TestFunction,
    TrialFunction,
)


def dirichlet_and_periodic_bcs(
    domain: dolfinx.mesh.Mesh,
    functionspace: fem.FunctionSpace,
    bc_type: list[str] = ["periodic", "periodic"],
    dbc_value: ScalarType = 0,
):
    """Create periodic and/or Dirichlet boundary conditions for a square domain.

    Parameters
    ----------
    domain - mesh of a square unit cell
    functionspace - function space on which to apply the bcs
    bc_type - types of bc to apply on the left/right and top/bottom boundaries,
        respectively. Allowable values are "dirichlet" or "periodic"
    dbc_value - value of the Dirichlet bc
    """

    fdim = domain.topology.dim - 1
    bcs = []
    pbc_directions = []
    pbc_slave_tags = []
    pbc_is_slave = []
    pbc_is_master = []
    pbc_meshtags = []
    pbc_slave_to_master_maps = []

    def generate_pbc_slave_to_master_map(i):
        def pbc_slave_to_master_map(x):
            out_x = x.copy()
            out_x[i] = x[i] - domain.geometry.x.max()
            return out_x

        return pbc_slave_to_master_map

    def generate_pbc_is_slave(i):
        return lambda x: np.isclose(x[i], domain.geometry.x.max())

    def generate_pbc_is_master(i):
        return lambda x: np.isclose(x[i], domain.geometry.x.min())

    # Parse boundary conditions
    for i, bc_type in enumerate(bc_type):
        if bc_type == "dirichlet":
            u_bc = fem.Function(functionspace)
            u_bc.x.array[:] = dbc_value

            def dirichletboundary(x):
                return np.logical_or(
                    np.isclose(x[i], domain.geometry.x.min()),
                    np.isclose(x[i], domain.geometry.x.max()),
                )

            facets = locate_entities_boundary(domain, fdim, dirichletboundary)
            topological_dofs = fem.locate_dofs_topological(functionspace, fdim, facets)
            bcs.append(fem.dirichletbc(u_bc, topological_dofs))

        elif bc_type == "periodic":
            pbc_directions.append(i)
            pbc_slave_tags.append(i + 2)
            pbc_is_slave.append(generate_pbc_is_slave(i))
            pbc_is_master.append(generate_pbc_is_master(i))
            pbc_slave_to_master_maps.append(generate_pbc_slave_to_master_map(i))

            facets = locate_entities_boundary(domain, fdim, pbc_is_slave[-1])
            arg_sort = np.argsort(facets)
            pbc_meshtags.append(
                dolfinx.mesh.meshtags(
                    domain,
                    fdim,
                    facets[arg_sort],
                    np.full(len(facets), pbc_slave_tags[-1], dtype=np.int32),
                )
            )

    # Create MultiPointConstraint object
    mpc = dolfinx_mpc.MultiPointConstraint(functionspace)

    N_pbc = len(pbc_directions)
    for i in range(N_pbc):
        if N_pbc > 1:

            def pbc_slave_to_master_map(x):
                out_x = pbc_slave_to_master_maps[i](x)
                idx = pbc_is_slave[(i + 1) % N_pbc](x)
                out_x[pbc_directions[i]][idx] = np.nan
                return out_x

        else:
            pbc_slave_to_master_map = pbc_slave_to_master_maps[i]

        if functionspace.num_sub_spaces == 0:
            mpc.create_periodic_constraint_topological(
                functionspace,
                pbc_meshtags[i],
                pbc_slave_tags[i],
                pbc_slave_to_master_map,
                bcs,
            )
            print("MPC DEFINED (tag a)")
        else:
            for i in range(functionspace.num_sub_spaces):
                mpc.create_periodic_constraint_topological(
                    functionspace.sub(i),
                    pbc_meshtags[i],
                    pbc_slave_tags[i],
                    pbc_slave_to_master_map,
                    bcs,
                )
                print("SUBSPACE MPC DEFINED (tag b)")

    if len(pbc_directions) > 1:
        # Map intersection(slaves_x, slaves_y) to intersection(masters_x, masters_y),
        # i.e. map the slave dof at (1, 1) to the master dof at (0, 0)
        def pbc_slave_to_master_map(x):
            out_x = x.copy()
            out_x[0] = x[0] - domain.geometry.x.max()
            out_x[1] = x[1] - domain.geometry.x.max()
            idx = np.logical_and(pbc_is_slave[0](x), pbc_is_slave[1](x))
            out_x[0][~idx] = np.nan
            out_x[1][~idx] = np.nan
            return out_x

        if functionspace.num_sub_spaces == 0:
            mpc.create_periodic_constraint_topological(
                functionspace,
                pbc_meshtags[1],
                pbc_slave_tags[1],
                pbc_slave_to_master_map,
                bcs,
            )
            print("MPC DEFINED (tag c)")
        else:
            for i in range(functionspace.num_sub_spaces):
                mpc.create_periodic_constraint_topological(
                    functionspace.sub(i),
                    pbc_meshtags[1],
                    pbc_slave_tags[1],
                    pbc_slave_to_master_map,
                    bcs,
                )
                print("SUBSPACE MPC DEFINED (tag d)")

    mpc.finalize()
    return mpc, bcs


def petsc_to_csr(A):
    """Converting a Petsc matrix to scipy csr array

    parameters
    -----------
    args:
        A - assembled Petsc matrix

    returns
    ---------
    Scipy csr array of A.
    """

    assert isinstance(A, PETSc.Mat)
    ai, aj, av = A.getValuesCSR()
    return sparse.csr_matrix((av, aj, ai))


def petsc_to_csr_complex(*args):
    """Converting a Petsc matrix to complex scipy csr array

    parameters
    -----------
    args:
        args[0] - assembled Petsc matrix containing real part of complex matrix
        args[1] - assembled Petsc matrix containing imaginary part of complex matrix

    returns
    ---------
    A complex scipy csr object. If only 1 arg is passed, then the imaginary values
    are set to 0 by default.
    """

    assert isinstance(args[0], PETSc.Mat)
    ai, aj, av = args[0].getValuesCSR()
    if len(args) > 1:
        assert isinstance(args[1], PETSc.Mat)
        _, _, av_im = args[1].getValuesCSR()
        return sparse.csr_matrix((av + 1j * av_im, aj, ai))
    else:
        return sparse.csr_matrix((av + 0j * av, aj, ai))


def solve_system(
    kx: float,
    ky: float,
    E: float,
    Mcomp: PETSc.Mat,
    mpc: dolfinx_mpc.MultiPointConstraint,
    bcs: list[dolfinx.fem.dirichletbc],
    n_solutions: int,
    mesh: dolfinx.mesh.Mesh,
    u_tr: Argument,
    u_test: Argument,
):
    """Assemble and solve dispersion at a single wavevector point.

    Parameters
    ----------
    kx - x-component of the wavevector
    ky - y-component of the wavevector
    E - modulus
    Mcomp - mass matrix
    mpc - multi-point constraint holding the periodic bcs
    bcs - Dirichlet bcs
    n_solutions - number of eigenvalue/eigenvector pairs to compute
    mesh - mesh of the unit cell
    u_tr - trial function
    u_test - test function
    """

    # Define the bilinear weak form
    K = fem.Constant(mesh, ScalarType((kx, ky)))
    kx = fem.Constant(mesh, ScalarType(kx))
    ky = fem.Constant(mesh, ScalarType(ky))
    a_form_re = (
        E**2
        * (inner(grad(u_tr), grad(u_test)) + u_tr * u_test * (kx**2 + ky**2))
        * dx
    )
    a_form_im = (
        E**2 * (u_tr * inner(grad(u_test), K) - u_test * inner(grad(u_tr), K)) * dx
    )
    a_re = form(a_form_re)
    a_im = form(a_form_im)

    # Assemble Solve the eigenproblem
    diagval_A = 1e6
    A_re = dolfinx_mpc.assemble_matrix(a_re, mpc, bcs=bcs, diagval=diagval_A)
    A_im = dolfinx_mpc.assemble_matrix(a_im, mpc, bcs=bcs, diagval=diagval_A)
    A_re.assemble()
    A_im.assemble()
    Kcomp = petsc_to_csr_complex(A_re, A_im)
    eval, evec = eigsh(Kcomp, M=Mcomp, k=n_solutions, sigma=0.1)
    return eval, evec


def solve_system_complex(
    eigensolver: SLEPc.EPS,
    n_solutions: int,
    fs: fem.FunctionSpace,
    mpc: dolfinx_mpc.MultiPointConstraint,
):
    """Solve dispersion and extract eigenfrequencies and eigenvectors.

    Parameters
    ----------
    - `eigensolver` SLEPc eigensolver object
    - `n_solutions` number of eigenvalue/eigenvector pairs to compute
    """

    eig_val = []
    eig_vec = []
    eigensolver.solve()
    for j in range(eigensolver.getConverged()):
        eig_val.append(eigensolver.getEigenvalue(j))
        vec = dolfinx.la.create_petsc_vector(fs.dofmap.index_map, fs.dofmap.bs)
        e_vec = eigensolver.getEigenvector(j, vec)
        mpc.backsubstitution(vec)
        eig_vec.append(e_vec)
    return np.array(eig_val), eig_vec


def assign_mat_props(
    mesh: dolfinx.mesh.Mesh, rho: list[float], c: list[float], ct: dolfinx.mesh.meshtags
):
    """Assign material properties to a domain.

    parameters
    ----------
        mesh - the dolfinx mesh
        rho - a list of densities for each mesh tag
        c - a list of wave speeds for each mesh tag
    returns
    -------
        A set of arrays containing mass and wave speed
        for each mesh coordinate
    """

    Q = FunctionSpace(mesh, ("DG", 0))
    E = Function(Q)
    Rho = Function(Q)
    for i, (rho_i, c_i) in enumerate(zip(rho, c)):
        # Assign material propoerties to each physical group.
        cells = ct.find(i + 1)
        Rho.interpolate(lambda x: np.full((x.shape[1],), rho_i), cells=cells)
        E.interpolate(lambda x: np.full((x.shape[1],), c_i**2 * rho_i), cells=cells)
    return E, Rho


def mass_matrix_complex(u_tr, u_test, Rho, mpc, bcs):
    """Get the complex valued mass matrix

    parameters
    ----------
        u_tr - trial function
        u_test - test function
        Rho - Array of densities on mesh coords
        mpc - multi-point periodic constraint
        bcs - Dirichlet BCs

    returns
    -------
        Complex valued mass matrix is scipy csr format
    """

    m_form = Rho * dot(u_tr, u_test) * dx
    m = form(m_form)
    diagval_B = 1e-2
    B = dolfinx_mpc.assemble_matrix(m, mpc, bcs=bcs, diagval=diagval_B)
    B.assemble()
    return petsc_to_csr_complex(B)


def solve_bands(
    HSpts: list = None,
    HSstr: list = None,
    n_wavevector: int = 60,
    n_solutions: int = 20,
    a_len: float = 1,
    c: list = [1e2],
    rho: list = [5e1],
    fspace: str = "CG",
    mesh: dolfinx.mesh.Mesh = None,
    ct: dolfinx.mesh.meshtags = None,
    order: int = 1,
):
    """Solve the band stucture on Γ-X-M-Γ.

    If the PETSc scalar type is complex, the problem is assumed to be (possibly) lossy,
    assembled using a fully complex formulation, and solved using SLEPc. If the PETSc
    scalar type is real, the problem is assumed to be lossless, assembled using a split
    formulation, and solved using Scipy.

    parameters
    ----------
    n_solutions  - Number of eigensolutions to generate for each wavevector
    n_wavevector - Number of wavevectors in the IBZ
    a_len   - Chareteristic unit cell length
    c       - Speed of sound in the material
    rho     - Desnity of the material
    fspace  - Function space to use
    mesh    - The mesh to solve over
    ct      - The cell tags on the mesh
    order   - The order of the function space
    HSpts   - A list of the high-symmetry points to sovle

    output
    -----------
    evals_disp  - The eigenvalues of each wavevector
    evecs_disp  - The eigenvectors of each wavevector
    mpc         - The multi-point constraint on the mesh
    """

    if HSpts is None:
        # Setting the high symmetr points
        P1 = [0, 0]  # Gamma
        P2 = [np.pi / a_len, 0]  # X
        P3 = [np.pi / a_len, np.pi / a_len]  # K
        HSpts = [P1, P2, P3, P1]
        HSstr = ["Γ", "X", "M", "Γ"]
    if HSstr is None:
        HSstr = np.linspace(0, len(HSpts), len(HSpts) + 1)

    # Get Material Properties
    E, Rho = assign_mat_props(mesh, rho, c, ct)

    # Define the function spaces and the mesh constraint
    V = FunctionSpace(mesh, (fspace, order))
    mpc, bcs = dirichlet_and_periodic_bcs(mesh, V, ["periodic", "periodic"])

    # Define trial and test functions
    u_tr = TrialFunction(V)
    u_test = TestFunction(V)

    # Intitialize loop params
    nvec_per_HS = int(round(n_wavevector / len(HSpts)))
    evals_disp, evecs_disp = [], []
    start = time.time()
    KX, KY = [], []
    kx, ky = HSpts[0][0], HSpts[0][1]
    KX.append(kx)
    KY.append(ky)

    if ScalarType == np.float64:
        # Make mass matrix
        Mcomp = mass_matrix_complex(u_tr, u_test, Rho, mpc, bcs)

        # Loop to compute band structure
        print("Computing Band Structure... ")
        for k in range(len(HSpts) - 1):
            print("Computing " + str(HSstr[k]) + " to " + str(HSstr[k + 1]))
            slope = np.array(HSpts[k + 1]) - np.array(HSpts[k])
            nsolve = nvec_per_HS
            for j in range(nsolve):
                kx = kx + slope[0] / nvec_per_HS
                ky = ky + slope[1] / nvec_per_HS
                ky = 0 if np.isclose(ky, 0) else ky
                kx = 0 if np.isclose(kx, 0) else kx
                KX.append(kx)
                KY.append(ky)
                eval, evec = solve_system(
                    kx, ky, E, Mcomp, mpc, bcs, n_solutions, mesh, u_tr, u_test
                )
                eval[np.isclose(eval, 0)] == 0
                eigfrq_sp_cmp = np.abs(np.real(eval)) ** 0.5
                eigfrq_sp_cmp = np.sort(eigfrq_sp_cmp)
                evals_disp.append(eigfrq_sp_cmp)
                evecs_disp.append(evec)

        t2 = round(time.time() - start, 3)

        print("Time to compute dispersion " + str(t2))
        print("Band computation complete")
        print("-----------------")
        print("N_dof...." + str(ct.values.shape[0]))
        print("N_vectors...." + str(n_solutions))
        print("N_wavenumbers...." + str(n_wavevector))
        print("T total...." + str(round(t2, 3)))

        return evals_disp, evecs_disp, mpc, KX, KY
    else:
        from fenonics import solver

        wave_vec = fem.Constant(mesh, ScalarType((0.0, 0.0)))
        mass_form = problem.mass_form(
            BlochProblemType.INDIRECT_TRANSFORMED, u_tr, u_test, Rho, wave_vec=wave_vec
        )
        (stiffness_form,) = problem.stiffness_form(
            BlochProblemType.INDIRECT_TRANSFORMED, u_tr, u_test, E, wave_vec=wave_vec
        )
        M = dolfinx_mpc.assemble_matrix(mass_form, mpc, bcs=bcs, diagval=1e-2)
        K = None
        eigensolver = None

        # Loop to compute band structure
        print("Computing Band Structure... ")
        for k in range(len(HSpts) - 1):
            print(f"Computing {HSstr[k]} to {HSstr[k + 1]}")
            slope = np.array(HSpts[k + 1]) - np.array(HSpts[k])
            nsolve = nvec_per_HS
            for j in range(nsolve):
                # Compute wavevector
                kx = kx + slope[0] / nvec_per_HS
                ky = ky + slope[1] / nvec_per_HS
                ky = 0 if np.isclose(ky, 0) else ky
                kx = 0 if np.isclose(kx, 0) else kx
                KX.append(kx)
                KY.append(ky)
                wave_vec.value = (kx, ky)

                # Assemble
                K = dolfinx_mpc.assemble_matrix(
                    stiffness_form, mpc, bcs=bcs, diagval=1e6, A=K
                )
                if eigensolver is None:
                    eigensolver = solver.get_EPS(K, M)
                else:
                    # TODO: this command may be unnecessary because we reuse K and M
                    eigensolver.setOperators(K, M)

                # Solve
                eig_val, eig_vec = solve_system_complex(
                    eigensolver, n_solutions, V, mpc
                )
                eig_val[np.isclose(eig_val, 0)] == 0
                eig_frq = np.abs(np.real(eig_val)) ** 0.5
                eig_frq = np.sort(eig_frq)
                evals_disp.append(eig_frq)
                evecs_disp.append(eig_vec)
        return evals_disp, evecs_disp, mpc, KX, KY

        t2 = time.time() - start

        print("Band computation complete")
        print("-------------------------")
        print(f"N_dof... {ct.values.shape[0]}")
        print(f"N_vectors... {n_solutions}")
        print(f"N_wavenumbers... {n_wavevector}")
        print("Time to compute dispersion: {t2:.3f}s".format(t2=t2))


########################################################################################
#                                   Testing the code
########################################################################################

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from dolfinx.io.gmshio import model_to_mesh
    import gmsh
    from PostProcess import *
    from MeshFunctions import get_mesh_SquareSpline

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
        c = [1500, 5100]  # if solid inclusion (mutlple materail model)
        rho = [1e3, 7e3]  # if solid inclusion (mutlple materail model)
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
        inset=True,
    )
    plt.show()
