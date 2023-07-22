

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


#%%  # noqa E265
# Imports


# General
import numpy as np
import time
import warnings

# Finite element modeling
import dolfinx
import dolfinx_mpc
from dolfinx.fem import form, Function, FunctionSpace
from dolfinx.mesh import locate_entities_boundary
from dolfinx import fem
from mpi4py import MPI
from petsc4py import PETSc
from petsc4py.PETSc import ScalarType
from scipy import sparse
from scipy.sparse.linalg import eigsh
from slepc4py import SLEPc
from ufl import dot, dx, grad, inner, Argument, TestFunction, TrialFunction


# ======================================================================================
#
#                               FINITE ELEMENT FUNCTIONS
#
# ======================================================================================


def petsc_to_numpy(A):
    '''Convert PETSc matrix to dense numpy matrix.'''

    warnings.warn("This function is inefficient and will be removed in the future.",
                  category=DeprecationWarning)

    sz = A.getSize()
    A_ = np.zeros(sz, dtype=ScalarType)
    for i in range(sz[0]):
        row = A.getRow(i)
        A_[i, row[0]] = row[1]
    return A_


def complex_formatter(x):
    '''Convert complex number to string with exponential notation.'''

    return "{0:-10.3e}{1:+10.3e}j".format(np.real(x), np.imag(x))


def get_EPS(A: PETSc.Mat, B: PETSc.Mat, nvec: int):
    '''Create and set up a SLEPc eigensolver.'''

    EPS = SLEPc.EPS()
    EPS.create(comm=MPI.COMM_WORLD)
    EPS.setOperators(A, B)
    EPS.setProblemType(SLEPc.EPS.ProblemType.GNHEP)
    EPS.setDimensions(nev=nvec)
    EPS.setType(SLEPc.EPS.Type.KRYLOVSCHUR)
    EPS.setWhichEigenpairs(SLEPc.EPS.Which.TARGET_MAGNITUDE)
    EPS.setTarget(0)
    EPS.setTolerances(tol=1e-5, max_it=12)
    ST = EPS.getST()
    ST.setType(SLEPc.ST.Type.SINVERT)
    ST.setShift(10)
    EPS.setST(ST)
    EPS.setFromOptions()
    return EPS


def scipysolve(A: PETSc.Mat, B: PETSc.Mat, nval: int):
    '''Solve eigenvalue problem using scipy sparse eigensolver.'''

    K = petsc_to_numpy(A)
    M = petsc_to_numpy(B)
    K = sparse.csr_array(K)
    M = sparse.csr_array(M)

    eval, evec = eigsh(K, M=M, k=nval, sigma=1.0)
    return eval, evec

def scipysolve_complex(A_re: PETSc.Mat, A_im: PETSc.Mat, B: PETSc.Mat, nval: int):
    '''Solve complex eigenvalue problem using scipy sparse eigensolver.

    In particular, solve the generalized eigenvalue problem

        A * x_i = lambda_i * B * x_i

    where `A = A_re + 1j * A_im` is a complex-valued matrix.
    '''

    A_re_np = petsc_to_numpy(A_re)
    A_im_np = petsc_to_numpy(A_im)
    Kcomp = A_re_np + 1j * A_im_np

    eval, evec = eigsh(Kcomp, M=B, k=24, sigma=1.0)
    return eval, evec


def dirichlet_and_periodic_bcs(
    domain: dolfinx.mesh.Mesh,
    functionspace: fem.FunctionSpace,
    bc_type: list[str] = ["peroidic", "periodic"],
    dbc_value: ScalarType = 0,
):
    '''Create periodic and/or Dirichlet boundary conditions for a square domain.

    Parameters
    ----------
    domain - mesh of a square unit cell
    functionspace - function space on which to apply the bcs
    bc_type - types of bc to apply on the left/right and top/bottom boundaries,
        respectively. Allowable values are "dirichlet" or "periodic"
    dbc_value - value of the Dirichlet bc
    '''

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
                return np.logical_or(np.isclose(x[i],
                                     domain.geometry.x.min()),
                                     np.isclose(x[i],
                                     domain.geometry.x.max()))
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
                    np.full(len(facets), pbc_slave_tags[-1], dtype=np.int32)
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
            mpc.create_periodic_constraint_topological(functionspace, pbc_meshtags[i],
                                                       pbc_slave_tags[i],
                                                       pbc_slave_to_master_map,
                                                       bcs)
            print('MPC DEFINED (tag a)')
        else:
            for i in range(functionspace.num_sub_spaces):
                mpc.create_periodic_constraint_topological(
                    functionspace.sub(i), pbc_meshtags[i],
                    pbc_slave_tags[i],
                    pbc_slave_to_master_map,
                    bcs,
                )
                print('SUBSPACE MPC DEFINED (tag b)')

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
            mpc.create_periodic_constraint_topological(functionspace, pbc_meshtags[1],
                                                       pbc_slave_tags[1],
                                                       pbc_slave_to_master_map,
                                                       bcs)
            print('MPC DEFINED (tag c)')

        else:
            for i in range(functionspace.num_sub_spaces):
                mpc.create_periodic_constraint_topological(
                    functionspace.sub(i), pbc_meshtags[1],
                    pbc_slave_tags[1],
                    pbc_slave_to_master_map,
                    bcs,
                )
                print('SUBSPACE MPC DEFINED (tag d)')

    mpc.finalize()
    return mpc, bcs

def solvesys(kx: float, ky: float, E: float, Mcomp: PETSc.Mat,
             mpc: dolfinx_mpc.MultiPointConstraint, bcs: list[dolfinx.fem.dirichletbc],
             nvec: int, mesh: dolfinx.mesh.Mesh, u_tr: Argument, u_test: Argument):
    '''Assemble and solve dispersion at a single wavevector point.

    Parameters
    ----------
    kx - x-component of the wavevector
    ky - y-component of the wavevector
    E - modulus
    Mcomp - mass matrix
    mpc - multi-point constraint holding the periodic bcs
    bcs - Dirichlet bcs
    nvec - number of eigenvalue/eigenvector pairs to compute
    mesh - mesh of the unit cell
    u_tr - trial function
    u_test - test function
    '''

    K = fem.Constant(mesh, ScalarType((kx, ky)))
    kx = fem.Constant(mesh, ScalarType(kx))
    ky = fem.Constant(mesh, ScalarType(ky))
    a_form_re = E**2 * (inner(grad(u_tr), grad(u_test)) + u_tr*u_test*(kx**2+ky**2))*dx
    a_form_im = E**2 * (u_tr*inner(grad(u_test), K) - u_test*inner(grad(u_tr), K))*dx
    a_re = form(a_form_re)
    a_im = form(a_form_im)
    diagval_A = 1e8
    A_re = dolfinx_mpc.assemble_matrix(a_re, mpc, bcs=bcs, diagval=diagval_A)
    A_im = dolfinx_mpc.assemble_matrix(a_im, mpc, bcs=bcs, diagval=diagval_A)
    ############################################
    # efficient conversion to scipy for
    # solving the complex problem (recommended)
    ############################################
    A_re.assemble()
    assert isinstance(A_re, PETSc.Mat)
    ai, aj, av = A_re.getValuesCSR()
    A_im.assemble()
    assert isinstance(A_im, PETSc.Mat)
    _, _, av_im = A_im.getValuesCSR()
    ############################################
    # Getting solutions
    ############################################
    Kcomp = sparse.csr_matrix((av+1j*av_im, aj, ai))
    eval, evec = eigsh(Kcomp, M=Mcomp, k=nvec, sigma=1.0)
    return eval, evec


def solve_bands(np1, np2, np3, nvec, a_len, c, rho, fspace, mesh, ct):
    '''Solve the band stucture on Γ-X-M-Γ.'''

    ##################################
    # Get Material Properties
    ###################################
    E, Rho = getMatProps(mesh, rho, c, ct)

    V = FunctionSpace(mesh, (fspace, 1))
    mpc, bcs = dirichlet_and_periodic_bcs(mesh, V, ["periodic", "periodic"])
    u_tr = TrialFunction(V)
    u_test = TestFunction(V)
    m_form = Rho * dot(u_tr, u_test) * dx

    #################################################################
    #        Define Mass Mat  outside of  IBZ loop                  #
    #################################################################
    m = form(m_form)
    diagval_B = 1e-2
    B = dolfinx_mpc.assemble_matrix(m, mpc, bcs=bcs, diagval=diagval_B)
    B.assemble()
    assert isinstance(B, PETSc.Mat)
    ai, aj, av = B.getValuesCSR()
    Mcomp = sparse.csr_matrix((av + 0j * av, aj, ai))

    ky = 0
    evals_disp = []
    maxk = np.pi/a_len
    start = time.time()
    evec_all = []
    print('Computing Band Structure... ')

    #################################################################
    #            Computing Γ to Χ                                   #
    #################################################################
    print('Computing Γ to X')
    for kx in np.linspace(0.01, maxk, np1):
        eval, evec = solvesys(kx, ky, E, Mcomp, mpc, bcs, nvec, mesh, u_tr, u_test)
        eval[np.isclose(eval, 0)] == 0
        eigfrq_sp_cmp = np.real(eval) ** 0.5
        eigfrq_sp_cmp = np.sort(eigfrq_sp_cmp)
        evals_disp.append(eigfrq_sp_cmp)
        evec_all.append(evec)

    #################################################################
    #            Computing Χ to M                                   #
    #################################################################
    print('Computing X to M')
    kx = maxk
    for ky in np.linspace(0.01, maxk, np2):
        eval, evec = solvesys(kx, ky, E, Mcomp, mpc, bcs, nvec, mesh, u_tr, u_test)
        eval[np.isclose(eval, 0)] == 0
        eigfrq_sp_cmp = np.real(eval) ** 0.5
        eigfrq_sp_cmp = np.sort(eigfrq_sp_cmp)
        evals_disp.append(eigfrq_sp_cmp)
        evec_all.append(evec)

    #################################################################
    #            Computing M To Γ                                   #
    #################################################################
    print('Computing M to Γ')
    for kx in np.linspace(maxk, 0.01, np3):
        ky = kx
        eval, evec = solvesys(kx, ky, E, Mcomp, mpc, bcs, nvec, mesh, u_tr, u_test)
        eval[np.isclose(eval, 0)] == 0
        eigfrq_sp_cmp = np.real(eval) ** 0.5
        eigfrq_sp_cmp = np.sort(eigfrq_sp_cmp)
        evals_disp.append(eigfrq_sp_cmp)
        evec_all.append(evec)
        # print(np.round(time.time()-start,3))
    t2 = round(time.time() - start, 3)
    print('Time to compute dispersion ' + str(t2))

    print('Band computation complete')
    print('-----------------')
    print('N_dof....' + str(ct.values.shape[0]))
    print('N_vectors....' + str(nvec))
    print('N_wavenumbers....' + str(np1 + np1 + np3))
    # print('Ttoal Eigenproblems....'  + str(nvec*(np1+np1+np3)))
    print('T total....' + str(round(t2, 3)))

    return evals_disp, evec_all, mpc


def getMatProps(mesh: dolfinx.mesh.Mesh, rho: list[float], c: list[float],
                ct: dolfinx.mesh.meshtags):
    '''Assign material properties to a domain.'''

    if len(rho) > 1:
        # E.g., if more than one physical group assigned.
        # Assign material propoerties to each physical group.
        Q = FunctionSpace(mesh, ("DG", 0))
        E = Function(Q)
        Rho = Function(Q)
        # material_tags = np.unique(ct.values)
        disk1_cells = ct.find(1)
        disk2_cells = ct.find(2)
        Rho.x.array[disk1_cells] = np.full_like(disk1_cells, rho[0], dtype=ScalarType)
        Rho.x.array[disk2_cells] = np.full_like(disk2_cells, rho[1], dtype=ScalarType)
        E.x.array[disk1_cells] = np.full_like(disk1_cells, c[0], dtype=ScalarType)
        E.x.array[disk2_cells] = np.full_like(disk2_cells, c[1], dtype=ScalarType)

    else:
        Rho = rho[0]
        E = c[0]
    return E, Rho


def solve_bands_repo(HSpts=None, nsol=60, nvec=20, a_len=1, c=[1e2], rho=[5e1],
                     fspace='cg', mesh=None, ct=None):
    '''Solve band stucture, interpolating between high-symmetry points.'''

    # Get Material Properties
    E, Rho = getMatProps(mesh, rho, c, ct)

    # Define the function spaces
    V = FunctionSpace(mesh, (fspace, 1))
    mpc, bcs = dirichlet_and_periodic_bcs(mesh, V, ["periodic", "periodic"])
    u_tr = TrialFunction(V)
    u_test = TestFunction(V)
    m_form = Rho * dot(u_tr, u_test) * dx

    # Form the mass matrix
    m = form(m_form)
    diagval_B = 1e-2
    B = dolfinx_mpc.assemble_matrix(m, mpc, bcs=bcs, diagval=diagval_B)
    B.assemble()
    assert isinstance(B, PETSc.Mat)
    ai, aj, av = B.getValuesCSR()
    Mcomp = sparse.csr_matrix((av + 0j*av, aj, ai))

    # Initializing data
    evals_disp = []
    start = time.time()
    evec_all = []
    print('Computing band structure...')

    # Computing dispersion across the high-symmmetry points
    print('Computing HS points 1 to 2')

    nvec_per_HS = int(round(nsol / len(HSpts)))
    kx = HSpts[0][0]
    ky = HSpts[0][0]
    KX = []
    KY = []
    KX.append(kx)
    KY.append(ky)
    for k in range(len(HSpts) - 1):
        # Get slope along IBZ boundary partition
        slope = np.array(HSpts[k + 1]) - np.array(HSpts[k])

        # Compute eigenvectors/values on line
        for j in range(nvec_per_HS):
            kx = kx + slope[0] / nvec_per_HS
            ky = ky + slope[1] / nvec_per_HS
            KX.append(kx)
            KY.append(ky)
            eval, evec = solvesys(kx, ky, E, Mcomp, mpc, bcs, nvec, mesh, u_tr, u_test)
            eval[np.isclose(eval, 0)] == 0
            eigfrq_sp_cmp = np.real(eval) ** 0.5
            eigfrq_sp_cmp = np.sort(eigfrq_sp_cmp)
            evals_disp.append(eigfrq_sp_cmp)
            evec_all.append(evec)

    t2 = time.time() - start
    print('Time to compute dispersion: {0:.3f}s'.format(t2))

    print('Band computation complete')
    print('-----------------')
    print('N_dof...{0:d}'.format(ct.values.shape[0]))
    print('N_vectors...{0:d}'.format(nvec))
    print('N_wavenumbers...{0:d}'.format(nsol))
    # print('Ttoal Eigenproblems....'  + str(nvec*(np1+np1+np3)))
    print('T total...{0:.3f}'.format(t2))

    return evals_disp, evec_all, mpc, KX, KY
