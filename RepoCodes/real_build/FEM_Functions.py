

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



#%%

# ################################################## #
# General imports                                    #
# ################################################## #
import numpy as np
import time
import os

# ################################################## #
# Imports for plotting                               #
# ################################################## #
import matplotlib.pyplot as plt
from dolfinx.io import gmshio
import pyvista
pyvista.start_xvfb()

# ################################################## #
# Imports for the meshing                            #
# ################################################## #
import gmsh
from MeshFunctions      import get_mesh_SquareSpline,get_mesh_SquareMultiSpline
from dolfinx.plot       import create_vtk_mesh
from dolfinx.io         import XDMFFile
from dolfinx.io.gmshio  import model_to_mesh

# ################################################## #
# Imports for finite element modeling                #
# ################################################## #
from mpi4py import MPI
import dolfinx_mpc
import dolfinx
from dolfinx.fem    import Function, FunctionSpace, VectorFunctionSpace
from dolfinx.mesh   import create_unit_square 
from dolfinx.mesh   import create_rectangle
from dolfinx.fem    import form
from dolfinx.fem.petsc import assemble_matrix
from dolfinx    import plot
from dolfinx.io import gmshio
from mpi4py     import MPI
from dolfinx    import fem
from ufl        import TrialFunction, TestFunction, grad, dx, dot, nabla_div, Identity, nabla_grad, inner, sym
from petsc4py.PETSc import ScalarType
from slepc4py   import SLEPc
from petsc4py   import PETSc
from typing     import List
import scipy.sparse
from scipy.sparse.linalg import eigsh
from scipy import sparse


# ==============================================================================================
#
#                               FINITE ELEMENT FUNCTIONS
#
# ==============================================================================================

#################################################################
#       Function for matrix conversion                          #
#################################################################
def petsc_to_numpy(A):
    '''
    ===================================
        Convert to np (not recomended)
    ===================================
    '''
    sz = A.getSize()
    A_ = np.zeros(sz, dtype=PETSc.ScalarType)
    for i in range(sz[0]):
        row = A.getRow(i)
        A_[i, row[0]] = row[1]
    return A_

def complex_formatter(x):
    return "{0:-10.3e}{1:+10.3e}j".format(np.real(x), np.imag(x))



#################################################################
#       Function for solving eigenproblem with SLEPc            #
#################################################################

def get_EPS(A, B, nvec):
    '''
    ===================================
        Solving evals with slepc (real)
    ===================================
    '''
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

#################################################################
#       Function for solving eigenproblem with scipy            #
#################################################################
def scipysolve(A,B, nval):
    '''
    ===================================
        Solving evals with scipy (real)
    ===================================
    '''
    K = petsc_to_numpy(A)       # Not recomended #
    M = petsc_to_numpy(B)       # Not recomended #
    K = sparse.csr_array(K)
    M = sparse.csr_array(M)
    eval, evec = eigsh(K, k = nval, M=M, sigma = 1.0)
    return eval, evec

#################################################################
#   Function for solving complex eigenproblem with scipy        #
#################################################################
def scipysolve_complex(A_re, A_im, B, nval):
    '''
    ===================================
        Solving evals with scipy (complex)
    ===================================
    '''
    A_re_np = petsc_to_numpy(A_re)       # Not recomended #
    A_im_np = petsc_to_numpy(A_im)       # Not recomended #
    Kcomp    = A_re_np+1j*A_im_np
    eval, evec = eigsh(Kcomp, k = 24, M=Mcomp, sigma = 1.0)
    return eval, evec
    
    

#################################################################
#   Function for multi point constraint (periodic BC)           #
#################################################################
def dirichlet_and_periodic_bcs(domain, functionspace, boundary_condition: List[str] = ["dirichlet", "periodic"], dbc_value = 0):
    """
    Function to set either dirichlet or periodic boundary conditions
    ----------
    boundary_condition
        First item describes b.c. on {x=0} and {x=1}
        Second item describes b.c. on {y=0} and {y=1}
    """
    
    fdim = domain.topology.dim - 1
    bcs             = []
    pbc_directions  = []
    pbc_slave_tags  = []
    pbc_is_slave    = []
    pbc_is_master   = []
    pbc_meshtags    = []
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
    for i, bc_type in enumerate(boundary_condition):
        
        if bc_type == "dirichlet":
            u_bc = fem.Function(functionspace)
            u_bc.x.array[:] = dbc_value # value of dirichlet bc needs to be passed into this function!

            def dirichletboundary(x):
                return np.logical_or(np.isclose(x[i], domain.geometry.x.min()), np.isclose(x[i], domain.geometry.x.max()))
            facets = locate_entities_boundary(domain, fdim, dirichletboundary)
            topological_dofs = fem.locate_dofs_topological(functionspace, fdim, facets)
            bcs.append(fem.dirichletbc(u_bc, topological_dofs))
        
        elif bc_type == "periodic":
            pbc_directions.append(i)
            pbc_slave_tags.append(i + 2)
            pbc_is_slave.append(generate_pbc_is_slave(i))
            pbc_is_master.append(generate_pbc_is_master(i))
            pbc_slave_to_master_maps.append(generate_pbc_slave_to_master_map(i))

            facets = dolfinx.mesh.locate_entities_boundary(domain, fdim, pbc_is_slave[-1])
            arg_sort = np.argsort(facets)
            pbc_meshtags.append(dolfinx.mesh.meshtags(domain,
                                        fdim,
                                        facets[arg_sort],
                                        np.full(len(facets), pbc_slave_tags[-1], dtype=np.int32)))

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

        # mpc.create_periodic_constraint_topological(functionspace, pbc_meshtags[i],
        #                                             pbc_slave_tags[i],
        #                                             pbc_slave_to_master_map,
        #                                             bcs)
        
        # print('MPC DEFINED (a)')
        if functionspace.num_sub_spaces == 0:
            mpc.create_periodic_constraint_topological(functionspace, pbc_meshtags[i],
                                            pbc_slave_tags[i],
                                            pbc_slave_to_master_map,
                                            bcs)
            print('MPC DEFINED (tag a)')
        else:
            for i in range(functionspace.num_sub_spaces):
                mpc.create_periodic_constraint_topological(functionspace.sub(i), pbc_meshtags[i],
                                                pbc_slave_tags[i],
                                                pbc_slave_to_master_map,
                                                bcs)
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
        # mpc.create_periodic_constraint_topological(functionspace, pbc_meshtags[1],
        #                                             pbc_slave_tags[1],
        #                                             pbc_slave_to_master_map,
        #                                             bcs)
        # print('MPC DEFINED (c)')
        
        if functionspace.num_sub_spaces == 0:
            mpc.create_periodic_constraint_topological(functionspace, pbc_meshtags[1],
                                                pbc_slave_tags[1],
                                                pbc_slave_to_master_map,
                                                bcs)
            print('MPC DEFINED (tag c)')
            
        else:
            for i in range(functionspace.num_sub_spaces):
                mpc.create_periodic_constraint_topological(functionspace.sub(i), pbc_meshtags[1],
                                                    pbc_slave_tags[1],
                                                    pbc_slave_to_master_map,
                                                    bcs)
                print('SUBSPACE MPC DEFINED (tag d)')
            
    mpc.finalize()
    return mpc, bcs

#################################################################
#    Function to assemble and solve system using Scipy          #
#################################################################
def solvesys(kx, ky, E, Mcomp, mpc, bcs, nvec, mesh, u_tr, u_test):
    '''
    ===================================
        Solving the FEM problem
    ===================================
    '''
    K = fem.Constant(mesh, PETSc.ScalarType((kx,ky)))
    kx = fem.Constant(mesh,PETSc.ScalarType(kx))
    ky  = fem.Constant(mesh,PETSc.ScalarType(ky))
    a_form_re = E**2*(inner(grad(u_tr), grad(u_test)) + u_tr*u_test*(kx**2+ky**2))*dx
    a_form_im = E**2*(u_tr*inner(grad(u_test),K) - u_test*inner(grad(u_tr),K))*dx
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
    _,_, av_im = A_im.getValuesCSR()
    ############################################
    # Getting solutions
    ############################################
    Kcomp = scipy.sparse.csr_matrix((av+1j*av_im, aj, ai))
    eval, evec= eigsh(Kcomp, k = nvec, M=Mcomp, sigma = 1.0)
    return eval, evec 

#################################################################
#               SOLVING THE DISPERSION PROBLEM                  #
#################################################################
def solve_bands(np1, np2, np3, nvec, a_len, c, rho, fspace, mesh, ct):
    '''Solve the band stucture on Γ-X-M-Γ.'''
    
    ##################################
    # Get Material Properties
    ###################################
    E, Rho = getMatProps(mesh,rho,c,ct)

    V = FunctionSpace(mesh, (fspace, 1))
    mpc, bcs    = dirichlet_and_periodic_bcs(mesh, V, ["periodic", "periodic"]) 
    u_tr        = TrialFunction(V)
    u_test      = TestFunction(V) 
    m_form      = Rho*dot(u_tr, u_test)*dx

    #################################################################
    #        Define Mass Mat  outside of  IBZ loop                  #
    #################################################################
    m = form(m_form)
    diagval_B = 1e-2
    B = dolfinx_mpc.assemble_matrix(m, mpc, bcs=bcs, diagval=diagval_B)
    B.assemble()
    assert isinstance(B, PETSc.Mat)
    ai, aj, av = B.getValuesCSR()
    Mcomp = scipy.sparse.csr_matrix((av + 0j * av, aj, ai))
    
    ky          = 0
    evals_disp  = []
    maxk        = np.pi/a_len
    start       = time.time()
    evec_all    = []
    print('Computing Band Structure... ')

    #################################################################
    #            Computing Γ to Χ                                   #
    #################################################################
    print('Computing Γ to X')
    for kx in np.linspace(0.01, maxk, np1):
        K = fem.Constant(mesh, PETSc.ScalarType((kx, ky)))
        eval, evec = solvesys(kx, ky, E, Mcomp, mpc, bcs, nvec, mesh, u_tr, u_test)
        eval[np.isclose(eval,0)] == 0
        eigfrq_sp_cmp = np.real(eval)**.5
        eigfrq_sp_cmp = np.sort(eigfrq_sp_cmp )
        evals_disp.append(eigfrq_sp_cmp )
        evec_all.append(evec)
        
    #################################################################
    #            Computing Χ to M                                   #
    #################################################################
    print('Computing X to M')
    kx = maxk
    for ky in np.linspace(0.01, maxk, np2):
        K = fem.Constant(mesh, PETSc.ScalarType((kx, ky)))
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
        evals_disp.append(eigfrq_sp_cmp )
        evec_all.append(evec)
        # print(np.round(time.time()-start,3))
    t2 = round(time.time() - start, 3)
    print('Time to compute dispersion '+ str(t2))

    print('Band computation complete')
    print('-----------------')
    print('N_dof....'  + str(ct.values.shape[0]))
    print('N_vectors....'  + str(nvec))
    print('N_wavenumbers....'  + str(np1 + np1 + np3))
    # print('Ttoal Eigenproblems....'  + str(nvec*(np1+np1+np3)))
    print('T total....'  + str(round(t2, 3))) 

    return evals_disp, evec_all, mpc



#################################################################
#              Plot the mesh                                    #
#################################################################
def plotmesh(mesh, fspace, ct):
    V = FunctionSpace(mesh, (fspace, 1))
    v = Function(V)
    plotter = pyvista.Plotter()
    grid    = pyvista.UnstructuredGrid(*create_vtk_mesh(mesh, mesh.topology.dim))
    num_local_cells = mesh.topology.index_map(mesh.topology.dim).size_local
    grid.cell_data["Marker"] = 1 - ct.values[ct.indices < num_local_cells]
    grid.set_active_scalars("Marker")
    actor = plotter.add_mesh(grid, show_edges=True, line_width=3, edge_color='k', style='wireframe')
    plotter.set_background('white', top='white')
    plotter.view_xy()
    plotter.camera.tight(padding=0.1)
    plotter.add_title(f'CELLS: {ct.values.shape[0]}  | NODES: {v.vector[:].shape[0]}')
    return plotter

#################################################################
#       Assign material properties to the domain                #
#################################################################
def getMatProps(mesh, rho, c, ct):
    if len(rho) > 1:
        # E.g., if more than one physical group assigned.
        # Assign material propoerties to each physical group.
        Q = FunctionSpace(mesh, ("DG", 0))
        E = Function(Q)
        Rho = Function(Q)    
        material_tags   = np.unique(ct.values)
        disk1_cells     = ct.find(1)
        disk2_cells     = ct.find(2)
        Rho.x.array[disk1_cells]    = np.full_like(disk1_cells, rho[0], dtype=ScalarType)
        Rho.x.array[disk2_cells]    = np.full_like(disk2_cells, rho[1], dtype=ScalarType)
        E.x.array[disk1_cells]      = np.full_like(disk1_cells, c[0], dtype=ScalarType)
        E.x.array[disk2_cells]      = np.full_like(disk2_cells, c[1], dtype=ScalarType)

    else:
        Rho = rho[0]
        E   = c[0]
    return E, Rho


#################################################

def solve_bands_repo(
        HSpts = None,
        nsol  = 60, 
        nvec = 20, 
        a_len = 1, 
        c = [1e2], 
        rho = [5e1], 
        fspace = 'cg', 
        mesh = None,
        ct = None
    ):
    
    '''
        Solving the band stucture
    '''
    
    ##################################
    # Get Material Properties
    ###################################
    E, Rho = getMatProps(mesh, rho, c, ct)

    # Define the function spaces 
    V = FunctionSpace(mesh,(fspace,1))
    mpc, bcs    = dirichlet_and_periodic_bcs(mesh, V, ["periodic", "periodic"]) 
    u_tr        = TrialFunction(V)
    u_test      = TestFunction(V) 
    m_form      = Rho * dot(u_tr, u_test) * dx

    # Form the mass matrix
    m = form(m_form)
    diagval_B = 1e-2
    B = dolfinx_mpc.assemble_matrix(m, mpc, bcs=bcs, diagval=diagval_B)
    B.assemble()
    assert isinstance(B, PETSc.Mat)
    ai, aj, av = B.getValuesCSR()
    Mcomp = scipy.sparse.csr_matrix((av + 0j*av, aj, ai))

    # Initializing data
    evals_disp  = []
    start       = time.time()
    evec_all    = []
    print('Computing band structure...')
    
    #################################################################
    # Computing dispersion across the high-symmmetry points
    #################################################################
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




#%%
# ==============================================================================================
#
#                               COMPUTING THE BAND STRUCTURE
#
# ==============================================================================================

if __name__ == "__main__":

    start = time.time()
    ######################################################################
    #                      Domain  Inputs                                #
    ######################################################################
    #   a_len               Charecterstic unit cell length (primitive basis)
    #   r                   vector of points to fit spline to
    #   Nquads              Number of quadrants to repeat spline through (radially)
    #   offset              Offset angle for inclusion geometry
    #   iscut               Choose if inclusion is void or an added domain
    #   dvar                Design variable to probe 
    ######################################################################
    a_len       = 0.1
    dvar        = 1 / 2       
    r           = np.array([1, dvar, 0.2, 0.8, 0.3]) * a_len * 0.95
    r           = np.array([1, np.random.rand(1)[0], 0.3]) * a_len * 0.95
    # r           = np.array([[0.94912596, 0.94895456, 0.57853153, 0.26535732, 0.94616877, 0.94918783]])*a_len*.95; r= r.reshape(6,)
    # r           = np.array([[0.94929848 ,0.94984762, 0.69426451, 0.94035399 ,0.86640294, 0.95    ]])*a_len*.95; r= r.reshape(6,)
    dvec        = r
    offset      = 0 * np.pi
    design_vec  = np.concatenate((r / a_len, [offset]))
    Nquads      = 4

    # Set file name to save figures based on input vector
    name = 'dvec'
    for k in range(len(design_vec)):
        name += '_' + str(int(100 * np.round(design_vec[k], 2)))

    ######################################################################
    #                  Physical Params                                   #
    ######################################################################
    #   c                   Speed of sound in media
    #   rho                 Density
    ######################################################################
    multiple_material = False
    if multiple_material:
        # Solid inclusion
        c           = [1500,5100]
        rho         = [1e3,7e3]
    else:
        # Void inclusion
        c           = [30]
        rho         = [1.2]

    ######################################################################
    #                       FEM Inputs                                 #
    ######################################################################
    #   refinement_level    Choose ratio of how dense mesh gets around refinement field 
    #   refinement_dist     Maximum distance of refinement field from refined edges
    #   isrefined           Choose whether or not to refine mesh around internal edges
    #   meshalg             Meshing algorithm for gmsh to use
    #   mesh_dens           Nominal mesh density
    #   da                  Nominal mesh size
    #   npi                 Number of points to loop through i-th cut of IBZ
    #   nvec                Number of eigenvectrs to solve for in each step
    #   fspace              Function space to use
    ######################################################################
    mesh_dens           = 10
    da                  = a_len / mesh_dens
    meshalg             = 7
    refinement_level    = 6
    refinement_dist     = a_len / 4
    np1                 = 20
    np2                 = 20
    np3                 = 20
    nvec                = 20
    fspace              = 'CG'


    ######################################################################
    #                  Generate a mesh with one inclusion                #
    ######################################################################
    meshalg                 = 6
    gmsh.model, xpt, ypt    = get_mesh_SquareSpline(a_len, da, r, Nquads, offset,meshalg,
                                                    refinement_level, refinement_dist,
                                                    isrefined=True, cut=True)

    #################################################################
    #            Import to dolfinx and save as xdmf                 #
    #################################################################
    mesh_comm = MPI.COMM_WORLD
    model_rank = 0
    mesh, ct, _ = model_to_mesh(gmsh.model, mesh_comm, model_rank, gdim=2)
  

    #################################################################
    #           Defining options for data saving                    #
    #################################################################
    savefigs  = False
    savedata  = False
    figpath   = 'figures//SE413_Proposal4//OptExample'
    datapath  = 'data/SE413_OptimizationData_nquad{Nquads:d}_nDvec_{len_r:d}_nomMesh_{nomMesh:d}_refinement_{refine:d}'.format(Nquads=Nquads, len_r=len(r), nomMesh=mesh_dens, refine=refinement_level)
    isExist = os.path.exists(figpath)
    if savefigs:
        if not isExist:
            os.makedirs(figpath)
    if savedata:
        if not os.path.exists(datapath):
            os.makedirs(figpath)
    plt.style.use('default')

    ##################################
    #   Solve the problem
    ###################################
    plotter = plotmesh(mesh, fspace, ct)
    plotter.show()
    # evals_disp, evec_all = solve_bands(np1, np2, np3, nvec, a_len, c, rho, fspace, mesh,ct)


    # Define the high symmetry points of the lattice
    P1 = [0, 0] # Γ
    P2 = [np.pi / a_len, 0] # X
    P3 = [np.pi / a_len, np.pi / a_len] # M
    P4 = [0, np.pi / a_len] # Y
    HSpts = [P1, P2, P3, P4, P1]


    # Define the number of eigenvectors to solve for each solution
    nvec = 20

    # Define number of eigensolutions desired
    nsol = len(HSpts) * 12

    # evals_disp, evec_all, mpc = solve_bands(
        # np1, np2, np3, nvec, a_len, c, rho, fspace, mesh, ct)

    evals_disp, evec_all, mpc, KX, KY = solve_bands_repo(
        HSpts=HSpts,
        nsol=nsol, 
        nvec=nvec, 
        a_len=a_len, 
        c=c, 
        rho=rho, 
        fspace=fspace, 
        mesh=mesh,
        ct=ct,
    )

#%%
if __name__ == "__main__":
    ################################################
    # Testing high-symmetry loop
    ################################################
    # Define the high symmetry points of the lattice
    P1 = [0, 0] # Γ 
    P2 = [np.pi / a_len, 0] # X
    P3 = [np.pi / a_len, np.pi / a_len] # M
    P4 = [0, np.pi / a_len] # Y
    HSpts = [P1, P2, P3, P4, P1]
        
    nsol = 61
    nvec_per_HS = int(round(nsol / len(HSpts)))

    kx = HSpts[0][0]
    ky = HSpts[0][0]
    KX = []
    KY = []
    KX.append(kx)
    KY.append(ky)
    for k in range(len(HSpts) - 1):
        slope = np.array(HSpts[k + 1]) - np.array(HSpts[k])
        for j in range(nvec_per_HS):
            kx = kx + slope[0] / nvec_per_HS  
            ky = ky + slope[1] / nvec_per_HS  
            KX.append(kx)
            KY.append(ky)
        print(slope)

#%%

if __name__ == '__main__':
    '''
    #////////////////////////////////////////////////////////////////////
    #
    #           SECTION 4: POST PROCESS THE DISPERSION SOLUTION
    #
    #////////////////////////////////////////////////////////////////////
    '''

    nK = np1 + np2 + np3
    #################################################################
    #               Identify the band gaps                          #
    #################################################################
    eigfqs = np.array(evals_disp)
    ef_vec = eigfqs.reshape((1, nvec * nK))
    evals_all = np.sort(ef_vec).T
    deval = np.diff(evals_all.T).T
    args =  np.flip(np.argsort(deval.T)).T
    lowlim = []
    uplim = []
    bgs = []

    # Finding the boundaries of the pass bands
    lowb = []
    uppb = []
    for k in range(nvec):
        lowb.append(np.min(eigfqs.T[k]))
        uppb.append(np.max(eigfqs.T[k]))

    # Finding the band gaps
    for k in range(nvec):
        LowerLim = np.max(eigfqs[:, k])
        if k < nvec - 1:
            UpperLim = np.min(eigfqs[:, k + 1])
        else:
            UpperLim = np.min(eigfqs[:, k])


        # Check if these limits fall in a pass band
        overlap = False
        for j in range(nvec):
            if LowerLim > lowb[j] and LowerLim < uppb[j]:
                overlap = True            
            if UpperLim > lowb[j] and UpperLim < uppb[j]:
                overlap = True
        
        if not overlap:
            print('isbg')
            lowlim.append(LowerLim)
            uplim.append(UpperLim)
            bgs.append(UpperLim - LowerLim)
            
    # Filter band gaps
    maxfq = np.max(eigfqs[:])
    isgap = [i for i,v in enumerate(bgs) if v > np.median(deval)] 
    gaps  = np.array(bgs)
    lower = np.array(lowlim)
    higher= np.array(uplim)
    gapwidths  = gaps[isgap]
    lowbounds  = lower[isgap]
    highbounds = higher[isgap]

    ###########################################################
    # Get spline fit from the mesh 
    ###########################################################
    node_interior = gmsh.model.mesh.getNodesForPhysicalGroup(1, 5)[1]
    x_int = node_interior[0::3]
    y_int = node_interior[1::3]
    x_int = np.concatenate([x_int, [x_int[0]]])
    y_int = np.concatenate([y_int, [y_int[0]]])
    # plt.plot(x_int,y_int,'-')
    xi = x_int - a_len / 2
    yi = y_int - a_len / 2


    #////////////////////////////////////////////////////////////////////
    #
    # SECTION 5: PLOT THE OUTPUT OF THE PROBLEM
    #
    #////////////////////////////////////////////////////////////////////


    # import patch as matplotlib.patches
    # import matplotlib.patches as patch
    from matplotlib.patches import Rectangle
    # plt.style.use('seaborn-v0_8-dark')
    # sns.set_theme(style="ticks", palette=None,)
    # design vector normalization for plotting

    # Define x-y coords for plotting design vector
    x  = np.array(xpt) - a_len / 2
    y  = np.array(ypt) - a_len / 2

    #################################################################
    #            FIG 1: Plotting the dispersion                     #
    #################################################################
    x1 = np.linspace(0, 1 - 1 / np1, np1)
    x2 = np.linspace(1, 2 - 1 / np1, np2)
    x3 = np.linspace(2, 2 + np.sqrt(2), np3)
    xx = np.concatenate((x1, x2, x3))
    # PLOT THE DISPERSION BANDS
    for n in range(nvec):
        ev = []
        for k in range(len(evals_disp)):
            ev.append(np.real(evals_disp[k][n]))
        if n == 0:
            plt.plot(xx, ev, 'b.-', markersize=3, label='Bands')
        else:
            plt.plot(xx, ev, 'b.-', markersize=3)
    plt.grid(color='gray', linestyle='-', linewidth=0.2)
    plt.xticks(np.array([0, 1, 2, 2 + np.sqrt(2)]), ['$\Gamma$', 'X', 'M', '$\Gamma$'], fontsize=18)
    plt.xlabel('Wave Vector', fontsize=18)
    plt.ylabel('$\omega$ [rad/s]', fontsize=18)
    plt.title('Dispersion Diagram', fontsize=18)
    plt.xlim((0, np.max(xx)))
    plt.ylim((0, np.max(maxfq)))
    currentAxis = plt.gca()
    for j in range(len(gapwidths)):
        lb = lowbounds[j]
        ub = highbounds[j]
        if j == 0:
            currentAxis.add_patch(Rectangle((np.min(xx), lb), np.max(xx), ub - lb, facecolor="k", ec='none', alpha=0.6, label='bandgap'))
        else:
            currentAxis.add_patch(Rectangle((np.min(xx), lb), np.max(xx), ub - lb, facecolor="k", ec='none', alpha=0.6))
    plt.legend()
    if savefigs:
        plt.savefig(figpath + '//BandGapDgm' + name + '.pdf' , bbox_inches='tight')
        plt.savefig('figures//jpeg//BandGapDgm' + name + '.jpeg' , bbox_inches='tight') 
    plt.show()
    ###########################################################


    ###########################################################
    # FIG 2: Plot dispersion with design vec and void tplgy   #
    ###########################################################
    fig = plt.figure(figsize=(10, 6))
    ax1 = fig.add_subplot(1, 2, 2)
    for n in range(nvec):
        ev = []
        for k in range(len(evals_disp)):
            ev.append(np.real(evals_disp[k][n]))
        if n == 0:
            ax1.plot(xx, ev, 'b.-', markersize=3, label='Bands')
        else:
            ax1.plot(xx, ev, 'b.-', markersize=3)
    ax1.grid(color='gray', linestyle='-', linewidth=0.2)
    ax1.set_xticks(np.array([0, 1, 2, 2 + np.sqrt(2)]), ['$\Gamma$', 'X', 'M', '$\Gamma$'], fontsize=18)
    ax1.set_xlabel('Wave Vector', fontsize=18)
    ax1.set_ylabel('$\omega$ [rad/s]', fontsize=18)
    ax1.set_title('Dispersion Diagram', fontsize=18)
    ax1.set_xlim((0, np.max(xx)))
    ax1.set_ylim((0, np.max(maxfq)))
    for j in range(len(gapwidths)):
        lb = lowbounds[j]
        ub = highbounds[j]
        if j == 0:
            ax1.add_patch(Rectangle((np.min(xx), lb), np.max(xx), ub - lb, facecolor="k", ec='none', alpha=0.6, label='bandgap'))
        else:
            ax1.add_patch(Rectangle((np.min(xx), lb), np.max(xx), ub - lb, facecolor="k", ec='none', alpha=0.6))
    ax1.legend()
    # =============================
    ax2 = fig.add_subplot(221)
    ax2.add_patch(Rectangle((-a_len / 2, -a_len / 2), a_len, a_len, facecolor="w", ec='k', alpha=1, label='bandgap'))
    ax2.plot(x[0:int(len(xpt) / Nquads)], y[0:int(len(xpt) / Nquads)], '.r')
    ax2.plot(x[int(len(xpt)/Nquads):-1], y[int(len(xpt) / Nquads):-1], '.', color='gray')
    for j in range(len(xpt)):
        # plt.plot([0,x[r]], [0,y[r]],'k')
        if j < int(len(x) / Nquads):
            ax2.plot([0, x[j]], [0, y[j]], 'k')
        else:
            ax2.plot([0, x[j]], [0, y[j]], '--', color='gray', linewidth=1)
    ax2.set_title('Design Vector', fontsize=18)
    ax2.set_xlabel('$x$ [m]', fontsize=18)
    ax2.set_ylabel('$y$ [m]', fontsize=18)
    ax2.set_aspect('equal', 'box')
    ax2.set_xlim((-a_len / 1.5, a_len / 1.5))
    ax2.set_ylim((-a_len / 1.5, a_len / 1.5))
    # =============================
    ax2 = fig.add_subplot(223)
    ax2.add_patch(Rectangle((-a_len / 2, -a_len / 2), a_len, a_len, facecolor="w", ec='w', alpha=0.2, label='bandgap'))
    ax2.plot(x, y, '.', color='gray')
    for j in range(len(xpt)):
        ax2.plot([0, x[j]], [0, y[j]], '--', color='gray', linewidth=1)
    ax2.plot(xi, yi, '-b')
    ax2.set_title('BSpline Cut', fontsize=18)
    ax2.set_xlabel('$x$ [m]', fontsize=18)
    ax2.set_ylabel('$y$ [m]', fontsize=18)
    ax2.set_aspect('equal', 'box')
    ax2.set_xlim((-a_len / 1.5, a_len / 1.5))
    ax2.set_ylim((-a_len / 1.5, a_len / 1.5))
    # =============================
    fig.tight_layout(pad=1.4)
    if savefigs:
        plt.savefig(figpath + '//BandGapDgm_withDesign_subpl' + name + '.pdf' , bbox_inches='tight')
        plt.savefig('figures//jpeg//BandGapDgm_withDesign_subpl' + name + '.jpeg' , bbox_inches='tight') 
    plt.show()
    ###########################################################


    ###########################################################
    # FIG 3: Plotting the dispersion and void overlayed      #
    ###########################################################
    plt.figure(figsize=(4 / 1.3, 6 / 1.3))
    for n in range(nvec):
        ev = []
        for k in range(len(evals_disp)):
            ev.append(np.real(evals_disp[k][n]))
        if n == 0:
            plt.plot(xx, ev, 'b-', markersize=3, label='Bands')
        else:
            plt.plot(xx, ev, 'b-', markersize=3)
    plt.grid(color='gray', linestyle='-', linewidth=0.2)
    plt.xticks(np.array([0, 1, 2, 2 + np.sqrt(2)]), ['$\Gamma$', 'X', 'M', '$\Gamma$'], fontsize=12)
    plt.xlabel('Wave Vector ', fontsize=12)
    plt.ylabel('$\omega$ [rad/s]', fontsize=12)
    plt.title('Dispersion Diagram', fontsize=12)
    plt.xlim((0, np.max(xx)))
    plt.ylim((0, np.max(maxfq)))
    currentAxis = plt.gca()
    for j in range(len(gapwidths)):
        lb = lowbounds[j]
        ub = highbounds[j]
        if j == 0:
            currentAxis.add_patch(Rectangle((np.min(xx), lb), np.max(xx), ub - lb, facecolor="k", ec='none', alpha=0.6, label='bandgap'))
        else:
            currentAxis.add_patch(Rectangle((np.min(xx), lb), np.max(xx), ub - lb, facecolor="k", ec='none', alpha=0.6))
    plt.legend()
    # plt.title( (' $ x  = $' + str(np.round(design_vec,2))  ), fontsize = 12)
    # =============================
    ax = plt.gca()
    ax2 = ax.inset_axes([0.5, 0.58, 0.5, 0.5])
    ax2.add_patch(Rectangle((-a_len / 2, -a_len / 2), a_len, a_len, facecolor="w", ec='k', alpha=0.75, label='bandgap'))
    ax2.plot(x[0:int(len(xpt) / Nquads)], y[0:int(len(xpt) / Nquads)], '.r')
    ax2.plot(x[int(len(xpt) / Nquads):-1], y[int(len(xpt) / Nquads):-1], '.', color='gray')
    for r in range(len(x)):
        if r < int(len(x) / Nquads):
            ax2.plot([0, x[r]], [0, y[r]], '-', color='black', linewidth=1)
        else:
            ax2.plot([0, x[r]], [0, y[r]], '--', color='gray', linewidth=1)
    ax2.set_aspect('equal', 'box')
    ax2.set_axis_off()
    # =============================
    ax2.plot(xi, yi, '-k')
    ax2.set_aspect('equal', 'box')
    ax2.set_axis_off()
    # plt.savefig('BandGapDgm_withDesign_inset.pdf', bbox_inches = 'tight') 
    if savefigs:
        plt.savefig(figpath + '//BandGapDgm_withDesign_inset_grid' + name + '.pdf' , bbox_inches='tight')
        plt.savefig('figures//jpeg//BandGapDgm_withDesign_inset_grid' + name + '.jpeg' , bbox_inches='tight') 
    plt.show()
    ###########################################################


    #%%

    # Section: Post-processes eigenvectors

    # os.mkdir('./data//')
    infile = np.array(evec_all)
    np.save('data//testFile', infile)
    testload = np.load('data//testFile.npy')
    plotter = pyvista.Plotter(shape=(5, 4) , window_size=(1000, 1000))
    ###########################################################
    # Post-processing the eigenvectors
    ###########################################################
    euse = 0
    for i in range(5):
        for j in range(4):
            plotter.subplot(i, j)
            et = testload[euse, :, 3]
            euse += 1
            vr = Function(V)
            vi = Function(V)
            vr.vector[:] = np.real(et)
            vi.vector[:] = np.imag(et)
            vr.x.scatter_forward()
            mpc.backsubstitution(vr.vector)
            vi.x.scatter_forward()
            mpc.backsubstitution(vi.vector)

            ###########################################################
            # Plotting eigenvectors with pyvista
            ###########################################################
            # mycmap = plt.cm.get_cmap('coolwarm', 10)
            mycmap = plt.cm.get_cmap('seismic', 10)
            u = Function(V)
            cells, types, x = plot.create_vtk_mesh(V)
            grid = pyvista.UnstructuredGrid(cells, types, x)
            grid.point_data["u"] = u.x.array
            u.vector.setArray(vr.vector[:] / np.max(vr.vector[:]) * np.sign(vr.vector[10]))
            edges = grid.extract_all_edges()
            warped = grid.warp_by_scalar("u", factor=0)
            plotter.add_mesh(warped, show_edges=False, show_scalar_bar=True, scalars="u", cmap=mycmap)
            # plotter.add_mesh(grid, style = 'wireframe', line_width = .5, color = 'black')
            plotter.view_xy()
            plotter.add_text(str(euse), position=[100, 80], color='cyan')
            plotter.camera.tight(padding=0.1)

    plotter.show()

    #%%

    import pandas as pd

    dinf = {"gapwidth": gapwidths, "upperLims": highbounds}
    # dinf = {dinf, "upperLims":highbounds}
    dframe = pd.DataFrame(dinf)
    dframe.to_csv('data//testcsv.csv')

    dinf = {"Evals": evals_disp}
    dframe = pd.DataFrame(dinf)


    #%%
    BG_normalized   = gapwidths / (0.5 * lowbounds + 0.5 * highbounds)
