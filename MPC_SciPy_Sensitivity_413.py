

"""
    
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                
                DISPERSION COMPUTATION WITH FENICSX
                
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                
                
    Josh Tempelman
    Universrity of Illinois
    jrt7@illinois.edu
    
    Originated      MARCH 10, 2023
    Last Modified   MARCH 24, 2023
    
    ! ! Do not distribute ! !
    
    
    About:
        This program formulates a square unit cell mesh with an imbedded
        geometry. The dolfinx-fenicsx routine and dolfinx-mpc solve the
        resulintg eigenvalue problems over the G-M-X-G boundaries of the
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
# ------------------
# Dependencies
# ------------------
import gmsh
import sys
import numpy as np
import matplotlib.pyplot as plt
import time

# ################################################## #
# Imports for the meshing                            #
# ################################################## #
import matplotlib.pyplot as plt
from mpi4py import MPI
from dolfinx.io import gmshio
from mpi4py import MPI
import pyvista
pyvista.start_xvfb()
# from MakeSqaureMeshFcn import get_mesh_SquareSpline
from MeshFunctions import get_mesh_SquareSpline,get_mesh_SquareMultiSpline
from dolfinx.plot import create_vtk_mesh
from dolfinx.io import XDMFFile
from dolfinx.io.gmshio import model_to_mesh


# ################################################## #
# Imports fr the eigenvlaues problem                 #
# ################################################## #
from mpi4py import MPI
from dolfinx.fem import Function, FunctionSpace, VectorFunctionSpace
from dolfinx.mesh import create_unit_square 
from dolfinx.mesh import create_rectangle
from dolfinx.fem import form
from dolfinx.fem.petsc import assemble_matrix
from dolfinx import plot
from dolfinx.io import gmshio
from mpi4py import MPI
from dolfinx import fem
from ufl import TrialFunction, TestFunction, grad, dx, dot, nabla_div, Identity, nabla_grad, inner, sym
from petsc4py.PETSc import ScalarType
from slepc4py import SLEPc
import ufl
from petsc4py import PETSc
from typing import List
import scipy.sparse
from scipy.sparse.linalg import eigsh
from scipy import sparse

# ################################################## #
# Imports fr the the MPC constraing                  #
# ################################################## #
import dolfinx
import dolfinx_mpc


#%% function

#////////////////////////////////////////////////////////////////////
#
# SECTION 1: SET PARAMETERS AND CREATE A MESH
#
#////////////////////////////////////////////////////////////////////
showplt     = [0,1,0]   # select which plots to show during loop
BGall       = []        # initlaize var to store bandgaps
LBall       = []        # || lower bounds of BG
HBall       = []        # || upper bounds of BG
EVall       = []        # || All eigenvalues
DGall       = []        # || All eigenvlaues seperaetd by band
SXall       = []        # || Xcoord of Spline curves generated by gmsh
SYall       = []        # || Ycoord of Spline curves generated by gmsh
DXall       = []        # || Xcoord of design vector
DYall       = []        # || Ycoord of design vector
savefigs    = False     # Decidie to save figures generated during analysis



figpath = 'figures//SE413_Proposal3//sensitivity'
isExist = os.path.exists(figpath)
if not isExist:
    os.makedirs(figpath)
    
# Master outter loop: Loop over design variable 'dvar'
for dvar in np.linspace(0,1,10):
    
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
    a_len       = .1
    r           = np.array([np.random.rand(),np.random.rand(),np.random.rand(),np.random.rand()])*a_len*.95
    r           = np.array([1,.3,dvar])*a_len*.95
    offset      = dvar*np.pi/4
    design_vec  = np.concatenate( (r/a_len, [offset] ))
    Nquads      = 4

    # Set file name to save figures
    name = 'dvec'
    for k in range(len(design_vec)):    
        name += '_' + str(int(100*np.round(design_vec[k],2)))
        
    # Nominal paramters (physcial)
    c           = [1500,5100]   # if solid inclusion
    rho         = [1e3,7e3]     # if solid inclusion 
    # c           = [30]          # if void inclusion
    # rho         = [1.2]         # if void inclusion

    ######################################################################
    #                      Mesh Inputs                                   #
    ######################################################################
    #   refinement_level    Choose ratio of how dense mesh gets around refinement field 
    #   refinement_dist     Maximum distance of refinement field from refined edges
    #   isrefined           Choose whether or not to refine mesh around internal edges
    #   meshalg             Meshing algorithm for gmsh to use
    #   da                  Nominal Mesh Density
    ######################################################################
    da                  =   a_len/30   
    meshalg             =   7           
    refinement_level    =   3           
    refinement_dist     =   a_len/4     

    ######################################################################
    #                        Solver inputs                               #
    ######################################################################
    # npi       Number of points to loop through i-th cut of IBZ
    # nvec      Number of eigenvectrs to solve for in each step
    # fspace    Function space to use
    ######################################################################
    np1  = 20
    np2  = 20
    np3  = 20
    nvec = 20

    ######################################################################
    #                  Generate a mesh with one inclsuion                #
    ######################################################################
    meshalg                 = 6
    gmsh.model, xpt, ypt    = get_mesh_SquareSpline(a_len ,da,r,Nquads,offset,meshalg,
                                                    refinement_level,refinement_dist,
                                                    isrefined = False, cut = False)

    #################################################################
    #            Import to dolfinx and save as xdmf                 #
    #################################################################
    mesh_comm = MPI.COMM_WORLD
    model_rank = 0
    mesh, ct, _ = model_to_mesh(gmsh.model, mesh_comm, model_rank, gdim=2)
    with XDMFFile(MPI.COMM_WORLD, "mt.xdmf", "w") as xdmf:
        xdmf.write_mesh(mesh)
        xdmf.write_meshtags(ct)
    t1 =  round(time.time()-start,3)
    print('Generated Mesh in ' + str(t1)  + ' Seconds')      
            
    #################################################################
    #              Plot the mesh                                    #
    #################################################################
    V = FunctionSpace(mesh,('CG',1))
    v = Function(V)
    plotter = pyvista.Plotter(window_size=[800,800])
    grid = pyvista.UnstructuredGrid(*create_vtk_mesh(mesh, mesh.topology.dim))
    num_local_cells = mesh.topology.index_map(mesh.topology.dim).size_local
    grid.cell_data["Marker"] = 1-ct.values[ct.indices<num_local_cells]
    grid.set_active_scalars("Marker")
    actor = plotter.add_mesh(grid, show_edges=True, line_width= 1, edge_color= 'w')
    plotter.set_background('black', top='black')
    plotter.view_xy()
    plotter.camera.tight(padding=0.1)
    plotter.add_title('CELLS:'+str(ct.values.shape[0])+ '  | NODES:'+str(v.vector[:].shape[0]))
    if savefigs:
        plotter.screenshot(figpath+'//Mesh' + name + '.jpeg',window_size=[1400,1400])
    plotter.show()


    #////////////////////////////////////////////////////////////////////
    #
    # SECTION 2: DEFINE HELPER FUNCTIONS TO SOLVE PROBLEM
    #
    #////////////////////////////////////////////////////////////////////
    
    #################################################################
    #       Function for matrix convesion                          #
    #################################################################
    def petsc_to_numpy(A):
        sz = A.getSize()
        A_ = np.zeros(sz, dtype=PETSc.ScalarType)
        for i in range(sz[0]):
            row = A.getRow(i)
            A_[i, row[0]] = row[1]
        return A_

    def complex_formatter(x):
        return "{0:-10.3e}{1:+10.3e}j".format(np.real(x), np.imag(x))

    #################################################################
    #       Function to displacy matrix components                  #
    #################################################################
    def print_matrix(A):
        print(
            np.array2string(
                A,
                max_line_width=np.inf,
                separator=" ",
                formatter={"complexfloat": complex_formatter}
            )
        )
        

    #################################################################
    #       Function for solving eigenproblem with SLEPc            #
    #################################################################
    # Function for computing eigenvalues
    def get_EPS(A, B, nvec):
        EPS = SLEPc.EPS()
        EPS.create(comm=MPI.COMM_WORLD)
        EPS.setOperators(A, B)
        EPS.setProblemType(SLEPc.EPS.ProblemType.GNHEP)
        # set the number of eigenvalues requested
        EPS.setDimensions(nev=nvec)
        # Set solver
        EPS.setType(SLEPc.EPS.Type.KRYLOVSCHUR)
        # set eigenvalues of interest
        EPS.setWhichEigenpairs(SLEPc.EPS.Which.TARGET_MAGNITUDE)
        EPS.setTarget(0)  # sorting
        # set tolerance and max iterations
        EPS.setTolerances(tol=1e-5, max_it=12)
        # Set up shift-and-invert
        # Only work if 'whichEigenpairs' is 'TARGET_XX'
        ST = EPS.getST()
        ST.setType(SLEPc.ST.Type.SINVERT)
        ST.setShift(10)
        EPS.setST(ST)
        # parse command line options
        EPS.setFromOptions()
        return EPS
    
    #################################################################
    #       Function for solving eigenproblem with scipy            #
    #################################################################
    def scipysolve(A,B, nval):
        K = petsc_to_numpy(A)
        M = petsc_to_numpy(B)
        K = sparse.csr_array(K)
        M = sparse.csr_array(M)
        eval, evec = eigsh(K, k = nval, M=M, sigma = 1.0)
        return eval, evec

    #################################################################
    #   Function for solving complex eigenproblem with scipy        #
    #################################################################
    def scipysolve_complex(A_re, A_im, B, nval):
        A_re_np = petsc_to_numpy(A_re)
        A_im_np = petsc_to_numpy(A_im)
        Kcomp    = A_re_np+1j*A_im_np
        eval, evec = eigsh(Kcomp, k = 24, M=Mcomp, sigma = 1.0)
        return eval, evec
        
    #################################################################
    #    Function to Assemble and solve system using Scipy          #
    #################################################################
    def solvesys(kx,ky,Mcomp,mpc,bcs,nvec):
        K = fem.Constant(mesh, PETSc.ScalarType((kx,ky)))
        a_form_re = E**2*(inner(grad(u_tr), grad(u_test)) + u_tr*u_test*(kx**2+ky**2))*dx
        a_form_im = E**2*(u_tr*inner(grad(u_test),K) - u_test*inner(grad(u_tr),K))*dx
        a_re = form(a_form_re)
        a_im = form(a_form_im)
        diagval_A = 1e8
        A_re = dolfinx_mpc.assemble_matrix(a_re, mpc, bcs=bcs, diagval=diagval_A)
        A_im = dolfinx_mpc.assemble_matrix(a_im, mpc, bcs=bcs, diagval=diagval_A)
        #################################################################
        #                Solving with scipy                             #
        #################################################################
        A_re.assemble()
        assert isinstance(A_re, PETSc.Mat)
        ai, aj, av = A_re.getValuesCSR()
        A_im.assemble()
        assert isinstance(A_im, PETSc.Mat)
        _,_, av_im = A_im.getValuesCSR()
        
        Kcomp = scipy.sparse.csr_matrix((av+1j*av_im, aj, ai))
        eval, evec= eigsh(Kcomp, k = nvec, M=Mcomp, sigma = 1.0)
        return eval, evec 

    #################################################################
    #       Function for multi point constraint                     #
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
                out_x = x.copy() # was ist x.copy()?
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
        mpc =dolfinx_mpc.MultiPointConstraint(functionspace)
        
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



    #////////////////////////////////////////////////////////////////////
    #
    # SECTION 3: SOLVE THE DISPERSION PROBLEM ON THE MESH
    #
    #////////////////////////////////////////////////////////////////////

    #################################################################
    #       Assign material properties to the domain                #
    #################################################################
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
        E.x.array[disk1_cells]      = np.full_like(disk1_cells,  c[0], dtype=ScalarType)
        E.x.array[disk2_cells]      = np.full_like(disk2_cells,  c[1], dtype=ScalarType)

    else:
        Rho = rho[0]
        E   = c[0]

    #################################################################
    #     Assign function space and constarint to mesh              #
    #################################################################
    V = FunctionSpace(mesh,('CG',1))
    mpc, bcs = dirichlet_and_periodic_bcs(mesh, V, ["periodic", "periodic"]) 
    u_tr    = TrialFunction(V)
    u_test  = TestFunction(V) 
    m_form = Rho*dot(u_tr, u_test)*dx



    #################################################################
    #        Define Mass Mat  outside of  IBZ loop                  #
    #################################################################
    m = form(m_form)
    diagval_B = 1e-2
    B = dolfinx_mpc.assemble_matrix(m, mpc, bcs=bcs, diagval=diagval_B)
    M_np = petsc_to_numpy(B)
    B.assemble()
    assert isinstance(B, PETSc.Mat)
    ai, aj, av = B.getValuesCSR()
    Mcomp = scipy.sparse.csr_matrix((av+0j*av, aj, ai))


    ######################################################################################
    #       ---             LOOPING THROUGH THE IBZ              ---                     #
    ######################################################################################
    ky = 0
    tsolve = []
    evals_disp =[]
    maxk = np.pi/a_len
    start=time.time()
    evec_all = []
    print('Computing Band Structure .... ')

    #################################################################
    #            Computing K to M                                   #
    #################################################################
    print('Computing Gamma to  X')
    for kx in np.linspace(0.01,maxk,np1):
        K = fem.Constant(mesh, PETSc.ScalarType((kx,ky)))
        eval, evec = solvesys(kx,ky,Mcomp,mpc,bcs,nvec)
        eval[np.isclose(eval,0)] == 0
        eigfrq_sp_cmp = np.real(eval)**.5
        eigfrq_sp_cmp = np.sort(eigfrq_sp_cmp )
        evals_disp.append(eigfrq_sp_cmp )
        evec_all.append(evec)
        # print(np.round(time.time()-start,3))

    #################################################################
    #            Computing K to M                                   #
    #################################################################
    print('Computing X to  M')
    kx = maxk
    for ky in np.linspace(0.01,maxk,np2):
        K = fem.Constant(mesh, PETSc.ScalarType((kx,ky)))
        # start=time.time()
        eval, evec = solvesys(kx,ky,Mcomp,mpc,bcs,nvec)
        eval[np.isclose(eval,0)] == 0
        eigfrq_sp_cmp = np.real(eval)**.5
        eigfrq_sp_cmp = np.sort(eigfrq_sp_cmp )
        evals_disp.append(eigfrq_sp_cmp )
        evec_all.append(evec)
        # print(np.round(time.time()-start,3))


    #################################################################
    #            Computing M To Gamma                               #
    #################################################################
    print('Computing M to Gamma')
    for kx in np.linspace(maxk,0.01,np3):
        ky = kx
        K = fem.Constant(mesh, PETSc.ScalarType((kx,ky)))
        # start=time.time()
        eval, evec = solvesys(kx,ky,Mcomp,mpc,bcs,nvec)
        eval[np.isclose(eval,0)] == 0
        eigfrq_sp_cmp = np.real(eval)**.5
        eigfrq_sp_cmp = np.sort(eigfrq_sp_cmp)
        evals_disp.append(eigfrq_sp_cmp )
        evec_all.append(evec)
        # print(np.round(time.time()-start,3))
    t2= round(time.time()-start,3)
    print('Time to compute Dispersion '+ str(t2))

    print('***************************')
    print('Total Time....'  + str(t1+t2))
    print('N_dof....'  + str(ct.values.shape[0]))
    print('N_vectors....'  + str(nvec))
    print('N_wavenumbers....'  + str(np1+np1+np3))
    print('Ttoal Eigenproblems....'  + str(nvec*(np1+np1+np3)))
    print('Avg T_eigsol....'  + str(round(t2/((np1+np1+np3)),3)) )




    #////////////////////////////////////////////////////////////////////
    #
    # SECTION 4: POST PROCESS THE DISPERSION SOLUTION
    #
    #////////////////////////////////////////////////////////////////////

    nK = np1+np2+np3
    #################################################################
    #               Idenitfy the band gaps                          #
    #################################################################
    eigfqs = np.array(evals_disp)
    ef_vec = eigfqs.reshape((1,nvec*nK))
    evals_all = np.sort(ef_vec).T
    deval = np.diff(evals_all.T).T
    args =  np.flip(np.argsort(deval.T)).T
    lowlim  = []
    uplim   = []
    bgs     = []

    # Finding the boundaries of the pass bands
    lowb      = []
    uppb      = []
    for k in range(nvec):
        lowb.append(np.min(eigfqs.T[k]))
        uppb.append(np.max(eigfqs.T[k]))

    # Finding the band gaps
    for k in range(nvec):
        
        LowerLim =  np.max(eigfqs[:,k])
        if k < nvec-1:
            UpperLim =  np.min(eigfqs[:,k+1])
        else:
            UpperLim =  np.min(eigfqs[:,k])

        # Check if these limits fall in a pass band
        overlap = False
        for j in range(nvec):
            if LowerLim > lowb[j] and LowerLim <uppb[j]:
                overlap = True            
            if UpperLim > lowb[j] and UpperLim <uppb[j]:
                overlap = True
        
        if overlap == False:
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
    # Append the current band gaps to the all band gaps
    ###########################################################
    BGall.append(gapwidths)
    LBall.append(lowbounds)
    HBall.append(highbounds)
    # EVall.append(evals_all.T)
    DGall.append(evals_disp)
    ###########################################################
    # Get spline fit from the mesh 
    ###########################################################
    node_interior = gmsh.model.mesh.getNodesForPhysicalGroup(1,5)[1]
    x_int = node_interior[0::3]
    y_int = node_interior[1::3]
    x_int = np.concatenate([x_int,[x_int[0]]])
    y_int = np.concatenate([y_int,[y_int[0]]])
    # plt.plot(x_int,y_int,'-')
    xi = x_int - a_len/2
    yi = y_int - a_len/2
    
    SXall.append(xi)
    SYall.append(yi)
    DXall.append(xpt)
    DYall.append(ypt)


    #////////////////////////////////////////////////////////////////////
    #
    # SECTION 5: PLOT THE OUTPUT OF THE PROBLEM
    #
    #////////////////////////////////////////////////////////////////////
    # import patch as matplotlib.patches
    # import matplotlib.patches as patch
    from matplotlib.patches import Rectangle
    plt.style.use('default')
    # plt.style.use('seaborn-v0_8-dark')
    # sns.set_theme(style="ticks", palette=None,)
    # design vector normalization for plotting


    # Define for visualization
    x  = np.array(xpt) - a_len/2
    y  = np.array(ypt) - a_len/2
    x1 = np.linspace(0,1-1/np1,np1)
    x2 = np.linspace(1,2-1/np1,np2)
    x3 = np.linspace(2,2+np.sqrt(2),np3)
    xx = np.concatenate((x1,x2,x3))

    if showplt[0]:
        #################################################################
        #            FIG 1: Plotting the dispersion                     #
        #################################################################
        for n in range(nvec):
            ev = []
            for k in range(len(evals_disp)):
                ev.append(np.real(evals_disp[k][n]))
            if n == 0:
                plt.plot( xx,(ev),'b.-',markersize = 3, label = 'Bands')
            else:
                plt.plot( xx,(ev),'b.-',markersize = 3)
        plt.grid(color='gray', linestyle='-', linewidth=0.2)
        plt.xticks(np.array([0,1,2,2+np.sqrt(2)]), ['$\Gamma$', 'X', 'M', '$\Gamma$'],fontsize=18)
        plt.xlabel(r'Wave Vector ',fontsize=18)
        plt.ylabel('$\omega$ [rad/s]',fontsize=18)
        plt.title('Dispersion Diagram',fontsize = 18)
        plt.xlim((0,np.max(xx)))
        plt.ylim((0,np.max(maxfq)))
        currentAxis = plt.gca()
        for j in range(len(gapwidths)):
            lb = lowbounds[j]
            ub = highbounds[j]
            if j == 0:
                currentAxis.add_patch( Rectangle((np.min(xx),lb), np.max(xx), ub-lb,  facecolor="k" ,ec='none', alpha =.6,label='bangap'))
            else:
                currentAxis.add_patch( Rectangle((np.min(xx),lb), np.max(xx), ub-lb,  facecolor="k" ,ec='none', alpha =.6))
        plt.legend()
        if savefigs:
            plt.savefig(figpath+'//BandGapDgm' + name + '.pdf' , bbox_inches = 'tight')
            plt.savefig('figures//jpeg//BandGapDgm' + name + '.jpeg' , bbox_inches = 'tight') 
        plt.show()
        ###########################################################



    if showplt[1]:
        ###########################################################
        # FIG 2: Plot dispersion with design vec and void tplgy  #
        ###########################################################
        fig = plt.figure(figsize=(10,6) )
        ax1 = fig.add_subplot(1,2,2)
        for n in range(nvec):
            ev = []
            for k in range(len(evals_disp)):
                ev.append(np.real(evals_disp[k][n]))
            if n == 0:
                ax1.plot( xx,(ev),'b.-',markersize = 3, label = 'Bands')
            else:
                ax1.plot( xx,(ev),'b.-',markersize = 3)
        ax1.grid(color='gray', linestyle='-', linewidth=0.2)
        ax1.set_xticks(np.array([0,1,2,2+np.sqrt(2)]), ['$\Gamma$', 'X', 'M', '$\Gamma$'],fontsize=18)
        ax1.set_xlabel(r'Wave Vector ',fontsize=18)
        ax1.set_ylabel('$\omega$ [rad/s]',fontsize=18)
        ax1.set_title('Dispersion Diagram',fontsize = 18)
        ax1.set_xlim((0,np.max(xx)))
        ax1.set_ylim((0,np.max(maxfq)))
        for j in range(len(gapwidths)):
            lb = lowbounds[j]
            ub = highbounds[j]
            if j == 0:
                ax1.add_patch( Rectangle((np.min(xx),lb), np.max(xx), ub-lb,  facecolor="k" ,ec='none', alpha =.6,label='bangap'))
            else:
                ax1.add_patch( Rectangle((np.min(xx),lb), np.max(xx), ub-lb,  facecolor="k" ,ec='none', alpha =.6))
        ax1.legend()
        # =============================
        ax2 = fig.add_subplot(221)
        ax2.add_patch( Rectangle( (-a_len/2,-a_len/2),a_len,a_len, facecolor="w" ,ec='k', alpha =1,label='bangap'))
        ax2.plot(x[0:int(len(xpt)/Nquads)], y[0:int(len(xpt)/Nquads)], ' .r')
        ax2.plot(x[int(len(xpt)/Nquads):-1], y[int(len(xpt)/Nquads):-1], '.', color = 'gray')
        for j in range(len(xpt)):
            # plt.plot([0,x[r]], [0,y[r]],'k')
            if j<int(len(x)/Nquads):
                ax2.plot([0,x[j]], [0,y[j]],'k')
            else:
                ax2.plot([0,x[j]], [0,y[j]],'--', color = 'gray', linewidth = 1)
        ax2.set_title('Design Vector',fontsize = 18)
        ax2.set_xlabel('$x$ [m]',fontsize = 18)
        ax2.set_ylabel('$y$ [m]',fontsize = 18)
        ax2.set_aspect('equal', 'box')
        ax2.set_xlim((-a_len/1.5,a_len/1.5))
        ax2.set_ylim((-a_len/1.5,a_len/1.5))
        # =============================
        ax2 = fig.add_subplot(223)
        ax2.add_patch( Rectangle( (-a_len/2,-a_len/2),a_len,a_len, facecolor="w" ,ec='w', alpha =.2,label='bangap'))
        ax2.plot(x, y, '.', color = 'gray')
        for j in range(len(xpt)):
            ax2.plot([0,x[j]], [0,y[j]],'--', color = 'gray', linewidth = 1)
        ax2.plot(xi, yi, '-b')
        ax2.set_title('BSpline Cut',fontsize = 18)
        ax2.set_xlabel('$x$ [m]',fontsize = 18)
        ax2.set_ylabel('$y$ [m]',fontsize = 18)
        ax2.set_aspect('equal', 'box')
        ax2.set_xlim((-a_len/1.5,a_len/1.5))
        ax2.set_ylim((-a_len/1.5,a_len/1.5))
        # =============================
        fig.tight_layout(pad=1.4)
        if savefigs:
            plt.savefig(figpath+'//BandGapDgm_withDesign_subpl' +name + '.pdf' , bbox_inches = 'tight')
            plt.savefig('figures//jpeg//BandGapDgm_withDesign_subpl' +name + '.jpeg' , bbox_inches = 'tight') 
        plt.show()
        ###########################################################

    if showplt[2]:
        ###########################################################
        # FIG 3: Plotting the dispersion and void overlayed      #
        ###########################################################
        plt.figure(figsize=(4/1.3,6/1.3))
        for n in range(nvec):
            ev = []
            for k in range(len(evals_disp)):
                ev.append(np.real(evals_disp[k][n]))
            if n == 0:
                plt.plot( xx,(ev),'b-',markersize = 3, label = 'Bands')
            else:
                plt.plot( xx,(ev),'b-',markersize = 3)
        plt.grid(color='gray', linestyle='-', linewidth=0.2)
        plt.xticks(np.array([0,1,2,2+np.sqrt(2)]), ['$\Gamma$', 'X', 'M', '$\Gamma$'],fontsize=12)
        plt.xlabel(r'Wave Vector ',fontsize=12)
        plt.ylabel('$\omega$ [rad/s]',fontsize=12)
        plt.title('Dispersion Diagram',fontsize = 12)
        plt.xlim((0,np.max(xx)))
        plt.ylim((0,np.max(maxfq)))
        currentAxis = plt.gca()
        for j in range(len(gapwidths)):
            lb = lowbounds[j]
            ub = highbounds[j]
            if j == 0:
                currentAxis.add_patch( Rectangle((np.min(xx),lb), np.max(xx), ub-lb,  facecolor="k" ,ec='none', alpha =.6,label='bangap'))
            else:
                currentAxis.add_patch( Rectangle((np.min(xx),lb), np.max(xx), ub-lb,  facecolor="k" ,ec='none', alpha =.6))
        plt.legend()
        # plt.title( (' $ x  = $' + str(np.round(design_vec,2))  ), fontsize = 12)
        # =============================
        ax = plt.gca()
        ax2 = ax.inset_axes([0.5, .58, 0.5, 0.5])
        ax2.add_patch( Rectangle( (-a_len/2,-a_len/2),a_len,a_len, facecolor="w" ,ec='k', alpha =.75,label='bangap'))
        ax2.plot(x[0:int(len(xpt)/Nquads)], y[0:int(len(xpt)/Nquads)], ' .r')
        ax2.plot(x[int(len(xpt)/Nquads):-1], y[int(len(xpt)/Nquads):-1], '.', color = 'gray')
        for r in range(len(x)):
            if r<int(len(x)/Nquads):
                ax2.plot([0,x[r]], [0,y[r]],'-', color = 'black', linewidth = 1)
            else:
                ax2.plot([0,x[r]], [0,y[r]],'--', color = 'gray', linewidth = 1)
        ax2.set_aspect('equal', 'box')
        ax2.set_axis_off()
        # =============================
        ax2.plot(xi, yi, '-k')
        ax2.set_aspect('equal', 'box')
        ax2.set_axis_off()
        # plt.savefig('BandGapDgm_withDesign_inset.pdf', bbox_inches = 'tight') 
        if savefigs:
            plt.savefig(figpath+'//BandGapDgm_withDesign_inset' +name + '.pdf' , bbox_inches = 'tight')
            plt.savefig('figures//jpeg//BandGapDgm_withDesign_inset' +name + '.jpeg' , bbox_inches = 'tight') 
        plt.show()
        ###########################################################


#%%
plt.style.use('default')
objf = []
L1 = len(SXall)
for k in range(len(BGall)):   
   #  plt.scatter( k*np.ones(len(LBall[k])) ,LBall[k], color='b')
   #  plt.scatter( k*np.ones(len(HBall[k])), HBall[k], color='r')
   objf.append(1/np.max(BGall[k]))
   plt.scatter( k,1/np.max(BGall[k]), color='r')
plt.plot(objf,'k-')
    # plt.scatter( k*np.ones(len(EVall[k].T)) ,EVall[k], color='k')
# plt.xlabel(r'$\theta$')
plt.xlabel('Design Vector Iterations',fontsize =18)
plt.ylabel('1/(Max Band Gap)',fontsize =18)

plt.savefig(figpath+'//BG_Analysis.pdf')
plt.show()

plt.figure(figsize=(6,6))
cmap = plt.cm.tab20
for i in range(len(BGall)):
    evd = DGall[i]   
    for n in range(2):
            ev = []
            for k in range(len(evd)):
                ev.append(np.real(evd[k][n]))
            # plt.plot( xx,(ev),'-',markersize = 3, label = 'Bands', color = cmap(i/5))
            plt.plot( xx,(ev),'-',markersize = 3, label = 'Bands', color = cmap(i/L1))
#%%
plt.style.use('dark_background')

plt.figure(figsize=(20,6))
# loop through subplots
for i in range(10):
    # add a new subplot iteratively
    
    n = i  
    #############################
    ax = plt.subplot(1,10, i + 1)
    cmap = plt.cm.tab20
    #############################
    
    evd = DGall[n]   
    for m in range(10):
            ev = []
            for k in range(len(evd)):
                ev.append(np.real(evd[k][m]))
            plt.plot( xx,(ev),'-',markersize = 3, label = 'Bands', color = cmap(i/10))
    plt.grid(color='gray', linestyle='-', linewidth=0.2)
    if i == 0:
        plt.ylabel('$\omega$ [rad/s]',fontsize=12)
    else:
        ax.set_yticklabels("")
        
    ax.plot(x[0:int(len(xpt)/Nquads)], y[0:int(len(xpt)/Nquads)], ' .r')
    ax.plot(x[int(len(xpt)/Nquads):-1], y[int(len(xpt)/Nquads):-1], '.', color = 'gray')
    plt.xticks(np.array([0,1,2,2+np.sqrt(2)]), ['$\Gamma$', 'X', 'M', '$\Gamma$'],fontsize=12)
    plt.xlabel(r'Wave Vector ',fontsize=12)
    currentAxis = ax
    for j in range(len(BGall[i])):
        lb = LBall[i][j]
        ub = HBall[i][j]
        if j == 0:
            currentAxis.add_patch( Rectangle((np.min(xx),lb), np.max(xx), ub-lb,  facecolor="w" ,ec='none', alpha =.2,label='bangap'))
        else:
            currentAxis.add_patch( Rectangle((np.min(xx),lb), np.max(xx), ub-lb,  facecolor="w" ,ec='none', alpha =.2))    
    plt.xlim((0,np.max(xx)))
    plt.xlim((0,np.max(xx)))
    plt.ylim((0,5500))
    
    ax = plt.gca()
    ax2 = ax.inset_axes([0.15, .8, 0.65, 0.65])
    
    for r in range(len(DXall[i])):
        ax2.plot([0,DXall[i][r]-a_len/2], [0,DYall[i][r]-a_len/2],'--', color = 'gray', linewidth = 1)
        ax2.set_aspect('equal', 'box')
        
    ax2.add_patch( Rectangle( (-a_len/2,-a_len/2),a_len,a_len, facecolor="w" ,ec='k', alpha =.35,label='bangap'))
    ax2.plot(np.array(DXall[n])-a_len/2, np.array(DYall[n])-a_len/2, '.',color = [1, .5, 0])
    ax2.plot(SXall[n],SYall[n], color = [.4, .4, 1],linewidth = 2)
    #ax2.set_xlabel("")
    ax2.set_xticklabels("")
    ax2.set_yticklabels("")
    ax2.set_aspect('equal', 'box')
    ax2.axis('off')
# plt.show()
#if 1:
# plt.savefig('figures//413//pdf//BandGapDgm_withDesign_inset_grid_dvec2' +name + '.pdf' , bbox_inches = 'tight')
# plt.savefig('figures//413//jpeg//BandGapDgm_withDesign_inset_grid_dvec2' +name + '.jpeg' , bbox_inches = 'tight') 
plt.show()
###########################################################


# %%