

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
import os

# ################################################## #
# Imports fr the the MPC constraing                  #
# ################################################## #
import dolfinx
import dolfinx_mpc
# form HelperFucntions import petsc_to_numpy, get_EPS, scipysolve, scipysolve_complex, solvesys,  
""""
#////////////////////////////////////////////////////////////////////

   
   !!!  SECTION 1: DEFINE SUPPORT FUNCTIONS FOR FEM !!!


#////////////////////////////////////////////////////////////////////
"""

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
    EPS.setTolerances(tol=1e-5, max_it=12)
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


#%% function
""""
#////////////////////////////////////////////////////////////////////

   
   !!!  SECTION 3: COMPUTE BAND STRUCTURE OVER LHS  !!!


#////////////////////////////////////////////////////////////////////
"""

#################################################################
#           Initializing Variables to loop over                    #
#################################################################
showplt     = [0,0,0]   # select which plots to show during loop
BGall       = []        # initlaize var to store bandgaps
LBall       = []        # || lower bounds of BG
HBall       = []        # || upper bounds of BG
EVall       = []        # || All eigenvalues
DGall       = []        # || All eigenvlaues seperaetd by band
SXall       = []        # || Xcoord of Spline curves generated by gmsh
SYall       = []        # || Ycoord of Spline curves generated by gmsh
DXall       = []        # || Xcoord of design vector
DYall       = []        # || Ycoord of design vector
BGnrm       = []        # || Normalized band-gap widths
Tinfo       = []        # Time taken at each iteration
# ===================================================================


######################################################################
#                      Mesh Inputs                                   #
######################################################################
a_len       = .1
Nquads      = 4
c           = [1500,5100]   # if solid inclusion
rho         = [1e3,7e3]     # if solid inclusion 
c           = [30]          # if void inclusion
rho         = [1.2]         # if void inclusion

######################################################################
#                      Mesh Inputs                                   #
######################################################################
da                  =   a_len/25
meshalg             =   6          
refinement_level    =   6        
refinement_dist     =   a_len/6     

######################################################################
#                       Solver inputs                               #
######################################################################
np1  = 20
np2  = 20
np3  = 20
nvec = 20



######################################################################
#                       Sampler Inputs                               #
######################################################################
from scipy.stats import qmc
LHS_Seed    = 4
Nsamp       = int(1e6)
sampler     = qmc.LatinHypercube(d=6, seed= LHS_Seed)
sample      = sampler.random(n=Nsamp)
# plt.plot(sample[:,0], sample[:,1],'.')
# plt.grid()


savefigs = False
savedata = True
overwrite = False

plt.style.use('default')
# plt.rcParams["font.family"] = "Times New Roman"

LoopStart = 0
Nloop = 2000
sampuse = np.linspace(LoopStart,LoopStart+Nloop-1,Nloop)
for sidx in sampuse:
    sidx = int(sidx)
    dvar  = 0
    start = time.time()
    ######################################################################
    #                      Domain  Inputs                                #
    ######################################################################
    r           = sample[sidx,:]*a_len*.95
    offset      = dvar*np.pi/4
    design_vec  = np.concatenate( (r/a_len, [offset] ))
    
    # Set file name to save figures
    name = 'dvec'
    for k in range(len(design_vec)):    
        name += '_' + str(int(100*np.round(design_vec[k],2)))
        
        
    #################################################################
    #           Defining options for data saving                    #
    #################################################################
    figpath = 'figures//SE413_Proposal3//example_for_inkscape2'
    datapath = ('data//TrainingData//SamplrSeed '  + str(LHS_Seed) +' SamplrDim '+  str(sample.shape[1])   +' SamplrNgen '+  str(Nsamp)   
                                                + '//Quads_' + str(Nquads) + ' Xdim_' 
                                                + str(len(r))    +  ' MshRs_'+ str(a_len/da)
                                                + ' rfnmt_' +  str(refinement_level)
                )
    isExist = os.path.exists(figpath)
    if savefigs:
        if not isExist:
            os.makedirs(figpath)
    if savedata:
        if not os.path.exists(datapath):
            os.makedirs(datapath+'//dispersiondata')
            os.makedirs(datapath+'//meshdata')
            os.makedirs(datapath+'//Splinecurves')
            os.makedirs(datapath+'//BGdata')
            os.makedirs(datapath+'//Dvecdata')
            os.makedirs(datapath+'//Splinepts')
    
    # Skip iteration if the file was already generated
    if os.path.isfile(datapath+'//dispersiondata//'+str(sidx)+'.csv'):
        if not overwrite:
            print('Skipping iteration '+str(sidx)+' because it already exists')
            continue
        
        
    ######################################################################
    #                  Generate a mesh with one inclsuion                #
    ######################################################################
    meshalg                 = 6
    gmsh.model, xpt, ypt    = get_mesh_SquareSpline(a_len ,da,r,Nquads,offset,meshalg,
                                                    refinement_level,refinement_dist,
                                                    isrefined = True, cut = True)

    #################################################################
    #            Import to dolfinx and save as xdmf                 #
    #################################################################
    
    mesh_comm = MPI.COMM_WORLD
    model_rank = 0
    try:
        mesh, ct, _ = model_to_mesh(gmsh.model, mesh_comm, model_rank, gdim=2)
    except:
        mesh, ct, _ = model_to_mesh(gmsh.model, mesh_comm, model_rank, gdim=2)
        
    with XDMFFile(MPI.COMM_WORLD, "mt.xdmf", "w") as xdmf:
        xdmf.write_mesh(mesh)
        xdmf.write_meshtags(ct)
    t1 =  round(time.time()-start,3)
    print('Generated Mesh in ' + str(t1)  + ' Seconds')      

    #################################################################
    #       Assign material properties to the domain                #
    #################################################################
    if len(rho) > 1:
        # E.g., if more than one physical group assigned.
        # Assign material propoerties to each physical group.
        Q               = FunctionSpace(mesh, ("DG", 0))
        E               = Function(Q)
        Rho             = Function(Q)    
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
    
    t2= round(time.time()-start,3)
    print('-------- TIME INFO -----------')
    print('Sol number:'+str(sidx))
    print('Sol Time....'  + str(t1+t2))
    Tinfo.append(t1+t2)
    print('Total Time....'  + str(np.mean(np.array(Tinfo))))
    print('-------- TIME INFO -----------')




    #////////////////////////////////////////////////////////////////////
    #
    # SECTION 4: POST PROCESS THE DISPERSION SOLUTION
    #
    #////////////////////////////////////////////////////////////////////

    nK = np1+np2+np3
    #################################################################
    #               Idenitfy the band gaps                          #
    #################################################################
    eigfqs      = np.array(evals_disp)
    ef_vec      = eigfqs.reshape((1,nvec*nK))
    evals_all   = np.sort(ef_vec).T
    deval   =   np.diff(evals_all.T).T
    args    =   np.flip(np.argsort(deval.T)).T
    lowlim  = []
    uplim   = []
    bgs     = []

    # =======================
    # Finding the boundaries of the pass bands
    # =======================
    lowb      = []; uppb      = []
    for k in range(nvec):
        lowb.append(np.min(eigfqs.T[k]))
        uppb.append(np.max(eigfqs.T[k]))
    for k in range(nvec):
        LowerLim =  np.max(eigfqs[:,k])
        if k < nvec-1:
            UpperLim =  np.min(eigfqs[:,k+1])
        else:
            UpperLim =  np.min(eigfqs[:,k])
        # =======================
        # Check if these limits fall in a pass band
        # =======================
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
    maxfq           = np.max(eigfqs[:])
    isgap           = [i for i,v in enumerate(bgs) if v > np.median(deval)] 
    gaps            = np.array(bgs)
    lower           = np.array(lowlim)
    higher          = np.array(uplim)
    gapwidths       = gaps[isgap]
    lowbounds       = lower[isgap]
    highbounds      = higher[isgap]
    BG_normalized   = gapwidths/(.5*lowbounds  + .5*highbounds)

    ###########################################################
    # Append the current band gaps to the all band gaps
    ###########################################################
    BGall.append(gapwidths)
    if not np.any(BG_normalized):
        mxbg=0
    else:
        mxbg = np.sum(BG_normalized)
    BGnrm.append(mxbg)
    LBall.append(lowbounds)
    HBall.append(highbounds)
    EVall.append(evals_all.T)
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
    # Define for visualization
    x  = np.array(xpt) - a_len/2
    y  = np.array(ypt) - a_len/2
    x1 = np.linspace(0,1-1/np1,np1)
    x2 = np.linspace(1,2-1/np1,np2)
    x3 = np.linspace(2,2+np.sqrt(2),np3)
    xx = np.concatenate((x1,x2,x3))
    
    
    ############################
    # Save data
    ############################
    ngaps = len(gapwidths)
    if ngaps == 0:
        BGdata = np.zeros(4)
    else:
        BGdata = np.hstack((gapwidths.reshape(ngaps,1),lowbounds.reshape(ngaps,1),
                            highbounds.reshape(ngaps,1),BG_normalized.reshape(ngaps,1)))

    lxi = len(xi)
    lxp = len(xpt)
    xptnd = np.array(xpt)
    yptnd = np.array(ypt)
    lxp = len(xptnd)
    xsv = np.empty(xi.shape)
    SplineDat = np.hstack( (xi.reshape(lxi,1), yi.reshape(lxi,1) ))  
    SplinePtDat = np.hstack( (xptnd.reshape(lxp,1), yptnd.reshape(lxp,1) ))  
    disp_info  = np.array(evals_disp)


    np.savetxt((datapath+'//Dvecdata//'     +str(sidx)+'.csv'),     design_vec, delimiter=",")
    np.savetxt((datapath+'//BGdata//'       +str(sidx)+'.csv'),     BGdata, delimiter=",")
    np.savetxt((datapath+'//dispersiondata//'+str(sidx)+'.csv'),    disp_info, delimiter=",")
    np.savetxt((datapath+'//Splinecurves//'  +str(sidx)+'.csv'),    SplineDat, delimiter=",")
    np.savetxt((datapath+'//Splinepts//'    +str(sidx)+'.csv'),     SplinePtDat, delimiter=",")
    gmsh.write((datapath+'//meshdata//'     +str(sidx)+'.msh'))
    # Test save the mesh

    del mesh, mpc 
    