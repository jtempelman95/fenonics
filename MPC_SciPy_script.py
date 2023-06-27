
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
from MeshFunctions import get_mesh_SquareSpline
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


figpath = 'figures//SE413_Proposal3//ConfirmationFigs//4quadsvg'
isExist = os.path.exists(figpath)
if not isExist:
    os.makedirs(figpath)
    
    
######################################################################
#                  Generate a mesh                                  #
######################################################################
a = 1
da = a/20
r =  a*np.array([1.1,.6, .2, .4])*.75
# r =  a*np.array([1,1,1])*.85
# r =  a*np.array([.4,1,.4])*.85
# r = np.random.rand(5,1)
dvar = 1/2       
a_len = a
r           = np.array([.4,1,1,dvar,.2,1,1])*a_len*.95
r           = np.array([1,dvar,.2,.5])*a*.95
Nquads      = 4

offset      = 0*np.pi/7
Nquads      = 4
meshalg     = 6
refinement_level = 2
refinement_dist = a/15
gmsh.model, xpt, ypt = get_mesh_SquareSpline(a,da,r,Nquads,offset,meshalg,
                                   refinement_level,refinement_dist,
                                   isrefined = True, cut = True)

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
V = FunctionSpace(mesh,('CG',1))
v = Function(V)
plotter = pyvista.Plotter()
grid = pyvista.UnstructuredGrid(*create_vtk_mesh(mesh, mesh.topology.dim))
num_local_cells = mesh.topology.index_map(mesh.topology.dim).size_local
grid.cell_data["Marker"] = 1-ct.values[ct.indices<num_local_cells]
grid.set_active_scalars("Marker")
actor = plotter.add_mesh(grid, show_edges=True, line_width= .1, edge_color= 'w',cmap='viridis')
# plotter.add_mesh(grid, style="wireframe", color = 'w')
plotter.view_xy()
# plotter.screenshot('Test.png')
plotter.camera.tight(padding=0.1)
plotter.add_title('CELLS:'+str(ct.values.shape[0])+ '  | NODES:'+str(v.vector[:].shape[0]))
plotter.screenshot(figpath+'//Mesh.jpeg',window_size=[2000,2000])
plotter.show()

#%% ////////////////////////////////////////////////////////////
#from https://fenicsproject.discourse.group/t/periodic-boundary-conditons-for-mixed-functionspace-with-dolfinx-mpc/9964

#################################################################
#       HELPERS FOR MTRICE                        #
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
#       Function for solving eigenproblem                       #
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

    bcs = []
    pbc_directions = []
    pbc_slave_tags = []
    pbc_is_slave = []
    pbc_is_master = []
    pbc_meshtags = []
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




#%% ////////////////////////////////////////////////////////////

#################################################################
#     Assign function space and constarint to mesh              #
#################################################################
V = FunctionSpace(mesh,('CG',1))
mpc, bcs = dirichlet_and_periodic_bcs(mesh, V, ["periodic", "periodic"]) 

#################################################################
#            Defining the variational form                     #
#################################################################
u, v = TrialFunction(V), TestFunction(V)
a =  inner(grad(u), grad(v)) * dx
b =  inner(u,v) * dx
diagval_A       = 1e4
diagval_B       = 1e-2
mass_form       = fem.form(b)
stiffness_form  = fem.form(a)

#################################################################
#            Assemble and solve the problem                     #
#################################################################
start = time.time()
A = dolfinx_mpc.assemble_matrix(stiffness_form, mpc, bcs=bcs, diagval=diagval_A)
B = dolfinx_mpc.assemble_matrix(mass_form, mpc, bcs=bcs, diagval=diagval_B)
end = time.time()
print('TIME TO ASSEMBLE:'+ str(end - start) + 'SECONDS')
start = time.time()
EPS = get_EPS(A,B,40)
EPS.solve()
end = time.time()
print('TIME TO SOLVE:'+ str(end - start) + 'SECONDS')
start = time.time()

#################################################################
#    Collect Eigenvectors and Eigenvalues  (SLEPc)              #
#################################################################
start = time.time()
eigval = [EPS.getEigenvalue(i) for i in range(EPS.getConverged())]
eigvec_r = list()
eigvec_i = list()
V = mpc.function_space
for i in range(EPS.getConverged()):
    vr = fem.Function(V)
    vi = fem.Function(V)
    EPS.getEigenvector(i, vr.vector, vi.vector)
    eigvec_r.append(vr)
    eigvec_i.append(vi)    # Sort by increasing real parts
idx = np.argsort(np.real(np.array(eigval)), axis=0)
eigval   = [eigval[i] for i in idx]
eigfrq   = [eigval[i]**.5 for i in idx]
eigvec_r = [eigvec_r[i] for i in idx]
eigvec_i = [eigvec_i[i] for i in idx]

#################################################################
#         Backsubstitue the overwitten DoFs  (SLEPc)            #
#################################################################
for i in range(len(eigval)):
    eigvec_r[i].x.scatter_forward()
    mpc.backsubstitution(eigvec_r[i].vector)
    eigvec_i[i].x.scatter_forward()
    mpc.backsubstitution(eigvec_i[i].vector)
end = time.time()
print('TIME TO POSTPROCESSES:'+ str(end - start) + 'SECONDS')


#################################################################
#       Solving in scipy                                        #
#################################################################'
K = petsc_to_numpy(A)
M = petsc_to_numpy(B)
end = time.time()
print('CONVERTING TO NUMPY:'+ str(end - start) + 'SECONDS')
start = time.time()
eval_sp, evec_sp  = scipysolve(A,B,40)
# eval_scipy, evec_scipy = eigsh(K, k = 40, M=M, sigma = 1.0)
end = time.time()
print('SCIPY SOLUTION:'+ str(end - start) + 'SECONDS')
eval_sp[np.isclose(eval_sp,0)] = 0
# csr_mat = scipy.sparse.rand(len(eigvec_i[1].vect
# or[:]),len(eigvec_i[1].vector[:]), density=0.001, format='csr')
# petsc_mat = A.createAIJ(size=csr_mat.shape,
#             csr=(csr_mat.indptr, csr_mat.indices,
#             csr_mat.data))

#################################################################
#        Collect Eigenvectors and Eigenvalues   (scipy)         #
#################################################################
start = time.time()
eigvec_r_sp = list()
eigvec_i_sp = list()
V = mpc.function_space
for i in range(len(eval_sp)):
    vr = fem.Function(V)
    vi = fem.Function(V)
    vr.vector[:] = np.real(evec_sp[:,i])
    vi.vector[:] = np.imag(evec_sp[:,i])
    eigvec_r_sp.append(vr)
    eigvec_i_sp.append(vi)    # Sort by increasing real parts
idx = np.argsort(np.real(np.array(eval_sp)), axis=0)
eigval_sp   = [eval_sp[i] for i in idx]
eigfrq_sp   = [eval_sp[i]**.5 for i in idx]
eigvec_r_sp = [eigvec_r_sp[i] for i in idx]
eigvec_i_sp = [eigvec_i_sp[i] for i in idx]

# #################################################################
#         Backsubstitue the overwitten DoFs  (scipy)              #
# #################################################################
for i in range(len(eval_sp)):
    eigvec_r_sp[i].x.scatter_forward()
    mpc.backsubstitution(eigvec_r_sp[i].vector)
    eigvec_i_sp[i].x.scatter_forward()
    mpc.backsubstitution(eigvec_i_sp[i].vector)
end = time.time()
print('TIME TO POSTPROCESSES:'+ str(end - start) + 'SECONDS')
#%%

#################################################################
#             Confirm scipy and pyvista match                   #
#################################################################
plt.style.use('dark_background')
plt.plot(eigfrq_sp,'o', label='ARPACK (Lanzcos)')
plt.plot(eigfrq,'r+',label = 'SLEPc (Krylov-Schur)')
plt.legend()

#%% ////////////////////////////////////////////////////////////
#################################################################
#             Plot some outputs with pyvista                    #
#################################################################
u = Function(V)
u.vector.setArray(eigvec_r[1].vector[:])
cells, types, x = plot.create_vtk_mesh(V)
grid = pyvista.UnstructuredGrid(cells, types, x)
grid.point_data["u"] = u.x.array
plotter = pyvista.Plotter(shape=(2, 5), window_size=(1300,350))
R  = 20;
mycmap = plt.cm.get_cmap('Set2')
mycmap = plt.cm.get_cmap('seismic', 30)

for j in range(2):
    R = 20
    for i in range(5):
        lmbda = eigval[R]
        u = Function(V)
        mx1 = np.max([np.abs(eigvec_r[R].vector[:])])
        mx2 = np.max([np.abs(eigvec_r_sp[R].vector[:])])
        if j ==0:
            u.vector.setArray(eigvec_r[R].vector[:]/mx1)
        else:
            u.vector.setArray(eigvec_r_sp[R].vector[:]/mx2)
        cells, types, x = plot.create_vtk_mesh(V)
        grid = pyvista.UnstructuredGrid(cells, types, x)
        edges = grid.extract_all_edges()
        grid.point_data["u"] = (u.x.array)
        warped = grid.warp_by_scalar("u", factor=0)
        ###########################
        plotter.subplot(j,i)
        plotter.add_mesh(warped, show_edges=False, show_scalar_bar=True, scalars="u",cmap=mycmap)
        #plotter.add_mesh(edges, line_width=0.0, color='gray',  opacity="geom_r")
        # plotter.add_mesh(grid, line_width=.05, style="wireframe", color = 'black')
        plotter.view_xy()
        # plotter.add_title('frq=' + str(np.round(2*np.pi*np.real(eigfrq[R]),1)) + ' Hz', font_size =  8)
        plotter.set_background('white', top='white')
        
       # if j == 0:
       #    plotter.add_title('SLEPc', font_size =  42)
       # else:
        #    plotter.add_title('ARPACK', font_size =  42)
        # plotter.set_background('white', top='white')
        R+=1
plotter.screenshot(figpath+'//ModeShapes.png', window_size=(2400,1200))
plotter.show()


#%% ////////////////////////////////////////////////////////////


from ufl import TrialFunction, TestFunction, grad, dx, dot, nabla_div, Identity, nabla_grad, inner, sym, TrialFunction, TestFunction, FiniteElement, MixedElement, split
# Define the wave vector
kx = np.pi; ky = .4
K = fem.Constant(mesh, PETSc.ScalarType((kx,ky)))


######################################################################################
#                       Solving the bloch-periodic problem
######################################################################################
from ufl import (grad, inner)
u_tr    = TrialFunction(V)
u_test  = TestFunction(V) 

E = 1
rho = 1
#################################################################
#                Variational form                               #
#################################################################
a_form_re = E*(inner(grad(u_tr), grad(u_test)) + u_tr*u_test*(kx**2+ky**2))*dx
a_form_im = E*(u_tr*inner(grad(u_test),K) - u_test*inner(grad(u_tr),K))*dx
m_form = rho*dot(u_tr, u_test)*dx
a_re = form(a_form_re)
a_im = form(a_form_im)
m = form(m_form)
A_re = dolfinx_mpc.assemble_matrix(a_re, mpc, bcs=bcs, diagval=diagval_A)
A_im = dolfinx_mpc.assemble_matrix(a_im, mpc, bcs=bcs, diagval=diagval_A)
B = dolfinx_mpc.assemble_matrix(m, mpc, bcs=bcs, diagval=diagval_B)

#################################################################
#                Solving with scipy                             #
#################################################################
A_re_np = petsc_to_numpy(A_re)
A_im_np = petsc_to_numpy(A_im)
M_np = petsc_to_numpy(B)

Kcomp    = A_re_np+1j*A_im_np
Mcomp   = M_np + 0j*M_np 
Kcomp = sparse.csr_array(Kcomp)
Mcomp = sparse.csr_array(Mcomp)
eval, evec = eigsh(Kcomp, k = 10, M=Mcomp, sigma = 1.0)
eval[np.isclose(eval,0)] == 0
eigfrq_sp_cmp = eval**.5
#%%
plt.plot(eigfrq,'ro')
plt.plot(eigfrq_sp,'g+')
plt.plot(eigfrq_sp_cmp,'bx')

#%%




######################################################################################
#                      Looping through k space
######################################################################################

# mass form stays constant
u_tr    = TrialFunction(V)
u_test  = TestFunction(V) 
m_form = rho*dot(u_tr, u_test)*dx
m = form(m_form)
B = dolfinx_mpc.assemble_matrix(m, mpc, bcs=bcs, diagval=diagval_B)
M_np = petsc_to_numpy(B)
Mcomp   = M_np + 0j*M_np 
Mcomp = sparse.csr_array(Mcomp)

ky = 0

E = 1
rho = 1
evals_disp =[]
for kx in np.linspace(0.01,np.pi,20):
    K = fem.Constant(mesh, PETSc.ScalarType((kx,ky)))
    #################################################################
    #                Variational form                               #
    #################################################################
    a_form_re = E*(inner(grad(u_tr), grad(u_test)) + u_tr*u_test*(kx**2+ky**2))*dx
    a_form_im = E*(u_tr*inner(grad(u_test),K) - u_test*inner(grad(u_tr),K))*dx
    a_re = form(a_form_re)
    a_im = form(a_form_im)
    A_re = dolfinx_mpc.assemble_matrix(a_re, mpc, bcs=bcs, diagval=diagval_A)
    A_im = dolfinx_mpc.assemble_matrix(a_im, mpc, bcs=bcs, diagval=diagval_A)

    #################################################################
    #                Solving with scipy                             #
    #################################################################
    A_re_np = petsc_to_numpy(A_re)
    A_im_np = petsc_to_numpy(A_im)

    Kcomp    = A_re_np+1j*A_im_np
    Kcomp = sparse.csr_array(Kcomp)
    eval, evec = eigsh(Kcomp, k = 24, M=Mcomp, sigma = 1.0)
    eval[np.isclose(eval,0)] == 0
    eigfrq_sp_cmp = np.real(eval)**.5
    eigfrq_sp_cmp = np.sort(eigfrq_sp_cmp )
    
    evals_disp.append(eigfrq_sp_cmp )
#%%

# PLOT THE DISPERSION BANDS
for n in range(10):
    ev = []
    for k in range(len(evals_disp)):
        ev.append(np.real(evals_disp[k][n]))
    plt.plot( np.linspace(0,np.pi,len(ev)),(ev),'c-')
    plt.plot(-np.linspace(0,np.pi,len(ev)),(ev),'c-')
    # plt.plot(-np.pi-np.linspace(0,np.pi,len(ev)),np.flip(ev),'b-')
    # plt.plot(np.pi+np.linspace(0,np.pi,len(ev)),np.flip(ev),'b-')

plt.grid(color='w', linestyle='-', linewidth=0.2)
plt.xticks(np.linspace(-1,1,5)*np.pi , ['$-\pi$', '$-\pi/2$','$0$','$\pi$', '$\pi/2$'])
plt.xlabel(r'Wave Vector $[\frac{\kappa_x a}{\pi}] $')
plt.ylabel('$\omega$ [rad/s]')
# %%
from matplotlib.patches import Rectangle

a_len = 1
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

plt.figure(figsize=(4/1.3,6/1.3))

x = np.array(xpt) - a_len/2
y = np.array(ypt) - a_len/2


fig = plt.figure()
ax2 = fig.add_subplot(111)

ax2.add_patch( Rectangle( (-a_len/2,-a_len/2),a_len,a_len, facecolor="w" ,ec=(1,.5,0), alpha =1,linewidth = 2 ,label='Unit Cell'))
ax2.plot(x[0:int(len(xpt)/Nquads)], y[0:int(len(xpt)/Nquads)], ' .r', label = 'Design Point')
ax2.plot(x[int(len(xpt)/Nquads):-1], y[int(len(xpt)/Nquads):-1], '.', color = 'gray', label= 'Copied Points')
for r in range(len(x)):
    if r<int(len(x)/Nquads):
        ax2.plot([0,x[r]], [0,y[r]],'-', color = 'black', linewidth = 1)
    else:
        ax2.plot([0,x[r]], [0,y[r]],'--', color = 'gray', linewidth = 1)
ax2.set_aspect('equal', 'box')
# ax2.set_axis_off()
# =============================
ax2.plot(xi, yi, '-b', label = 'Spline Inperpolation')
ax2.set_aspect('equal', 'box')
ax2.set_xlabel('$x$ Coord.')
ax2.set_ylabel('$y$ Coord.')
ax2.legend()
plt.savefig(figpath+'//UnitCellSpline.pdf')
plt.show()


# %%
########################################################
# CONFIRMING THE PERIODICITIY
########################################################

coords = V.tabulate_dof_coordinates()
xc = coords[:,0]
yc = coords[:,1]
bcl = np.isclose(xc, 0)
xbcl = xc[bcl]
ybcl = yc[bcl]
bcr = np.isclose(xc, 1)
xbcr = xc[bcr]
ybcr = yc[bcr]
bct = np.isclose(yc, 1)
xbct = xc[bct]
ybct = yc[bct]
bcb = np.isclose(yc, 0)
xbcb = xc[bcb]
ybcb = yc[bcb]
########################################################
# CONFIRMING THE PERIODICITIY
########################################################
plt.figure(figsize=(15,6))
f, (ax1, ax2) = plt.subplots(1, 2,figsize=(10,4))
ax1.plot(xc[::1],yc[::1],'.',color = 'gray',label='Interior Domain')
ax1.plot(xbcl,ybcl,'b.',label = 'LR Boundary Node')
ax1.plot(xbcr,ybcr,'b.')
ax1.plot(xbcb,ybcb,'r.',label = 'TB Boundary Nodes')
ax1.plot(xbct,ybct,'r.')
ax1.set_xlim([-.25,1.25])
ax1.set_ylim([-.25,1.25])
ax1.axis('equal')
ax1.set_title('Mesh')
ax1.set_xlabel('$x$ Coord.')
ax1.set_ylabel('$y$ Coord.')
ax1.legend()
# p.show()
argl = np.argsort(ybcl)
ybcl = np.sort(ybcl)
argr = np.argsort(ybcr)
ybcr = np.sort(ybcr)
argt = np.argsort(ybct)
xbct = np.sort(xbct)
argb = np.argsort(ybcb)
xbcb = np.sort(xbcb)
mx1 = np.min([len(ybcl), len(ybcr)])
mx2 = np.min([len(xbct), len(ybcb)])
ax2.plot(ybcl[0:mx1]-ybcr[0:mx1],'b',label = 'LR Boundary Node')
ax2.plot(xbct[0:mx2]-xbcb[0:mx2],'r--',label = 'TB Boundary Node')
ax2.set_title('Difference in boundary DoFs')
ax2.set_ylabel('Difference In Periodic Coordinate Values')
ax2.set_xlabel('Boundary Node Number')
ax2.legend()
# plt.savefig('BoudnaryDOFS.pdf')

plt.savefig(figpath+'//BoundaryDofs.pdf')
plt.show()
