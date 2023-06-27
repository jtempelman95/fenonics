
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

# ################################################## #
# Imports fr the the MPC constraing                  #
# ################################################## #
import dolfinx
import dolfinx_mpc


#%% function

######################################################################
#                  Generate a mesh                                  #
######################################################################
a = 1
da = a/50
# r =  a*np.array([1.1,.1,0,.1, 1.65,.1,0,.1])*.75
# r =  a*np.array([1.1,0,1.65,0,.95,0,])*.75
r =  a*np.array([1,1,1])*.85
# r = np.random.rand(5,1)
offset      = np.pi/7
Nquads      = 4
meshalg     = 6
refinement_level = 2
refinement_dist = a/15
gmsh.model = get_mesh_SquareSpline(a,da,r,Nquads,offset,meshalg,
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
plotter.screenshot('Mesh.jpeg',window_size=[2000,1400])
plotter.add_title('CELLS:'+str(ct.values.shape[0])+ '  | NODES:'+str(v.vector[:].shape[0]))
plotter.show()

#%% from https://fenicsproject.discourse.group/t/periodic-boundary-conditons-for-mixed-functionspace-with-dolfinx-mpc/9964



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

        mpc.create_periodic_constraint_topological(functionspace, pbc_meshtags[i],
                                                    pbc_slave_tags[i],
                                                    pbc_slave_to_master_map,
                                                    bcs)
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
        mpc.create_periodic_constraint_topological(functionspace, pbc_meshtags[1],
                                                    pbc_slave_tags[1],
                                                    pbc_slave_to_master_map,
                                                    bcs)
    mpc.finalize()
    return mpc, bcs




#%%

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
diagval_A = 1e4
diagval_B = 1e-2
mass_form = fem.form(b)
stiffness_form = fem.form(a)

#################################################################
#            Assemble and solve the problem                     #
#################################################################
start = time.time()
A = dolfinx_mpc.assemble_matrix(stiffness_form, mpc, bcs=bcs, diagval=diagval_A)
B = dolfinx_mpc.assemble_matrix(mass_form, mpc, bcs=bcs, diagval=diagval_B)
end = time.time()
print('TIME TO ASSEMBLE:'+ str(end - start) + 'SECONDS')
start = time.time()
EPS = get_EPS(A,B,120)
EPS.solve()
end = time.time()
print('TIME TO SOLVE:'+ str(end - start) + 'SECONDS')

#################################################################
#            Collect Eigenvectors and Eigenvalues               #
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
eigfrq   = [eigval[i] for i in idx]
eigvec_r = [eigvec_r[i] for i in idx]
eigvec_i = [eigvec_i[i] for i in idx]

#################################################################
#             Backsubstitue the overwitten DoFs                 #
#################################################################
for i in range(len(eigval)):
    eigvec_r[i].x.scatter_forward()
    mpc.backsubstitution(eigvec_r[i].vector)
    eigvec_i[i].x.scatter_forward()
    mpc.backsubstitution(eigvec_i[i].vector)

end = time.time()
print('TIME TO POSTPROCESSES:'+ str(end - start) + 'SECONDS')
#%%
#################################################################
#             Plot some outputs with pyvista                    #
#################################################################
u = Function(V)
u.vector.setArray(eigvec_r[1].vector[:])
cells, types, x = plot.create_vtk_mesh(V)
grid = pyvista.UnstructuredGrid(cells, types, x)
grid.point_data["u"] = u.x.array
plotter = pyvista.Plotter(shape=(3, 4), window_size=(1200,800))
R  = 40;
mycmap = plt.cm.get_cmap('Set2')
mycmap = plt.cm.get_cmap('seismic', 30)
# mycmap = plt.cm.get_cmap('coolwarm', 15)
# mycmap.set_gamma(.5)
# mycmap = plt.cm.get_cmap('Pastel2',20)

for j in range(3):
    for i in range(4):
        lmbda = eigval[R]
        u = Function(V)
        u.vector.setArray(eigvec_r[R].vector[:])
        cells, types, x = plot.create_vtk_mesh(V)
        grid = pyvista.UnstructuredGrid(cells, types, x)
        edges = grid.extract_all_edges()
        grid.point_data["u"] = (u.x.array)
        warped = grid.warp_by_scalar("u", factor=0)
        ###########################
        plotter.subplot(j,i)
        ###########################
        plotter.add_mesh(warped, show_edges=False, show_scalar_bar=True, scalars="u",cmap=mycmap)
        #plotter.add_mesh(edges, line_width=0.0, color='gray',  opacity="geom_r")
        # plotter.add_mesh(grid, style="wireframe")
        plotter.view_xy()
        plotter.add_title('frq=' + str(np.round(2*np.pi*np.real(eigfrq[R]),1)) + ' Hz', font_size =  8)
        # plotter.set_background('white', top='white')
        plotter.set_background('black', top='black')
        R+=1
plotter.screenshot('ModeShapes.png', window_size=(2400,1600))
plotter.show()


#%%
from ufl import TrialFunction, TestFunction, grad, dx, dot, nabla_div, Identity, nabla_grad, inner, sym, TrialFunction, TestFunction, FiniteElement, MixedElement, split

##############################################################
#           Attempt the mixed funciton space probem
##############################################################
from ufl import (grad, inner)
Vfem =  FiniteElement('Lagrange', mesh.ufl_cell(),1)
ME = fem.FunctionSpace(mesh, MixedElement([Vfem,Vfem]))
mpc, bcs = dirichlet_and_periodic_bcs(mesh, ME, ["periodic", "periodic"]) 


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
ax2.legend()
# plt.savefig('BoudnaryDOFS.pdf')

plt.show()