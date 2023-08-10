# Test fenonics import
#%%
# ################################################## #
# Genearal imports                                   #
# ################################################## #
import numpy as np
import matplotlib.pyplot as plt
import time
import os
from mpi4py import MPI

# ################################################## #
# Imports for the meshing                            #
# ################################################## #
from dolfinx.io.gmshio  import model_to_mesh
import gmsh
import fenonics

# Meshing parameters
cut         = True
a_len       = .1
r           = np.array([1,.9,.3,.8,.6])*a_len*.75
offset      = 0*np.pi/4
design_vec  = np.concatenate( (r/a_len, [offset] ))
Nquads      = 5
da                  =   a_len/15
refinement_level    =   4
refinement_dist     =   a_len/10
meshalg                 = 6

# Make the mesh with Gmsh
gmsh.model, xpt, ypt    = fenonics.get_mesh_SquareSpline(
        a_len, da, r, Nquads, offset, meshalg,
        refinement_level, refinement_dist,
        isrefined = True,   cut = cut)

# Import to dolfinx               
mesh_comm = MPI.COMM_WORLD
model_rank = 0
mesh, ct, _ = model_to_mesh(gmsh.model, mesh_comm, model_rank, gdim=2)

# Plot the design vector and the produced mesh
plt = fenonics.PlotSpline(gmsh, r, Nquads, a_len, xpt, ypt)
plt.show()
fenonics.plotmesh(mesh,ct)

# Define material properties
if not cut:
    c           = [1500,5100]   # if solid inclusion (mutlple materail model)
    rho         = [1e3,7e3]     # if solid inclusion (mutlple materail model) 
else:
    c           = [30]          # if void inclusion  (if iscut)
    rho         = [1.2]         # if void inclusion  (if iscut)
    
# Define the high symmetry points of the lattice
G  = np.array([0,0])                    
X  = np.array([np.pi/a_len, 0])              
M  = np.array([np.pi/a_len, np.pi/a_len])    
Y  = np.array([0, np.pi/a_len] )   
Mp = np.array([-np.pi/a_len, np.pi/a_len])    
Xp = np.array([-np.pi/a_len, 0] )   
HSpts = [G, X, M,
            G, Y, M,
            G, Xp, Mp,
            G, Y, Mp,
            G]

# Define the number of solutiosn per wavevec and number of wavevecs to solve for
n_solutions  = 30
n_wavevector = len(HSpts)*10

# Solve the dispersion problem
evals_disp, evec_all, mpc, KX, KY = fenonics.solve_bands(HSpts  = HSpts, n_wavevector  = n_wavevector,  
                    n_solutions = n_solutions, a_len = a_len,
                        c = c,  rho = rho,  fspace = 'CG',  
                        mesh = mesh, ct = ct)

bgnrm, gapwidths, gaps, lowbounds, highbounds = fenonics.getbands(np.array(evals_disp))
HS_labels = ['$\Gamma$', 'X', 'M',
                '$\Gamma$', 'Y', 'M',
                '$\Gamma$', 'X*', 'M*',
                '$\Gamma$', 'Y*', 'M*',
                '$\Gamma$'
            ]

plt = fenonics.plotbands(np.array(evals_disp),figsize = (5,5), HSpts = HSpts, HS_labels = HS_labels, a_len = a_len,
                    KX = KX, KY = KY, inset = True)
plt.show()



# %%
