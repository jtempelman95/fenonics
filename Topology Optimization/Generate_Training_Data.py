

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

from FEM_Functions import *
from PostProcess import*
#%% function
""""
#////////////////////////////////////////////////////////////////////

   
   !!!  SECTION 3: COMPUTE BAND STRUCTURE OVER LHS  !!!


#////////////////////////////////////////////////////////////////////
"""

######################################################################
#                      Mesh Inputs                                   #
######################################################################
a_len       = .1
Nquads      = 4
c           = [1500,5100]   # if solid inclusion
rho         = [1e3,7e3]     # if solid inclusion 
c           = [30]          # if void inclusion
rho         = [1.2]         # if void inclusion
np1                     =  20
np2                     =   20
np3                     =   20
nvec                    =   20
######################################################################
#                      Mesh Inputs                                   #
######################################################################
da                  =   a_len/15
meshalg             =   4          
refinement_level    =   6        
refinement_dist     =   a_len/10     
######################################################################
#                       Sampler Inputs                               #
######################################################################
from scipy.stats import qmc
LHS_Seed    = 4
Nsamp       = int(1e6)
sampler     = qmc.LatinHypercube(d=6, seed= LHS_Seed)
sample      = sampler.random(n=Nsamp)

savefigs    = False
savedata    = True
overwrite   = False

LoopStart   = 0
Nloop       = 5000
sampuse = np.linspace(LoopStart,LoopStart+Nloop-1,Nloop)
for sidx in sampuse:
    sidx = int(sidx)
    dvar  = 0
    start = time.time()
    ######################################################################
    #                      Domain  Inputs                                #
    ######################################################################
    r           = sample[sidx,:]*a_len*.95
    offset      = 0*dvar*np.pi/4
    design_vec  = np.concatenate( (r/a_len, [offset] ))
    fspace      =   'CG'
    
    name = 'dvec'
    for k in range(len(design_vec)):    
        name += '_' + str(int(100*np.round(design_vec[k],2)))
    #################################################################
    #           Defining options for data saving                    #
    #################################################################
    figpath = '..//figures//SE413_Proposal3//example_for_inkscape2'
    datapath = ('..//data//TrainingData//SamplrSeed '  + str(LHS_Seed) +' SamplrDim '+  str(sample.shape[1])   +' SamplrNgen '+  str(Nsamp)   
                                                + '//Quads_' + str(Nquads) + ' Xdim_' 
                                                + str(len(r))    +  ' MshRs_'+ str(a_len/da)
                                                + ' rfnmt_' +  str(refinement_level))
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
    #            Import to dolfinx                                 #
    #################################################################
    mesh_comm = MPI.COMM_WORLD
    model_rank = 0
    try:
        mesh, ct, _ = model_to_mesh(gmsh.model, mesh_comm, model_rank, gdim=2)
    except:
        mesh, ct, _ = model_to_mesh(gmsh.model, mesh_comm, model_rank, gdim=2)
    ###########################################################
    # Solve the dispersion problem
    ###########################################################
    evals_disp, evec_all  = solve_bands(np1, np2, np3,nvec, a_len, c, rho, fspace, mesh,ct)
    bgnrm, gapwidths, gaps, lowbounds, highbounds = getbands(np.array(evals_disp))
    BGdata = gapwidths
    ###########################################################
    # Get spline fit from the mesh 
    ###########################################################
    node_interior = gmsh.model.mesh.getNodesForPhysicalGroup(1,5)[1]
    x_int = node_interior[0::3]
    y_int = node_interior[1::3]
    x_int = np.concatenate([x_int,[x_int[0]]])
    y_int = np.concatenate([y_int,[y_int[0]]])
    xi = x_int - a_len/2
    yi = y_int - a_len/2
    lxi = len(xi)
    lxp = len(xpt)
    xptnd = np.array(xpt)
    yptnd = np.array(ypt)
    lxp = len(xptnd)
    xsv = np.empty(xi.shape)
    SplineDat = np.hstack( (xi.reshape(lxi,1), yi.reshape(lxi,1) ))  
    SplinePtDat = np.hstack( (xptnd.reshape(lxp,1), yptnd.reshape(lxp,1) ))  
    disp_info  = np.array(evals_disp)
    
    ############################
    # Save data
    ############################
    ngaps = len(gapwidths)
    if ngaps == 0:
        BGdata = np.zeros(4)
    else:
        BGdata = np.hstack((gapwidths.reshape(ngaps,1),lowbounds.reshape(ngaps,1),
                            highbounds.reshape(ngaps,1),bgnrm.reshape(ngaps,1)))
        
    ###########################################################
    # Save the data
    ###########################################################
    np.savetxt((datapath+'//Dvecdata//'     +str(sidx)+'.csv'),     design_vec, delimiter=",")
    np.savetxt((datapath+'//BGdata//'       +str(sidx)+'.csv'),     BGdata, delimiter=",")
    np.savetxt((datapath+'//dispersiondata//'+str(sidx)+'.csv'),    disp_info, delimiter=",")
    np.savetxt((datapath+'//Splinecurves//'  +str(sidx)+'.csv'),    SplineDat, delimiter=",")
    np.savetxt((datapath+'//Splinepts//'    +str(sidx)+'.csv'),     SplinePtDat, delimiter=",")
    gmsh.write((datapath+'//meshdata//'     +str(sidx)+'.msh'))
    # Test save the mesh
    
    print('***************************')
    print(f'Solution Number: {sidx}')
    print('***************************')   
del mesh 
    