"""
    
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                
                    POST PROCESSING SIMULATION RESULTS
                
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                
                
    Josh Tempelman
    Universrity of Illinois
    jrt7@illinois.edu
    
    Originated      MARCH 10, 2023
    Last Modified   MARCH 24, 2023
    
    ! ! Do not distribute ! !
    
    
    About:
        This program loads in dispersion data from FEM
        and fits surrogate models to the band diagram objective functions
    
        
"""

#%%
import numpy as np
import matplotlib.pyplot as plt
import os

#%%

##########################################
# Design parameters to study
##########################################

MshRs       = float(9)
Xdim        = int(3)
refinement_level = 2
SamplrDim   = 3
Nsamp       = int(1e6)
LHS_Seed    = 4
Nquads      = 4


##########################################
# Load some data
##########################################
datapath = ('data//TrainingData//SamplrSeed '  + str(LHS_Seed) +' SamplrDim '+  str(SamplrDim)   +' SamplrNgen '+  str(Nsamp)   
                                                + '//Quads_' + str(Nquads) + ' Xdim_' 
                                                + str(Xdim)    +  ' MshRs_'+ str(MshRs)
                                                + ' rfnmt_' +  str(refinement_level)
            )

os.listdir(datapath)


# TrainingData//SamplrSeed 4 SamplrDim 3 SamplrNgen 1000000//Quads_4 Xdim_3 MshRs_9.0 rfnmt_2'
# TrainingData//SamplrSeed 4 SamplrDim 3 SamplrNgen 1000000//Quads_4 Xdim_3 MshRs_9.0 rfnmt_2
