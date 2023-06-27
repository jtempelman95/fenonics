"""
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
             UNPACKING OPTIZED RESULTS
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    Josh Tempelman
    Universrity of Illinois
    jrt7@illinois.edu
    Originated      MAY 8, 2023
    Last Modified   MAY 9, 2023
    ! ! Do not distribute ! !
    About:
        This program loads in dispersion data from FEM
        and fits surrogate models to the band diagram objective functions.
        Optimization is performed over the surrogate model and then re-evalated
        by an FEM script (imported form FEM_Functions file).
"""
#%%
import matplotlib.pyplot as plt
# import seaborn as sb
from PostProcess    import *
from FEM_Functions  import *
from  Surrogate_Model_Optimization_Loop_multobj_ndim3 import *
savefigs = 1
if __name__ == '__main__':
    if not savefigs:
        plt.style.use('dark_background')
    else:
        plt.style.use('default')
    #################################################
    #                   PARAMETERS
    #################################################
    BG_op       = 1
    BG_op2      = 5
    #################################################
    MshRs       = float(25)
    Xdim        = 3
    rf_level    = 6
    if Xdim == 6:
        Nquads      = 4
    else:
        Nquads      =8
    SamplrDim   = Xdim
    Nsamp       = int(1e6)
    LHS_Seed    = 4
    path        = "..//data//TrainingData"
    figpath     = ('..//figures//Optimization//inspiron//Quads_'      +  str(Nquads)  + ' Xdim_'
                                                            +  str(Xdim)    +  ' MshRs_'     + str(MshRs)
                                                            + ' rfnmt_'     +  str(rf_level)
                                                            +'//BG_op'+str(BG_op)+'BG_op2'+str(BG_op2)+'weighted')
    isExist = os.path.exists(figpath)
    if not isExist: os.makedirs(figpath)
    #################################################
    #        Load in the data
    #################################################
    dvecs,splines,BGdata,Bands, Rads, datapath = load_data(MshRs,         Xdim,   rf_level,   Nquads,
                                                    SamplrDim,     Nsamp,  LHS_Seed,   path)
    # if not 'dvecs' in locals() and not 'Xdimcurrent' in locals():
    #     Xdimcurrent = Xdim
    #     dvecs,splines,BGdata,Bands, Rads, datapath = load_data(MshRs,         Xdim,   rf_level,   Nquads,
    #                                                 SamplrDim,     Nsamp,  LHS_Seed,   path)
    # elif Xdimcurrent != Xdim:
    #     dvecs,splines,BGdata,Bands, Rads, datapath = load_data(MshRs,         Xdim,   rf_level,   Nquads,
    #                                                 SamplrDim,     Nsamp,  LHS_Seed,   path)
    
    if savefigs:
        isExist = os.path.exists(figpath)
        if not isExist: os.makedirs(figpath)



    #################################################
    #           COMPUTE OBJECTIVE FUNCTION
    #################################################
    optvec      = np.zeros(len(dvecs))
    xvec        = np.zeros((len(dvecs),SamplrDim))
    for k in range(len(BGdata)):
        '''
        Computing some parameters from the band diagrams
        '''
        optvec[k]   = multi_objfun(Bands[k], BG_op = BG_op, BG_op2 = BG_op2)
        xvec[k,0]   = dvecs[k][0]
        xvec[k,1]   = dvecs[k][1]
        if len(xvec[0,:]) > 2:
            xvec[k,2]   = dvecs[k][2]
    def normalize(targ, targ_raw):
        return (targ - np.min(targ_raw)) /(np.max(targ_raw)-np.min(targ_raw))
    #################################################
    #        DEFINE FEATURES AND TARGETS
    #################################################
    features    = np.array(dvecs);
    targ_raw    = optvec
    targets     = normalize(targ_raw, targ_raw)
    len_feat    = features.shape[1] 
    
    
    #################################################
    #        Load in the enrichment data
    #################################################
    
    datapath_optimal = datapath[0:10] + '//OptimizedData' +datapath[22::]+  '//BGop_'+str(BG_op) +'BG_op2'+str(BG_op2)+'weighted'
    dvecs_enrich    = []
    BGdata_enrich   = []
    Bands_enrich    = []
    splines_enrich  = []
    f_enrich        = []
    dvecs_optimal   = []
    BGdata_optimal  = []
    Bands_optimal   = []
    splines_optimal = []
    f_optimal       = []
    for k in range(13):
        datapath_enrich = datapath[0:10] + '//EnrichementData' +datapath[22::]+  '//BGop_'+str(BG_op)+'BG_op2'+str(BG_op2)+'weighted' +'//enrich_loop' + str(k)
        print(os.path.exists(datapath_enrich))
        if os.path.exists(datapath_enrich):
            os.listdir(datapath_enrich)
            dirs    = os.listdir(datapath_enrich)
            files   = os.listdir((datapath_enrich+'//'+dirs[0]))
            nfile   = len(files)
            for j in range(nfile):
                fname = str(j)+'.csv'
                dvec_dat = np.loadtxt(datapath_enrich+'//Dvecdata//'+fname)
                splne_dat  = np.loadtxt(datapath_enrich+'//Splinecurves//'+fname, delimiter=',')
                BG_dat    = np.loadtxt(datapath_enrich+'//BGdata//'+fname, delimiter=',')
                Band_dat  = np.loadtxt(datapath_enrich+'//dispersiondata//'+fname, delimiter=',')
                dvecs_enrich.append(dvec_dat)
                splines_enrich.append(splne_dat)
                BGdata_enrich.append(BG_dat)
                Bands_enrich.append(Band_dat)
                fraw_enrich  = multi_objfun(np.array(Band_dat), BG_op = BG_op)
                f_enrich.append(normalize(fraw_enrich,targ_raw))
                
    os.listdir(datapath_optimal)
    dirs    = os.listdir(datapath_optimal)
    files   = os.listdir((datapath_optimal+'//'+dirs[0]))
    nfile   = len(files)
    for j in range(nfile):
        fname = ('NN'+ str(j)+'.csv')
        if os.path.exists(datapath_optimal+'//Dvecdata//'+fname):
            dvec_dat =  np.loadtxt(datapath_optimal+'//Dvecdata//'+fname)
            splne_dat  = np.loadtxt(datapath_optimal+'//Splinecurves//'+fname, delimiter=',')
            BG_dat    = np.loadtxt(datapath_optimal+'//BGdata//'+fname, delimiter=',')
            Band_dat  = np.loadtxt(datapath_optimal+'//dispersiondata//'+fname, delimiter=',')
            dvecs_optimal.append(dvec_dat)
            splines_optimal.append(splne_dat)
            BGdata_optimal.append(BG_dat)
            Bands_optimal.append(Band_dat)
            fraw_optimal  = multi_objfun(np.array(Band_dat), BG_op = BG_op)
            f_optimal.append(normalize(fraw_optimal,targ_raw))
            
            dvecs_enrich.append(dvec_dat)
            splines_enrich.append(splne_dat)
            BGdata_enrich.append(BG_dat)
            Bands_enrich.append(Band_dat)
            f_enrich.append(normalize(fraw_optimal,targ_raw))
                
########################################
# Load in fval and epsilon
########################################
fstar       =    np.loadtxt(datapath_optimal + '//FEM_star.csv')
fstar_raw   =  np.loadtxt(datapath_optimal + '//FEM_star_raw.csv')
epsilon     =  np.loadtxt(datapath_optimal + '//epsilon.csv')



########################################
# Select optimal index
########################################
inds_keep = epsilon <= 0.05
inds_keep = [i for i, x in enumerate(inds_keep) if x]
fstar_keep = fstar[inds_keep]
indopt = inds_keep[fstar_keep.argmax()]
        

#
########################################
# Plot histogram of samples 
########################################
fig, ax1 = plt.subplots(figsize = (3,3))
color = 'tab:red'
ax1.set_xlabel('Normalized $f(x)$')
ax1.set_ylabel('Counts', color = color)
p1 = ax1.hist(targets, alpha = .5,color ='tab:red',label = 'LHS Samples') 
ax1.tick_params(axis ='y', labelcolor = color)
plt.legend(loc=(-.1,1.05))
ax2 = ax1.twinx()
color = 'tab:green'
ax2.set_ylabel('Counts', color = color)
ax2.hist(f_enrich ,alpha = .5,facecolor = 'g',label = 'Enrichment')
# ax2.hist(f_optimal ,alpha = .5,facecolor = 'b',label = 'Optimal')
ax2.tick_params(axis ='y', labelcolor = color)
plt.legend(loc=(.65,1.05))
if savefigs:  plt.savefig(figpath+'//SampleHist.pdf', bbox_inches='tight')
plt.show()

a_len = .1
from matplotlib.patches import Rectangle
cdata   = np.linspace(0,9,len(splines_optimal))
cdata   = pre.MinMaxScaler().fit_transform(cdata.reshape(-1,1))
plt.figure(figsize = (3,3))
j = 0
for k in range(len(splines_optimal)):
    if k == epsilon.argmin():
        a=0
        # plt.plot(splines_optimal[k][:,0],splines_optimal[k][:,1] ,c='r'  ,label = 'Optimized')
    else:
        if j == 1:
            plt.plot(splines_optimal[k][:,0],splines_optimal[k][:,1] ,color = 'b'  ,alpha=.5  ,label='iterations')
        else:
            plt.plot(splines_optimal[k][:,0],splines_optimal[k][:,1] ,c=cm.Blues(cdata[k] ,alpha=.5  ) )
        j+=1
plt.plot(splines[targets.argsort()[-3]][:,0],splines[targets.argsort()[-3]][:,1],'k--',label = 'Initial')
# plt.plot(splines_optimal[fstar.argmax()][:,0],splines_optimal[fstar.argmax()][:,1] ,c='r', label = 'Optimal')
plt.plot(splines_optimal[indopt][:,0],splines_optimal[indopt][:,1] ,c='r', label = 'Optimal')
plt.legend()
ax = plt.gca()
ax.add_patch( Rectangle( (-a_len/2,-a_len/2),a_len,a_len, facecolor="w" ,ec='k', alpha =.75,label='Unit Cell'))
plt.axis('equal')
plt.axis('square')
plt.xticks([])
plt.yticks([])
if savefigs:  plt.savefig(figpath+'//OptimalSplines.pdf', bbox_inches='tight')
plt.show()

plt.figure(figsize = (3,3))
plt.subplots_adjust(left    =0.1,bottom  =0.1,right   =0.9,top     =0.9, wspace  =0.35, hspace  =0.5)
plt.subplot(211)
plt.plot(epsilon,'ro-',markerfacecolor = 'w')
plt.plot(indopt,epsilon[indopt],'b*',markerfacecolor = 'w')
plt.ylabel('$\epsilon$')
plt.grid()
plt.subplot(212)
plt.plot(fstar,'ro-',markerfacecolor = 'w')
plt.plot(indopt,fstar[indopt],'b*',markerfacecolor = 'w')
plt.xlabel('Enrichement Iteration')
plt.ylabel('$f^*(x)$')
plt.grid()
if savefigs:  plt.savefig(figpath+'//EpsSummary.pdf', bbox_inches='tight')
plt.show()



plt.figure(figsize = (3,3))
optuse = fstar.argmax()
# optuse = epsilon.argmin()



# for k in range(len(epsilon)):
#     if epsilon[k]>0.05: epsilon[k] = 0.05*np.random.rand()
np1,np2,np3 = 20,20,20
x1 = np.linspace(0,1-1/np1,np1)
x2 = np.linspace(1,2-1/np1,np2)
x3 = np.linspace(2,2+np.sqrt(2),np3)
xx = np.concatenate((x1,x2,x3))

ub1 = np.min(Bands_optimal[optuse][:,BG_op])
lb1 = np.max(Bands_optimal[optuse][:,BG_op-1])
lb2 = np.max(Bands[np.argsort(targets)[-3]][:,BG_op-1])
ub2 = np.min(Bands[np.argsort(targets)[-3]][:,BG_op])

ub12 = np.min(Bands_optimal[optuse][:,BG_op2])
lb12 = np.max(Bands_optimal[optuse][:,BG_op2-1])
lb22 = np.max(Bands[np.argsort(targets)[-3]][:,BG_op2-1])
ub22 = np.min(Bands[np.argsort(targets)[-3]][:,BG_op2])
plt.figure(figsize = (3,3))

for k in range(20):
    if k == BG_op-1 or k == BG_op:
        plt.plot(xx,Bands_optimal[optuse][:,k]/lb1,'r-')
        plt.plot(xx,Bands[np.argsort(targets)[-3]][:,k]/lb2,'k-')
    elif k == BG_op2-1 or k == BG_op2:
        plt.plot(xx,Bands_optimal[optuse][:,k]/lb1,'r-')
        plt.plot(xx,Bands[np.argsort(targets)[-3]][:,k]/lb2,'k-')
    else:
        plt.plot(xx,Bands_optimal[optuse][:,k]/lb1,'r--', alpha = .25)
        plt.plot(xx,Bands[np.argsort(targets)[-3]][:,k]/lb2,'k--', alpha = .25)

plt.ylabel('Normalized Freq.')
ax = plt.gca()
# xx = np.linspace(0,60,60)
# ax.add_patch( Rectangle((np.min(xx),lb1/lb1), np.max(xx), (ub1-lb1)/lb1,  facecolor="r" ,ec='r', alpha =.5,label = 'optimized'))
# ax.add_patch( Rectangle((np.min(xx),lb2/lb2), np.max(xx), (ub2-lb2)/lb2,  facecolor="k" ,ec='k', alpha =.5,label = 'un-optimized'))
# ax.add_patch( Rectangle((np.min(xx),lb12/lb12), np.max(xx), (ub12-lb12)/lb12,  facecolor="r" ,ec='r', alpha =.5))
# ax.add_patch( Rectangle((np.min(xx),lb22/lb22), np.max(xx), (ub22-lb22)/lb22,  facecolor="k" ,ec='k', alpha =.5))

# plt.ylim((np.min(Bands_optimal[optuse][:,BG_op-1]/lb1/2),  np.max(Bands_optimal[optuse][:,BG_op]/lb1*1.25)))
plt.legend()
plt.xticks(np.array([0,1,2,2+np.sqrt(2)]), ['$\Gamma$', 'X', 'M', '$\Gamma$'],fontsize=12)
plt.xlim((0,np.max(xx)))
plt.title(f'raw score: {fstar_raw[optuse]:.2f}')
if savefigs:  plt.savefig(figpath+'//OptimalBands.pdf', bbox_inches='tight')
plt.show()


plt.figure(figsize = (3,3))
plt.subplot(121)
for k in range(20):
    if k == BG_op-1 or k == BG_op:
        plt.plot(xx,Bands_optimal[optuse][:,k]/lb1,'r-')
        plt.plot(xx,Bands[np.argsort(targets)[-3]][:,k]/lb2,'k-')
    else:
        plt.plot(xx,Bands_optimal[optuse][:,k]/lb1,'r--', alpha = .25)
        plt.plot(xx,Bands[np.argsort(targets)[-3]][:,k]/lb2,'k--', alpha = .25)
# xx = np.linspace(0,60,60)
ax = plt.gca()
ax.add_patch( Rectangle((np.min(xx),lb1/lb1), np.max(xx), (ub1-lb1)/lb1,  facecolor="r" ,ec='r', alpha =.5,label = 'optimized'))
ax.add_patch( Rectangle((np.min(xx),lb2/lb2), np.max(xx), (ub2-lb2)/lb2,  facecolor="k" ,ec='k', alpha =.5,label = 'un-optimized'))
plt.xticks(np.array([0,1,2,2+np.sqrt(2)]), ['$\Gamma$', 'X', 'M', '$\Gamma$'],fontsize=12)
plt.ylabel('Normalized Freq.')

plt.subplot(122)
for k in range(20):
    if k == BG_op2-1 or k == BG_op2:
        plt.plot(xx,Bands_optimal[optuse][:,k]/lb1,'r-')
        plt.plot(xx,Bands[np.argsort(targets)[-3]][:,k]/lb2,'k-')
    else:
        plt.plot(xx,Bands_optimal[optuse][:,k]/lb1,'r--', alpha = .25)
        plt.plot(xx,Bands[np.argsort(targets)[-3]][:,k]/lb2,'k--', alpha = .25)
# xx = np.linspace(0,60,60)
ax = plt.gca()
# ax.add_patch( Rectangle((np.min(xx),lb1/lb1), np.max(xx), (ub1-lb1)/lb1,  facecolor="r" ,ec='r', alpha =.5,label = 'optimized'))
# ax.add_patch( Rectangle((np.min(xx),lb2/lb2), np.max(xx), (ub2-lb2)/lb2,  facecolor="k" ,ec='k', alpha =.5,label = 'un-optimized'))
ax.add_patch( Rectangle((np.min(xx),lb12/lb1), np.max(xx), (ub12-lb12)/lb1,  facecolor="r" ,ec='r', alpha =.5))
ax.add_patch( Rectangle((np.min(xx),lb22/lb2), np.max(xx), (ub22-lb22)/lb2,  facecolor="k" ,ec='k', alpha =.5))
plt.xticks(np.array([0,1,2,2+np.sqrt(2)]), ['$\Gamma$', 'X', 'M', '$\Gamma$'],fontsize=12)
plt.yticks(color='none')
if savefigs:  plt.savefig(figpath+'//OptimalBands_subplots.pdf', bbox_inches='tight')

#%%











