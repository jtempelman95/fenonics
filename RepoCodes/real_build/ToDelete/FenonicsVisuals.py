
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np


def plotbands(bands, figsize = (5,4)):
    """
    Plot the disprsion bands with bandgaps highlighted
    """ 
    bgnrm, gapwidths, gaps, lowbounds, highbounds = getbands(bands)

    plt.figure(figsize=figsize)
    np1 = 20; np2 = 20; np3 = 20
    x1 = np.linspace(0,1-1/np1,np1)
    x2 = np.linspace(1,2-1/np1,np2)
    x3 = np.linspace(2,2+np.sqrt(2),np3)
    xx = np.concatenate((x1,x2,x3))
    nvec = 20
    maxfq = 5e3
    # PLOT THE DISPERSION BANDS
    for n in range(nvec):
        ev = bands[:,n]
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
            currentAxis.add_patch( Rectangle((np.min(xx),lb), np.max(xx), ub-lb,  facecolor="g" ,ec='none', alpha =.3,label='bangap'))
        else:
            currentAxis.add_patch( Rectangle((np.min(xx),lb), np.max(xx), ub-lb,  facecolor="g" ,ec='none', alpha =.3))
    plt.legend()
    return plt
