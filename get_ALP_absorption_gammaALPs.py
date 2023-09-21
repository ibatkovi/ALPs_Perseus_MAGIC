import sys
import matplotlib.pyplot as plt
import matplotlib as mpl
plt.rc('xtick', labelsize=20)   
plt.rc('ytick', labelsize=20)
plt.rcParams['axes.linewidth'] = 2.5
plt.rc('text', usetex=True)
plt.rc('font', family='serif',size=25)

import numpy as np
import astropy.units as u
from pathlib import Path
import glob
from astropy.io import fits
from astropy.coordinates import SkyCoord, Angle
from regions import CircleSkyRegion
import pickle

from scipy.stats import chi2
from functools import partial
import scipy.special as scipys

###This notebook requires installation of GammaALPs, otherwise it will not function. Hence, in case you are missing it, check https://gammaalps.readthedocs.io/en/latest/index.html for instructions on the installation.

from gammaALPs.core import Source, ALP, ModuleList
from gammaALPs.base import environs, transfer

#############################################
# True if you want a log file

LogOnFile  = True


# To get a log file 
class Logger(object):
    def __init__(self):
        self.terminal = sys.stdout
        self.log = open("logfile.log", "a")

    def write(self, message):
        self.terminal.write(message)
        self.terminal.flush()
        self.log.write(message)
        self.log.flush()

    def flush(self):
        #this flush method is needed for python 3 compatibility.
        #this handles the flush command by doing nothing.
        #you might want to specify some extra behavior here.
        pass



def compute_ALP_absorption(modulelist, axion_mass, coupling, emin, emax, bins):
    ''' Input:
            -  modulelist:     ModuleList object assuming a given source
            -  axion_mass:     axion mass / 1 neV
            -  coupling  :     axion-gamma coupling / 1e-11 GeV^-1
            -  emin      :     min energy / GeV
            -  emin      :     max energy / GeV
            -  bins      :     number of points in energy log-sperated
        Output:
            -  energy points
            -  gamma absorption for the above energy points

    '''
    ebins            = np.logspace(np.log10(emin),np.log10(emax),bins)
    
    # unit conversion
    axion_mass       = axion_mass.to(u.neV)
    coupling         = coupling.to(1e-11/u.GeV)
    # passing m and g to the gammaALPs modulelist object
    modulelist.alp.m = axion_mass.value
    modulelist.alp.g = coupling.value
    modulelist.EGeV  = ebins

    px,  py,  pa     = modulelist.run(multiprocess=2)
    pgg              = px + py
    
    return modulelist.EGeV, pgg


################################################################################

###############################
## Standalone mode           ##
###############################

#This is executed if the program is called standalone:
#
#  ./get_ALP_absorption.py 
#

if __name__ == '__main__':
    
    if(LogOnFile):
        sys.stdout = Logger()

#If you consider altering the model of the galaxy cluster magnetic field, modify the values assigned to the parameters below

    source     = Source(z = 0.017559, ra = '03h19m48.1s', dec = '+41d30m42s') # this is for ngc1275

    pin        = np.diag((1.,1.,0.)) * 0.5
    alp        = ALP(0,0) 
    modulelist = ModuleList(alp, source, pin = pin)
    modulelist.add_propagation("ICMGaussTurb", 
              0, # position of module counted from the source. 
              nsim = 100, # number of random B-field realizations
              B0 = 10.,  # rms of B field
              n0 = 39.,  # normalization of electron density
              n2 = 4.05, # second normalization of electron density, see Churazov et al. 2003, Eq. 4
              r_abell = 500., # extension of the cluster
              r_core = 80.,   # electron density parameter, see Churazov et al. 2003, Eq. 4
              r_core2 = 280., # electron density parameter, see Churazov et al. 2003, Eq. 4
              beta = 1.2,  # electron density parameter, see Churazov et al. 2003, Eq. 4
              beta2= 0.58, # electron density parameter, see Churazov et al. 2003, Eq. 4
              eta = 0.5, # scaling of B-field with electron denstiy
              kL = 0.18, # maximum turbulence scale in kpc^-1, taken from A2199 cool-core cluster, see Vacca et al. 2012 
              kH = 9.,  # minimum turbulence scale, taken from A2199 cool-core cluster, see Vacca et al. 2012
              q = -2.80, # turbulence spectral index, taken from A2199 cool-core cluster, see Vacca et al. 2012
              seed=0 # random seed for reproducability, set to None for random seed.
             )
    modulelist.add_propagation("EBL",1, model = 'dominguez') # EBL attenuation comes second, after beam has left cluster
    modulelist.add_propagation("GMF",2, model = 'jansson12', model_sum = 'ASS') # finally, the beam enters the Milky Way Field

    ### set the values or the array of values for the mass of ALPs that you want to consider
    m =  0.4641588834 * u.neV
  
    ### set the values or the array of values for the coupling to photons that you want to consider    
    g_array    = np.logspace(np.log10(8) , np.log10(50), 4)*1e-11/u.GeV    
    
    
    j = 0
    #for m in m_array:
    for g in g_array:

        ms = m.to( 1e-2 * u.neV)
        gs = g.to( 1e-14 / u.GeV)

        enpoints, pgg   = compute_ALP_absorption(
                            modulelist = modulelist, # modulelist from gammaALP
                            axion_mass = m, # neV
                            coupling   = g , # 10^(-11) /GeV
                            emin       = 50,  # Gev
                            emax       = 2.2e4, # GeV
                            bins       = 100) # log-bins in enrgy for which computing the ALP-absorption

        arr = [enpoints] 
        for ipg in pgg:
            arr.append(ipg )
        arr = np.array(arr)
        file_name = f'ngc1275_100real_m_%0.f_g_%0.f_'  % (ms.value, gs.value) 
        np.save(  f"./path/where/you/want/to/store/models/{file_name}.npy", arr) 
        
        fig, ax = plt.subplots(figsize=(15,10))
        ax.grid(True,which='both',linewidth=0.3)
        ax.set_ylabel('Photon survival probability', size=30)
        ax.set_xlabel('E [GeV]',size=30)
        ax.set_xlim([3e1,3e4])
        ax.set_ylim([0,1.05])
        ax.set_xscale("log")
        
        for ipg in pgg:
            line,  =  ax.plot(enpoints, ipg)

        line.set_color("white")
        mlegend = m.to( 1e1 * u.neV)
        glegend = g.to( 1e-11 / u.GeV)

        str_leg = f'm \t = %0.f   {mlegend.unit}  \n g  \t = %0.1f   {glegend.unit}'  % (mlegend.value, glegend.value) 
        line.set_label(str_leg)
        ax.legend(fontsize = 25, loc="lower left", markerscale=1)


        file_name = f'%0.f_ngc1275_100real_m_%0.f_g_%0.f_'  % (j,ms.value, gs.value) 
        fig.savefig(f"./path/where/you/want/to/store/models/{file_name}.png")
        fig.clf()
        #plt.close()
        j += 1

        plt.close('all')
