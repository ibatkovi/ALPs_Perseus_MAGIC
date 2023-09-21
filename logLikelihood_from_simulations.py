#!/usr/bin/env python3

'''

This is a script for calculation of the likelihood for considered models of photon-ALP mixing in the ALPs parameter space.

It was written by Giacomo D'Amico and Ivana Batković for the needs of the article "Constraints on axion-like particles with the Perseus Galaxy Cluster with MAGIC". 

If you wish to use the script and reproduce the results, you are invited to contact the authors:

Giacomo D'Amico, giacomo.damico@uib.no
Ivana Batković, ivana.batkovic@phd.unipd.it

For running this script, gammapy-0.20 version is needed, in case you miss it, check: https://docs.gammapy.org/0.20/


'''


import sys
import os

import numpy as np
import astropy.units as u
from pathlib import Path
import glob

from gammapy.stats.fit_statistics import wstat

import pickle
import gzip

from ALPgrid import  check_keys_in_dict




class likelihood_profile:
    def __init__(self, total_s, total_n_on, total_n_off, alpha, ampl_factor):
        
        self.total_s     = total_s
        self.total_n_on  = total_n_on
        self.total_n_off = total_n_off
        self.alpha       = alpha
        self.ampl_factor = ampl_factor
                
    def compute_likelihood(self):

        lkl_nuisance_B  = np.sum( [ self.compute_likelihood_for_each_dataset(i_s,i_n_on,i_n_off ) for i_s, i_n_on, i_n_off 
                                            in zip( self.total_s, self.total_n_on, self.total_n_off)] ,axis=0)
                
        self.lkl_min = np.sort(  lkl_nuisance_B )[5]      

        
    def compute_likelihood_for_each_dataset(self,s,n_on,n_off ):
        # B-field, reco energy, 3 sed param.
        # Add amplitufde as extra dimension
        exp_signal_counts   = s[:,:,:,:,None]*self.ampl_factor[None,None,None,None,:]
        n_on                = n_on[None,:,None,None,None]
        n_off               = n_off[None,:,None,None,None]
        # get likelihood and sum over energy bin
        lkl_nuisance_i          = np.sum( wstat(n_on=n_on, n_off=n_off, mu_sig=exp_signal_counts, alpha=self.alpha), axis=1)
        # profile over source SED parameters
        lkl_nuisance_i          = np.amin( lkl_nuisance_i, axis=(1,2,3))
        
        return  lkl_nuisance_i
        
        


def get_filename_by_number(num, file_path):
    with open(file_path, 'r') as f:
        for line in f:
            if line.startswith(str(num)):
                return line.split("\t")[1].strip()
    print("Line "+str(num)+" already computed")
    
    
def add_hash_to_line(string, file_path):
    # Read the file into a list of lines
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    # Iterate through the lines, checking if the string is in the line
    for i, line in enumerate(lines):
        if string in line:
            # If the string is in the line, add "#" at the beginning of the line
            lines[i] = "#" + line
    
    # Write the modified lines back to the file
    with open(file_path, 'w') as f:
        f.writelines(lines)
    
def check_keys_in_dict(input_dict):
    
    sorted_dict = {}
    
    x      = u.Quantity( [i[0] for i in input_dict.keys() ] )
    x_unit = x.unit
    y      = u.Quantity( [i[1] for i in input_dict.keys() ] )
    y_unit = y.unit
    
    
    # CHECK DIMENSIONS AND 
    # ASSIGN MASS AND COUPLING LIST
    m_list, g_list = None, None
    # FOR X LIST
    try:
        test   = x_unit/u.eV
        test   = test.to( u.dimensionless_unscaled)
        m_list = x
    except:
        try:
            test   = x_unit*u.GeV
            test   = test.to( u.dimensionless_unscaled)
            g_list = x
        except:
            raise ValueError("Parameter dimension should be eV for the ALP mass and 1/GeV for the coupling!")
    # FOR Y LIST
    try:
        test   = y_unit/u.eV
        test   = test.to( u.dimensionless_unscaled)
        m_list = y
    except:
        try:
            test   = y_unit*u.GeV
            test   = test.to( u.dimensionless_unscaled)
            g_list = y
        except:
            raise ValueError("Parameter dimension should be eV for the ALP mass and 1/GeV for the coupling!")
    ##
    if m_list is None or g_list is None:
            raise ValueError("Parameter dimension should be eV for the ALP mass and 1/GeV for the coupling!") 
            
    m_list   = np.sort( np.unique( m_list ) )
    g_list   = np.sort( np.unique( g_list ) )
    grid     = np.meshgrid(m_list,g_list)
    for im,ig in zip(  np.ravel( grid[0] )  , np.ravel( grid[1] ) ):
        im_converted   = im.to(u.eV)
        ig_converted   = ig.to( 1/u.GeV)
        try:
            sorted_dict[im_converted,ig_converted] = input_dict[im,ig]
        except:
            sorted_dict[im_converted,ig_converted] = input_dict[ig,im]
        
    return sorted_dict


    
    
    
def main():
    # Check if the user provided an argument
    if len(sys.argv) < 2:
        print("Please provide an argument")
        return
    
    # Get the first argument 
    num = sys.argv[1]
    
    #list_of_file_names file has to contain list of paths to the files containing ON counts extracted from the datasets considering each of the models of ALPs considered, i.e. fake_on_off_counts/fake_on_counts_1000__1744_.pk where "1000" represents m_a = 10 neV and "1744" is coupling to photons g_a\gamma = 1.744 * 10^-11 Gev^-1.
    
    file_name_on = get_filename_by_number(num, "./list_of_file_names.txt")
    if file_name_on == None:
        return
    print("Computing LogLs using counts in "+file_name_on)
       
    
    names_split = str.split(file_name_on, "_")
    
    g = names_split[-2]
    m = names_split[-4]
    g = float(g) * 1e-14 / u.GeV
    m = float(m) * 1e-2 * 1e-9 * u.eV

    assumed_true_g_m = (m,g)
    
    # LOAD ON DATA
    with open(file_name_on, 'rb') as f:
            n_on =  pickle.load(f)
        

    names_split = str.split(file_name_on, "_")
    

    # CREATE DIRECT WHERE TO STORE THE RESULTS
    name_m_folder        = str(int(m.to(0.01*u.neV).value))
    name_g_folder        = str(int(g.to( 1e-14 /u.GeV).value))
    name_folder          = "logLikelihood_values/logLikelihood_values_"+name_m_folder+"__"+name_g_folder+"/"
    
    some_logLs_already_computed = False
    if not os.path.exists(name_folder):
        os.makedirs(name_folder)
    else:
        some_logLs_already_computed = True


    # LOAD OFF DATA
    file_name_off = "fake_on_off_counts/fake_off_counts_"+name_m_folder+"__"+name_g_folder+"_.pkl"
    with open(file_name_off, 'rb') as f:
                n_off =  pickle.load(f)
            
            

            
    # RUN THE LOGL COMPUTATION ON ALL SIMULATED COUNTS
    for number_of_simulation in range(len(n_on)):
        file_name = name_folder+str(number_of_simulation)+"_LogLs_.pkl"
       

        # CHECK IF LogLs were already computed
        if some_logLs_already_computed:
            # Load the previously obtained logLs
            with open(file_name, 'rb') as f:
                all_logl = pickle.load(f)

        else:
            all_logl     =  {}
        
        n_on_i  = n_on[number_of_simulation]
        n_off_i = n_off[number_of_simulation] 
        ampl_factors = np.linspace(0.5,1.5,30)

        # LOOP OVER ALL  EXPECTED COUNTS - WE GET g AND m FROM FILE NAME
        for name in glob.glob("fake_on_off_counts/fake_off_counts_*.pkl"):

            ## GET INFO ON M AND G FROM FILE NAME
            names_split = str.split(name, "_")
            g = names_split[-2]
            m = names_split[-4]
            g = float(g) * 1e-14 / u.GeV
            m = float(m) * 1e-2 * u.neV


            m_g_inside_square = (m,g) in all_logl.keys()


            if m_g_inside_square and some_logLs_already_computed:
                continue


            # LOAD THE EXPECTED COUNTS
            total_s = []
            for k in range(4):
                name_m = str(int(m.to(0.01*u.neV).value))
                name_g = str(int(g.to( 1e-14 /u.GeV).value))
                file_name2 = "expected_counts/expected_counts_"+name_m+"__"+name_g+"_"+str(k)+"_array.npy.gz"
                print("I am loading the expected counts from: "+file_name2)
                with gzip.open(file_name2, 'rb') as f:
                    total_s.append( np.load(f) )

            # COMPUTE THE LIKELIHOOD
            likelihood = likelihood_profile( 
                total_s      = total_s , 
                total_n_on   = n_on_i, 
                total_n_off  = n_off_i, 
                alpha        = 1/3, 
                ampl_factor  = ampl_factors)
            print("Now I am about to compute likelihood")
            likelihood.compute_likelihood()

            # SAVE THE RESULTS
            all_logl[m,g]           = likelihood.lkl_min


        # ORDER THE KEYS
        all_logl = check_keys_in_dict(all_logl)

        # SAVE THE RESULT
        file_name = name_folder+str(number_of_simulation)+"_LogLs_.pkl"
        #file_name = direct+"observed_logLs_.pkl"
        with open(file_name, 'wb') as f:
            pickle.dump(all_logl,f)
        print("\n")
        print("Result saved in "+file_name)
        print("\n")


    TS_list = []
    for file_name in glob.glob(name_folder+"*LogLs_.pkl"): 
        with open(file_name, 'rb') as f:
            all_logl = pickle.load(f)

        #GET MIN
        for i, mg in enumerate(all_logl.keys()):
            logl   = all_logl[mg]
            if i ==0:
                logl_min = logl
            if logl < logl_min:
                logl_min = logl

        all_statistic = {}
        for i in all_logl.keys():

            logl      = all_logl[i]
            all_statistic[i] = logl- logl_min 


        TS_list.append( all_statistic[assumed_true_g_m])

    file_name = name_folder+"All_TS_.npy"
    np.save(file_name,  np.array(TS_list))

    
    
    
    add_hash_to_line(string=file_name_on, file_path="./list_of_file_names.txt")
    
    
if __name__ == "__main__":
    main()
