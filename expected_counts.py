#!/usr/bin/env python3

'''

This is a script for extraction of the expected counts from the datasets considering different models of photon-ALP mixing in the ALPs parameter space.

It was written by Giacomo D'Amico and Ivana Batković for the needs of the article "Constraints on axion-like particles with the Perseus Galaxy Cluster with MAGIC". 

If you wish to use the script and reproduce the results, you are invited to contact the authors:

Giacomo D'Amico, giacomo.damico@uib.no
Ivana Batković, ivana.batkovic@phd.unipd.it


For running this script, gammapy-0.20 version is needed, in case you miss it, check: https://docs.gammapy.org/0.20/


'''


import sys

import matplotlib.pyplot as plt


import numpy as np
import astropy.units as u
from pathlib import Path
import glob
from astropy.io import fits
from astropy.coordinates import SkyCoord, Angle
from regions import CircleSkyRegion, PointSkyRegion
import pickle

from scipy.stats import chi2
from functools import partial
import scipy.special as scipys

# gammapy modules

from gammapy.modeling import Fit
import gammapy.irf as irf
from gammapy.irf import load_cta_irfs
from gammapy.data import Observation, Observations, DataStore
from gammapy.utils.random import get_random_state
from gammapy.maps import MapAxis



# models modules

from gammapy.modeling.models import (
    Model,
    Models,
    SkyModel,
    PowerLawSpectralModel,
    PowerLawNormSpectralModel,
    ExpCutoffPowerLawSpectralModel,
    PointSpatialModel,
    GaussianSpatialModel,
    TemplateSpatialModel,
    FoVBackgroundModel,
    SpectralModel, 
    TemplateSpectralModel,
    EBLAbsorptionNormSpectralModel,
)
# dataset modules
from gammapy.datasets import (
    MapDataset, 
    MapDatasetOnOff, 
    MapDatasetEventSampler,
    SpectrumDatasetOnOff,
    SpectrumDataset, 
    Datasets,
    FluxPointsDataset
)

from gammapy.maps import MapAxis, WcsGeom, Map, RegionGeom
from gammapy.makers import MapDatasetMaker, SpectrumDatasetMaker, ReflectedRegionsBackgroundMaker, WobbleRegionsFinder
from gammapy.estimators import FluxPointsEstimator
from gammapy.stats.fit_statistics import wstat



## IRF FUNCTIONS

def get_mask_migra( observation, ereco, etrue):
    
    
    edisp                  = observation.edisp
    edisp.normalize()
    
    migra_min , migra_max  = edisp.axes['migra'].bounds.value
     
    etrue_min  = ereco[:,None]   / migra_max
    etrue_max  = ereco[:,None]   / migra_min
    mask_migra = (etrue[None,:] >= etrue_min ) * (etrue[None,:] <= etrue_max )
    
    return mask_migra


def make_edisp_factors(observation,true_offset, etrue,  ereco, mask_migra ):

        
    ereco       =  ereco[:,None]
    etrue       =  etrue[None,:]
    
    edisp2D        = observation.edisp
    edisp2D.normalize()
    edisp        =  edisp2D.evaluate( 
                     offset      = true_offset,
                     energy_true = etrue,
                     migra       = ereco/etrue 
                    ) / etrue
    
    edisp       =  edisp*mask_migra
    
    return edisp


def make_aeff_factors(observation, true_offset, etrue, mask_migra):
    
    etrue       =  etrue[None,:]
    
    etrue       =  etrue*mask_migra
    
    aeff         = observation.aeff.evaluate(offset=true_offset , energy_true=etrue)
    return aeff

def get_IRF( observation, true_offset, ereco,etrue):
    
    mask_migra = get_mask_migra(    observation, ereco,  etrue)
    edisp      = make_edisp_factors( observation,true_offset, etrue,  ereco, mask_migra) 
    aeff       = make_aeff_factors( observation, true_offset, etrue,  mask_migra)
    
    time_unit = u.Unit( observation.obs_info['TIMEUNIT'] )
    
    return edisp*aeff *observation.obs_info['LIVETIME']*time_unit
    
    
    
### Function for computing counts, expected signals 
def get_array_from_value_and_error(res_dict, n_sigma=3, nbins=20):
    center = res_dict['value']
    err    = res_dict['error']
    xmin   = center - n_sigma*err
    if xmin <0:
        xmin = 0
    xmax    = center + n_sigma*err
    x_arr   = np.linspace(xmin,xmax, nbins)
    if center not in x_arr:
        x_arr   = np.linspace(xmin,xmax, nbins-1)
        x_arr   = np.append(x_arr, center )
        x_arr   = np.sort(x_arr)
 
    return x_arr *  u.Unit( res_dict['unit'] )


def get_exp_signal_counts_from_obs_list( true_flux, observations, source_coord, ereco,etrue, delta_ereco, delta_etrue):
    
    delta_etrue = delta_etrue[None,None,:,None,None] 
    delta_ereco = delta_ereco[None,:,None,None]
    
    IRF  = np.zeros( (len(ereco),len(etrue)) ) * u.m**2 * u.s/u.GeV
    for obs in observations:
        true_offset = obs.pointing_radec.separation( source_coord)
        IRF        += get_IRF( obs, true_offset, ereco,etrue)
               
    obs_flux           = np.sum( true_flux[:,None,:,:,:]*IRF[None,:,:,None,None]*delta_etrue,axis=2)
    obs_flux           = obs_flux.to(1/( u.GeV  ))
    exp_signal_counts  = obs_flux*delta_ereco
        

    return np.array( exp_signal_counts.to('').value)# dtype=np.float16)

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


    

##################################################
##################################################
#################################################
##################################################

# LOAD DL3 OBSERVATION

observations_flare = Observations()
for filename in glob.glob(f"../path/where/the/fits/files/are/stored/flaring_state/*fits"):
    observations_flare.append(Observation.read(filename))

observations_post_flare = Observations()
for filename in glob.glob(f"../path/where/the/fits/files/are/stored/post-flaring_state/*fits"):
    observations_post_flare.append(Observation.read(filename)) 

observations_low_state = Observations()
for filename in glob.glob(f"../path/where/the/fits/files/are/stored/low-state/*fits"):
    observations_low_state.append(Observation.read(filename)) 


# Define energy bins and geometry for EACH DATASET
nbins_true = 200

# FLARE
emin             = 50*u.GeV
emax             = 2.1*u.TeV
nbins            = 27     
en_edges               = np.geomspace(  emin, emax, nbins) 
nergy_axis             = MapAxis.from_edges(en_edges, interp='log' , unit="GeV", name="energy")
energy_reco_axis_flare = MapAxis.from_edges(en_edges, interp='log' , unit="GeV", name="energy")
energy_true_axis_flare = MapAxis.from_energy_bounds(5, 5e4, nbin=nbins_true, per_decade=False, unit="GeV", name="energy_true")


# POST FLARE
emin             = 64*u.GeV
emax             = 1.4*u.TeV
nbins            = 25    
en_edges                    = np.geomspace(  emin, emax, nbins) 
nergy_axis                  = MapAxis.from_edges(en_edges, interp='log' , unit="GeV", name="energy")
energy_reco_axis_post_flare = MapAxis.from_edges(en_edges, interp='log' , unit="GeV", name="energy")
energy_true_axis_post_flare = MapAxis.from_energy_bounds(5, 5e4, nbin=nbins_true, per_decade=False, unit="GeV", name="energy_true")

# LOW STATE
emin             = 70*u.GeV
emax             = 2.1*u.TeV
nbins            = 20     
en_edges                   = np.geomspace(  emin, emax, nbins) 
nergy_axis                 = MapAxis.from_edges(en_edges, interp='log' , unit="GeV", name="energy")
energy_reco_axis_low_state = MapAxis.from_edges(en_edges, interp='log' , unit="GeV", name="energy")
energy_true_axis_low_state = MapAxis.from_energy_bounds(5, 5e4, nbin=nbins_true, per_decade=False, unit="GeV", name="energy_true")


source_coordinates_ngc1275 = SkyCoord.from_name("NGC1275")


# Decide on the choice of the EBL absorprtion model, in our case, we chose:
# Dominguez et al. (doi:10.1111/j.1365-2966.2010.17631)

dominguez = EBLAbsorptionNormSpectralModel.read_builtin("dominguez", redshift=0.01790)



def compute_expected_counts(name):

    
    # FLARE
    
    p_gamma_gamma = []
    for i in range(0,100):
        en_absorp_array     = np.load(name)
        energy              = en_absorp_array[0] * u.GeV
        values              = en_absorp_array[1+i] * u.Unit("") # whatever numer from 1 to 100
        absorption          = TemplateSpectralModel(energy, values)
        p_gamma_gamma.append( absorption( energy_true_axis_flare.center).to('').value )
    p_gamma_gamma       = np.array(p_gamma_gamma)
 
    # Best fit parameters from fitting the intrinsic spectrum of the dataset with a simple function, in our case an EPWL.
    reference = 0.3 * u.TeV
    amplitude = 16.1e-10 * u.Unit("TeV-1 cm-2 s-1")
    index     = 2.11
    index     = np.linspace( index*0.5, index*1.5, 20)
    lambda_   = 1.24 * u.Unit("TeV-1")
    lambda_   = np.linspace( lambda_*0.5, lambda_*1.5, 20)
    
    energies = energy_true_axis_flare.center
    energies = energies[:,None,None]
    index    = index[None,:,None]
    lambda_  = lambda_[None,None,:]

    true_flux = amplitude*(energies/reference)**(-index) * np.exp(-energies*lambda_) 
    true_flux = true_flux[None,:,:,:] * p_gamma_gamma[:,:,None,None]

    exp_signals_flare = get_exp_signal_counts_from_obs_list( 
                                        true_flux    = true_flux, 
                                        observations = observations_flare, 
                                        source_coord = source_coordinates_ngc1275, 
                                        ereco        = energy_reco_axis_flare.center,  
                                        etrue        = energy_true_axis_flare.center, 
                                        delta_ereco  = energy_reco_axis_flare.bin_width,
                                        delta_etrue  = energy_true_axis_flare.bin_width,

                                                     )

    
    # POST FLARE

    p_gamma_gamma = []
    for i in range(0,100):
        en_absorp_array     = np.load(name)
        energy              = en_absorp_array[0] * u.GeV
        values              = en_absorp_array[1+i] * u.Unit("") # whatever numer from 1 to 100
        absorption          = TemplateSpectralModel(energy, values)
        p_gamma_gamma.append( absorption( energy_true_axis_post_flare.center).to('').value )
    p_gamma_gamma       = np.array(p_gamma_gamma

    # Best fit parameters from fitting the intrinsic spectrum of the dataset with a simple function, in our case an EPWL.
    reference = 0.3 * u.TeV
    amplitude = 11.4e-10 * u.Unit("TeV-1 cm-2 s-1")
    index     = 1.79
    index     = np.linspace( index*0.5, index*1.5, 20)
    lambda_   = 3.45 * u.Unit("TeV-1")
    lambda_   = np.linspace( lambda_*0.5, lambda_*1.5, 20)
    
    energies = energy_true_axis_post_flare.center
    energies = energies[:,None,None]
    index    = index[None,:,None]
    lambda_  = lambda_[None,None,:]

    true_flux = amplitude*(energies/reference)**(-index) * np.exp(-energies*lambda_) 
    true_flux = true_flux[None,:,:,:] * p_gamma_gamma[:,:,None,None]


    exp_signals_post_flare = get_exp_signal_counts_from_obs_list( 
                                        true_flux    = true_flux, 
                                        observations = observations_post_flare, 
                                        source_coord = source_coordinates_ngc1275, 
                                        ereco        = energy_reco_axis_post_flare.center,  
                                        etrue        = energy_true_axis_post_flare.center, 
                                        delta_ereco  = energy_reco_axis_post_flare.bin_width,
                                        delta_etrue  = energy_true_axis_post_flare.bin_width,

                                                     )

   
    # LOW STATE
    
    p_gamma_gamma = []
    for i in range(0,100):
        en_absorp_array     = np.load(name)
        energy              = en_absorp_array[0] * u.GeV
        values              = en_absorp_array[1+i] * u.Unit("") # whatever numer from 1 to 100
        absorption          = TemplateSpectralModel(energy, values)
        p_gamma_gamma.append( absorption( energy_true_axis_low_state.center).to('').value )
    p_gamma_gamma       = np.array(p_gamma_gamma)
    
    # Best fit parameters from fitting the intrinsic spectrum of the dataset with a simple function, in our case an EPWL.
    reference=0.3 * u.TeV
    amplitude = 1.1e-10 * u.Unit("TeV-1 cm-2 s-1")
    index     = 2.54
    index     = np.linspace( index*0.5, index*1.5, 20)
    lambda_   = 2 * u.Unit("TeV-1")
    lambda_   = np.linspace( lambda_*0.5, lambda_*1.5, 20)
    
    energies = energy_true_axis_low_state.center
    energies = energies[:,None,None]
    index    = index[None,:,None]
    lambda_  = lambda_[None,None,:]

    true_flux = amplitude*(energies/reference)**(-index) * np.exp(-energies*lambda_) 
    true_flux = true_flux[None,:,:,:] * p_gamma_gamma[:,:,None,None]


    exp_signals_low_state = get_exp_signal_counts_from_obs_list( 
                                        true_flux    = true_flux, 
                                        observations = observations_low_state, 
                                        source_coord = source_coordinates_ngc1275, 
                                        ereco        = energy_reco_axis_low_state.center,  
                                        etrue        = energy_true_axis_low_state.center, 
                                        delta_ereco  = energy_reco_axis_low_state.bin_width,
                                        delta_etrue  = energy_true_axis_low_state.bin_width,

                                                     )
    
    # An array containing the expected counts from all the three datasets, respectively.                               
    return [exp_signals_flare, exp_signals_post_flare, exp_signals_low_state] 
