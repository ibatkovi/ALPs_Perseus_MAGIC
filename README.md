# ALPs_Perseus_MAGIC
Constraints on axion-like particles with the Perseus Galaxy Cluster with MAGIC

This gitHub page is devoted to the artile "Constraints on axion-like particles with the Perseus Galaxy Cluster with MAGIC".

In this paper, we search for axion-like particle (ALP) signatures in the spectral energy distribution (SED) of NGC1275 in the centre of the Perseus galaxy cluster.
In case axion-like particles exist, propagation of the very-high-energy gamma rays through astronomical distances and environments can be hindered with photon-ALP oscillations due to the strong external magnetic field. 
Considering magnetic fields of the environments in the line of sight, and obtaining the equations of motion, we plan to calculate the so-called photon survival probability. 
Photon-survival probability is the probability that once emitted photons of either polarization will survive the propagation and be detected as such. 
For this purpose, we will be using the Gammapy, open-source package for gamma-ray astronomy, and GammaALPs, used for the ALPs analysis and calculation of photon survival probability. 
Statistical analysis will be performed with Gammapy using two different approaches, binned and unbinned analysis, with the aim to compare their performance in the case of ALPs studies. 
Our goal is to use the NGC1275 containing one flaring state, post-flaring and low activity state to test the values of ALPs mass and their coupling to photons, and ultimately constrain the ALPs parameter space.

*Repository is currently under construction and the paper is in the phase of the submission to the journal. Updates with the DOI and arXiv number of the article will follow once the submission is concluded and the paper is published.*

Content of the respository:

This repository is prepared with the intention to allow the reproduction of the main results obtained in the paper. It contains:
a) Python scripts for the calculation of the ALPs models with GammaALPs./b
b) Scripts for the extraction of information from the datasets of NGC1275 by MAGIC fits files and calculation of the likelihood as explained in the article/n
c) ECSV tables for creating the SED plots./n
d) Folder "Axion_Photon" containing the information on where to find a collection of previously set constraints in the ALPs parameter space, together with the constraint set in this article. 
