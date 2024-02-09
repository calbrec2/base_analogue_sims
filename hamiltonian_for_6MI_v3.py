#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 22 15:46:47 2023

@author: calbrec2
"""

# build up a calculation to develop the hamiltionian for the 2PE 6-MI experiments
# Tools
import scipy as sp
from scipy.optimize import minimize, minimize_scalar, brute , differential_evolution, fmin, basinhopping

#import ext_dip as ed  # This file contains the definitions for 
                      #  the dipole orientation angles and the
                      #  extended dipole J calculation
# import sys
# sys.path.append("/Volumes/Dylans Exte/Transition Charge Project/tq code")

import os 
print(os.getcwd())

# import atomistic_tq_jan23 as tc 
# import atomistic_tq_jan23_CSA as tc 
                        # This file contains the definitions for 
                        # the dipole orientation angles and the
                        # atomistic transition charge J calculation

import numpy as np
import math as ma
import matplotlib.pyplot as plt
import scipy.linalg as la
import pandas as pd
#import ext_dip_New as ed


# Time/date stamp runs:
from datetime import datetime as dt
import time

# Stop Warning Messages
import warnings
warnings.filterwarnings('ignore')

import os 
cwd = os.path.dirname(os.path.realpath(__file__))
print(cwd)
#%%
###############################################
# Plotting function for looking at matrices
def matrix_plotter(omegas, alpha_x, alpha_y, title, frac=0.8, size=6, fontsize=14,title_fontsize=18,label_fontsize=12):
    fig = plt.figure(figsize=[size,size])
    ax = fig.add_subplot(111)
    cax = ax.matshow(np.real(omegas)/1e3,interpolation='nearest',cmap='Blues')
    # ax.set_title(r'Differences between eigenenergies ($cm^{-1}$ x$10^{3}$)',fontsize=14)
    ax.set_title(title,fontsize=title_fontsize)
    fig.colorbar(cax)
    for (i, j), z in np.ndenumerate(np.real(omegas)/1e3):
        if z > np.max(np.real(omegas)/1e3)*frac: color='w' 
        else: color='k'
        ax.text(j, i, '{:0.1f}'.format(z), ha='center', va='center', color=color,fontsize=label_fontsize)
    ax.set_xticks(np.arange(len(alpha_x)));
    ax.set_yticks(np.arange(len(alpha_y)));
    ax.set_xticklabels(alpha_x,fontsize=fontsize);
    ax.set_yticklabels(alpha_y,fontsize=fontsize);
    # ax.set_xticklabels(['']+alpha_x,fontsize=fontsize);
    # ax.set_yticklabels(['']+alpha_y,fontsize=fontsize);

###############################################
# =============================================================================
# # Functions  to grab data
# =============================================================================
def Closeup(data, low, high):
    '''trims a data set down to fit in x range between low and high
    
    array, num, num -> array'''
    
    temp = []
    for dat in data:
        if (dat[0] > low) and (dat[0] < high):
            temp.append(dat)
    return np.array(temp)

# def get_c_spectra(temp):
def get_c_spectra():
    '''generates Abs and CD spectra from a text file
    
    str (e.g. "J89"), str (e.g. "15") -> list of arrays'''
    
    terminalID = '/Users/fulton/'
    
    # os.chdir('/Users/clairealbrecht/Dropbox/MATLAB_programs/claire_programs/from_Lulu/20230726')
    # os.chdir('/Users/calbrecht/Dropbox/MATLAB_programs/claire_programs/from_Lulu/20230726')
    # os.chdir(terminalID+'Dropbox/MATLAB_programs/claire_programs/from_Lulu/20230726')
    # file_name = 'DNTDP_10perc_20230306_finescan_10nm_min.txt'
    # file_name = '20230726_MNS_4uM_20230726_finescan.txt'
    
    
    import os.path
    from pathlib import Path

    home = str(Path.home())
    path = os.path.join(home,'Documents','Github','base_analogue_sims','Data','CD_Abs','20230726')
    os.chdir(path)
    
    buf_file_name = '20230726_buffer2.txt'
    buf_data = np.loadtxt(buf_file_name, skiprows=21)
    file_name = '20230726_MNS_4uM_20230726_finescan.txt'
    
    
    # # os.chdir('/Users/clairealbrecht/Dropbox/Claire_Dropbox/Data/6MI MNS/CD/20230726')
    # # os.chdir('/Users/calbrecht/Dropbox/Claire_Dropbox/Data/6MI MNS/CD/20230726')
    # # os.chdir(terminalID+'Dropbox/Claire_Dropbox/Data/6MI MNS/CD/20230726')
    # # file_name = 'MNS_4uM_20230726_finescan_10nm_min_smoothed.txt'
    # file_name = 'DNTDP_10perc_20230306_finescan_10nm_min_smoothed.txt'
    
    # # os.chdir('/Users/clairealbrecht/Dropbox/Claire_Dropbox/Data/6MI DNTDP/CD/20230823-juliaCD')
    # # os.chdir('/Users/calbrecht/Dropbox/Claire_Dropbox/Data/6MI DNTDP/CD/20230823-juliaCD')
    # os.chdir(terminalID+'Dropbox/Claire_Dropbox/Data/6MI DNTDP/CD/20230823-juliaCD')
    # file_name = 'DNTDP_10perc_window5mm_pathlength10mm_QS-accum-BS'
    # # os.chdir('/Users/clairealbrecht/Dropbox/Claire_Dropbox/Data/6MI MNS/CD/20230823-juliaCD')
    # # os.chdir('/Users/calbrecht/Dropbox/Claire_Dropbox/Data/6MI MNS/CD/20230823-juliaCD')
    # os.chdir(terminalID+'Dropbox/Claire_Dropbox/Data/6MI MNS/CD/20230823-juliaCD')
    # file_name = 'MNS_fresh_window5mm_pathlength10mm_QS-accum_smoothed'
    # print('NOTE: using data from julias instrument')
    
    files = os.listdir()
    file_name = files[1]
    print('...loading: '+file_name)
    data = np.loadtxt(file_name, skiprows=21)
    # data = np.loadtxt(file_name).T
    

    
    wavelengths = data[:,0]
    # CDSpec = data[:,1] - buf_data[:,1]
    # HT = data[:,2]
    # AbsSpec = data[:,3] - buf_data[:,3]
    # for smoothed data files:
    CDSpec = data[:,2] #- buf_data[:,2]
    AbsSpec = data[:,1] #- buf_data[:,1]
    
    CDSpec = CDSpec - np.mean(CDSpec[:10])
    AbsSpec = AbsSpec - np.mean(AbsSpec[:10])
    print('...forcing CD And ABS to go to zero at ~398-400nm')
    
    
    def moving_average(x, w):
        # return np.convolve(x, np.ones(w), 'valid') / w
        return np.convolve(x, np.ones(w)/w, 'valid')
    
    # window=12
    # print('!! Smoothing over '+str(window)+' data points!!')
    # CDSpec = np.array(moving_average(CDSpec, window))
    # AbsSpec = np.array(moving_average(AbsSpec,window))
    # wavelengths = np.array(moving_average(wavelengths,window))

    AbsSpec = np.vstack([10**7/wavelengths, AbsSpec]).T
    CDSpec = np.vstack([10**7/wavelengths, CDSpec]).T
    
    return [Closeup(AbsSpec, low, high), Closeup(CDSpec, low, high)]

#%%
########### VARIABLE DEFINITIONS ##############
Pi = np.pi

D = 3.33e-30  #C*m per Debye
#D = 2.99e-29 #Debeye per C*m
mumonomer = 12.8 * D  #EDTM for Cy-3
# gamma = 493 #93 # In this version of the code 2gamma=FWHM*)
# low = 16700
# high = 22222 # Data ranges
# low = 17000 
# high = 20000  # Data ranges
# low = 27000
# high = 35000 # Data ranges
low = 25000
high = 32500 # 40000 # Data ranges
low = 10000
high = 36000
c0 = 2.9979e8
H = 6.626e-34
Hbar = 1.054e-34
nubar2nu = 100*c0
permFree = 8.8542e-12

J2nubar = 1/(100*c0*H)

################################################
### Conditional Statements #####################
################################################

chi_int = True
make_plots = True
################################################



################# LINEAR ALGEBRA ################
def kr(a,b): return np.kron(a,b)
def kr4(a,b,c,d): return kr(kr(kr(a,b),c),d)
def dot(a,b): return np.dot(a,b)    

def normalize(v):
    norm = np.linalg.norm(v)
    if norm == 0: 
       return v
    return v / norm

#################################################

########## BROADENING DISTRIBUTIONS #############

def NormalDist(x, mu, sig):
    return ma.exp(-(x - mu)**2/(2 * sig**2))/(ma.sqrt(2*Pi) * sig)

def CauchyDist(x, mu, gam):
    return 1/(Pi * gam * (((x - mu)/gam)**2 + 1))
### HACKED FOR GAUSSIAN
def PseudoVoigtDistribution(x, gamma, sigma, epsilon):
    g = ((gamma**5) + (sigma**5) + (2.69296)*(sigma**4)*(gamma) + \
        2.42843*(sigma**3)*(gamma**2) + 4.47163*(sigma**2)*(gamma**3) + \
                0.07842*(sigma)*(gamma**4))**(1/5)
    eta = gamma/g
    eta = eta * (1.36603 - 0.47719*eta + 0.11116*eta**2)
    return eta * CauchyDist(x, epsilon, g) + (1 - eta) * NormalDist(0, epsilon, g)

cAbsSpectrum, cCDSpectrum = get_c_spectra()
    
def SimData(stickdata, data, gamma, sigma, norm):
    '''simulates a broadened spectra from a stick spectra
    array, array, num, num, num -> array'''
    output = []
    for point in data:
        # print('point: '+str(point))
        simval = 0
        for stick in stickdata:
            # print(stick)
            simval += stick[1]/norm * \
                PseudoVoigtDistribution(point, gamma, sigma, stick[0])
                # PseudoVoigtDistribution(point[0], gamma, sigma, stick[0])
                #CauchyDist(point[0],stick[0],gamma)
                #NormalDist(point[0],stick[0], sigma)
        # output.append([point[0], simval])
        output.append([point, simval])
    return np.array(output)

###################################################

#%%

# phiN,thetaN,rollN, shiftN,shearN,R12ang,sigma,chiAbs,chiCD, epsilon0,omega0,lambdaSq,gamma, nVib = phiN_thetaN_rollN_shiftN_shearN_R12ang_sigma_chiAbs_chiCD
# epsilon0 = 28500/2 #28000
# omega0 = 400
# lambdaSq = lam = 1.54 
# phiN = 77.63
# thetaN = 21.42
# rollN = 0 #104.27
# shiftN = 0
# shearN = 0
# R12ang = 1e10
# sigma = 1000 #316.3164
# chiAbs = 0.022
# chiCD = 0.37574886
# gamma = 800 #1000

# =============================================================================
# define energy parameters (...later optimize)
# =============================================================================
# lam = 5 #3 #1.5 #2 #0.71#1.54 # huang-rhys parameter: displacement of excited state well w.r.t. ground state well
# epsilon0 = 27000 #29300 #29500 # energy gap from ground state (g) to real excited state (f)
# omega0=50#100#80 #160#100 #800/2 # spacing of vibrational levels (set via ground state?)
# omega_ge=epsilon0/2 # virtual state energy spacing
# 20240116: the above set of parameters seems to work but they are all out of wack
# lam = 3 #3 #1.5 #2 #0.71#1.54 # huang-rhys parameter: displacement of excited state well w.r.t. ground state well
# epsilon0 = 28500 #29300 #29500 # energy gap from ground state (g) to real excited state (f)
# omega0=50#100#80 #160#100 #800/2 # spacing of vibrational levels (set via ground state?)
# omega_ge=epsilon0/2 # virtual state energy spacing
# 20240116: the above here are also close...
lam = 2.5 #3 #1.5 #2 #0.71#1.54 # huang-rhys parameter: displacement of excited state well w.r.t. ground state well
epsilon0 = 28600 #29300 #29500 # energy gap from ground state (g) to real excited state (f)
omega0=80#100#80 #160#100 #800/2 # spacing of vibrational levels (set via ground state?)
omega_ge=epsilon0/2 # virtual state energy spacing

# =============================================================================
# set number of electronic and vibrational states in hamiltonian
# =============================================================================
nVib = 2
nEle = 3 # CSA - changed this so that we can vary the # of electronic states
# ...assuming 0=ground state, 1=virtual state, 2=real excited state
# =============================================================================

# some quick conversions... from parameters obtained through orientation calculation (dimer case)
# phi = phiN * (Pi/180)       #
# theta = thetaN * (Pi/180)   
# roll = rollN * (Pi/180)
# shift = shiftN *10**-(10)
# shear = shearN *10**-(10)
# R12 = R12ang  *10**-(10)
# lam = ma.sqrt(lambdaSq)
# writing in J for now... actually comes from orientational parameters in dimer case
J = 0

################## OPERATORS ###################
# electronic raising and lowering operators: c, cD
c = np.zeros((nEle,nEle)) # need to be nxn where n is the number of electronic states
for i in range(nEle-1):
    c[i,i+1] = np.sqrt(i+1)  
cD = c.T    

muOp = cD + c # proportional to position operator (x = sqrt(hbar/ 2m omega) (c + cD))

# Vibrational Modes                                #***#   6
nVib = int(nVib)
# print('...using '+str(nVib)+' vibrational modes')

# electronic raising and lowering operators: b, bD
b = np.zeros((nVib,nVib)) # need to be mxm where m is the number of vibrational states
for i in range(nVib-1):
    b[i,i+1] = np.sqrt(i+1)  # vibrational raising and lowering ops
bD = b.T

# identity ops
Iel = np.eye(nEle)  # g, e, fg,        ( from selection rules model: e1, f0 )
Ivib = np.eye(nVib) # 0, 1             ( from selection rules model: e3, f2 )

# number ops (need to be same size and corresponding identity ops)
cDc = np.dot(cD,c)  # electronic number operator
bDb = np.dot(bD,b)  # vibrational number operator

#################################################



#%
# =============================================================================
# Generate Hamiltonian for monomer A (see eq 17 from Kringel et al)
# =============================================================================
h1A = omega_ge * kr(cDc, Ivib) # electronic levels
h4A = omega0 * kr(Iel, bDb)  # vibrational levels
h6A = omega0 * kr(cDc, lam * (bD + b) + (lam**2)*Ivib) # coupling between electronic and vibrational
hamA = h1A + h4A + h6A
# how to build in selection rules? do we need them?

# create labels for the nEle x nVib matrices
# electronic states: g, e, f      vibrational states: 1,2,3,...
alpha = [['g'] * nVib , ['e']*nVib, ['f'] * nVib] #['ABC', 'DEF', 'GHI', 'JKL']
alpha_tex = []
for i in range(len(alpha)):
    alpha_list = alpha[i]
    for j in range(len(alpha_list)):
        alpha_tex.append(r'$'+alpha_list[j]+'_'+str(j)+'$')
alpha=alpha_tex
        
# matrix_plotter(hamA, alpha, alpha, title=r'Hamiltonian of monomer $(cm^{-1} x10^{3})$',size=nEle*nVib,title_fontsize=20,label_fontsize=16,fontsize=22)#14)
# matrix_plotter(h6A, alpha, alpha, title=r'Hamiltonian of monomer $(cm^{-1} x10^{3})$',size=nEle*nVib,title_fontsize=20,label_fontsize=16,fontsize=22)


# Diagonalize Hamiltonian
epsA, vecsA = la.eig(hamA)


idxA = epsA.argsort()[::-1]   
epsA = epsA[idxA]
vecsA = vecsA[:,idxA]
#print(vecs)           # debugging
epsA = np.flip(epsA, 0)
vecsA = np.fliplr(vecsA)

diag_ham = np.diag(np.real(epsA))
# np.count_nonzero(diag_ham - np.diag(np.diagonal(diag_ham))) # check that it is diagonal matrix

# plot diagonalized hamiltonian
# matrix_plotter(diag_ham, alpha, alpha, title=r'Diagonalized Hamiltonian of monomer $(cm^{-1} x10^{3})$' ,frac=0.8,size=nEle*nVib,title_fontsize=20,label_fontsize=16,fontsize=22)

#%
# =============================================================================
#  look at all possible energy differences given the eigenenergies of hamA
#  need these for the omega21, oemga_32, omega_43 values
# =============================================================================
omegas= np.real(np.subtract.outer(epsA,epsA))
matrix_plotter(omegas, alpha, alpha, title=r'Differences between eigenenergies ($cm^{-1}$ x$10^{3}$)',frac=0.8,size=nEle*nVib,title_fontsize=20,label_fontsize=16,fontsize=22)

omegas_ges = omegas[nVib:nVib*(nEle-1),0:nVib]
omegas_efs = omegas[(nEle-1)*nVib:nEle*nVib, nVib:2*nVib]
omegas_eeps = omegas[nVib:2*nVib, nVib:(nEle-1)*nVib]
omegas_gfs = omegas[(nEle-1)*nVib:nVib*nEle,0:nVib]

alpha_gs = alpha[0:nVib]
alpha_es = alpha[nVib:nVib*(nEle-1)]
alpha_fs = alpha[nVib*(nEle-1):nVib*nEle]
 

# look at sub matrices for transitions
matrix_plotter(omegas_ges, alpha_gs, alpha_es,title=r'Energies for $\Sigma_i \omega_{ge_i}$',frac=0.99,label_fontsize=18)
matrix_plotter(omegas_efs, alpha_es, alpha_fs,title=r'Energies for $\Sigma_{i,j} \omega_{e_if_j}$',frac=0.99,label_fontsize=18)
matrix_plotter(omegas_eeps, alpha_es, alpha_es,title=r"Energies for $\Sigma_{i,j} \omega_{e_ie'_j}$",frac=0.99,label_fontsize=18)
matrix_plotter(omegas_gfs, alpha_gs, alpha_fs,title=r'Energies for $\Sigma_{i} \omega_{gf_i}$',frac=0.99,label_fontsize=18)
#%%
# Calculating pathways for omega 1, 2 3.
omega1 = [omegas_ges[0,0], omegas_ges[1,1],   omega_gep,  omega_ge,   omega_gep,   omega_ge,     omega_ge,   omega_gep]
omega2 = [omegas_gfs[0,0], omega_gfp,   omega_gf,   omega_gf,   omega_gfp,   omega_gfp,    omega_gfp,  omega_gf]        
omega3 = [omegas_efs[1,0], omega_epfp,  omega_ef,   omega_epf,  omega_efp,   omega_epfp,   omega_efp,  omega_epf]
    

#%%
plt.figure(figsize=[3,10]);
plt.scatter(np.ones(len(epsA)),epsA,marker='.')
plt.figure();
plt.scatter(np.ones(len(omegas_efs[::2,1::2].flatten()))+1,omegas_efs[::2,1::2].flatten(),color='r',marker='o'); 
# plt.scatter(np.ones(len(omegas_ges[1::2,::2].flatten())),omegas_ges[1::2,::2].flatten(),color='b',marker='o');
plt.scatter(np.ones(len(omegas_ges[1::2,0].flatten())),omegas_ges[1::2,0].flatten(),color='b',marker='o');
plt.xlim(0,3)
#%%
# =============================================================================
# Impose selection rules by only allowing even g's, odd e's and even f's
# =============================================================================
# 20240103 CSA: the shift imposed by this method doesn't seem to look right... I am missing something. It pushes the peaks too low.
# need omegas_efs to be shifted down by omega0 (see non-selection rule attempt below this section), not by omegas_gep - omega_ge... 
# what determines this spacing?
# also with this method the 'middle square' is not the same proportions as the top and bottom squares, why is this? What is causing this?
# =============================================================================

omegas= np.real(np.subtract.outer(epsA,epsA))
omegas_ges = omegas[nVib:nVib*(nEle-1),0:nVib]
omegas_ges[:,1::2] = np.zeros(omegas_ges[:,1::2].shape) # replace odd columns with zeros
omegas_ges[::2,:] = np.zeros(omegas_ges[::2,:].shape) # replace even rows with zeros
# check with matrix plotter that this is doing what I want it to
# matrix_plotter(omegas_ges, alpha_gs, alpha_es,title=r'Energies for $\Sigma_i \omega_{ge_i}$',frac=0.89,label_fontsize=18)

omegas_ges = omegas_ges[1::2,::2] # only select the omegas_ges that we want
# matrix_plotter(omegas_ges, alpha_gs[::2], alpha_es[1::2],title=r'Energies for $\Sigma_i \omega_{ge_i}$',frac=0.99,label_fontsize=18)

omegas_ges = omegas_ges[:,0].reshape(omegas_ges[:,0].shape[0],1) # we actually dont want any g other than g0
# matrix_plotter(omegas_ges, [alpha_gs[0]], np.array(alpha_es[1::2]),title=r'Energies for $\Sigma_i \omega_{ge_i}$',frac=0.99,label_fontsize=18)

#%
# repeat for omegas_efs
omegas= np.real(np.subtract.outer(epsA,epsA))
omegas_efs = omegas[(nEle-1)*nVib:nEle*nVib, nVib:2*nVib]
# matrix_plotter(omegas_efs, alpha_es, alpha_fs,title=r'Energies for $\Sigma_i \omega_{ge_i}$',frac=0.96,label_fontsize=18)
# omegas_efs[1::2,::2] = np.zeros(omegas_efs[1::2,::2].shape) # replace odd columns with zeros
omegas_efs[:,::2] = np.zeros(omegas_efs[:,::2].shape) # replace even columns with zeros
omegas_efs[1::2,:] = np.zeros(omegas_efs[1::2,:].shape) # replace odd rows with zeros
# check with matrix plotter that this is doing what I want it to
# matrix_plotter(omegas_efs, alpha_es, alpha_fs,title=r'Energies for $\Sigma_i \omega_{ge_i}$',frac=0.89,label_fontsize=18)

omegas_efs = omegas_efs[::2,1::2] # select omegas_efs that we want
# matrix_plotter(omegas_efs, alpha_es[1::2], alpha_fs[::2],title=r'Energies for $\Sigma_{i,j} \omega_{e_if_j}$',frac=0.99,label_fontsize=18)

# =============================================================================
# what selection rules do we need for eeps???
# omegas_eeps = 
# matrix_plotter(omegas_eeps, alpha_es,alpha_es,title='omegas_eeps',frac=0.99)



#%
# =============================================================================
# =============================================================================
# # IF NO SELECTION RULES, USE THE FOLLOWING TO TEST SHIFT OFF DIAGONAL
# =============================================================================
# omegas_efs = omegas_efs - omega0 #200
# omegas_ges = omegas_ges[:,0]
# looks like data with the following params:
    # nVib = 2
    # lam = 1.5 
    # epsilon0 = 29300 
    # omega0=160
# =============================================================================
# =============================================================================
# =============================================================================
#%%
# flatten the 2x2 arrays so we can plot scatter
omegas_ges = omegas_ges.flatten()
omegas_efs = omegas_efs.flatten()
omegas_eeps = omegas_eeps.flatten()
omegas_gfs = omegas_gfs.flatten()



# =============================================================================
# % let's look at all combinations of omegas_ges on xaxis and omegas_efs on the yaxis 
#  (peaks for the t32 = 0 experiment should be close to these locations)
# =============================================================================
# omega_ges_arr = np.tile(omegas_ges,len(omegas_ges)).reshape(len(omegas_ges),len(omegas_ges))
omega_ges_arr = np.tile(np.tile(omegas_ges,int(len(omegas_efs)/len(omegas_ges))),len(omegas_efs)).reshape(len(omegas_efs),len(omegas_efs))
omega_efs_arr = np.tile(omegas_efs,len(omegas_efs)).reshape(len(omegas_efs),len(omegas_efs))


omega_ges_arr = omega_ges_arr.T # transpose energies for the xaxis
# omega_efs_arr = omega_efs_arr.T # transpose energies for the yaxis
# if you DON'T transpose you get a diamond, if you do you get a square... why? which is right?
# need a transpose to get all the combinations...

w,h = plt.figaspect(1.)
fig = plt.figure(figsize=[w,h])
ax = fig.add_subplot(111)
ax.scatter(omega_ges_arr/1e3,omega_efs_arr/1e3)
i = 0 #np.arange(0,4)
j = 3 #np.arange(0,4) #1
print('omega_ges:' + str(omega_ges_arr[i,j]))
print('oemga_efs:' + str(omega_efs_arr[i,j]))
# print((omega_ges_arr.T)[2,:])
# print(omega_efs_arr[:,2])
# ax.scatter((omega_ges_arr.T)[i,:]/1e3,omega_efs_arr[:,i]/1e3,color='r',marker='x')
ax.scatter(omega_ges_arr[i,j]/1e3,omega_efs_arr[i,j]/1e3,color='r',marker='x')
ax.plot(np.arange(14200/1e3,16300/1e3),np.arange(14200/1e3,16300/1e3),'k--')
# ax.set_xlim(14100/1e3, 16300/1e3)
# ax.set_ylim(14100/1e3, 16300/1e3)
ax.set_xlim(13800/1e3, 16800/1e3)
ax.set_ylim(13800/1e3, 16800/1e3)
# ax.set_xlim(12800/1e3, 18800/1e3)
# ax.set_ylim(12800/1e3, 18800/1e3)
# plt.hlines(10**7/675,min(omegas_ges_arr[0,:]),max(omegas_ges_arr[0,:]),color='k',linestyle='--')
# plt.vlines(10**7/675,min(omegas_efs_arr[0,:]),max(omegas_efs_arr[0,:]),color='k',linestyle='--')
laser_loc = [10**7/675/1e3, 10**7/675/1e3]
ax.scatter(laser_loc[0],laser_loc[1],color='k',marker='x')
circle=plt.Circle((laser_loc[0],laser_loc[1]),(10**7/(675-15) - 10**7/(675+15))/1e3,color='r',alpha=0.1,clip_on=True)
ax.add_patch(circle);
ax.set_xlabel(r'$\omega_{ge}$',fontsize=14)
ax.set_ylabel(r'$\omega_{ef}$',fontsize=14)




#%% try the same thing for the t21= 0 experiment?
omega_eeps_arr = np.tile(omegas_eeps,len(omegas_eeps)).reshape(len(omegas_eeps),len(omegas_eeps))
omega_gfs_arr = np.tile(omegas_gfs, len(omegas_gfs)).reshape(len(omegas_gfs),len(omegas_gfs))
w,h = plt.figaspect(1.)
fig = plt.figure(figsize=[w,h])
ax = fig.add_subplot(111)
omega_eeps_arr = omega_eeps_arr.T # transpose energies for the xaxis
ax.scatter(omega_eeps_arr/1e3,omega_gfs_arr/1e3)
i = 3 #np.arange(0,4)
j = 3 #np.arange(0,4) #1
print('omega_eeps:' + str(omega_eeps_arr[i,j]))
print('oemga_gfs:' + str(omega_gfs_arr[i,j]))
# print((omega_ges_arr.T)[2,:])
# print(omega_efs_arr[:,2])
# ax.scatter((omega_ges_arr.T)[i,:]/1e3,omega_efs_arr[:,i]/1e3,color='r',marker='x')
ax.scatter(omega_eeps_arr[i,j]/1e3,omega_gfs_arr[i,j]/1e3,color='r',marker='x')
# ax.plot(np.arange(14200/1e3,16300/1e3),np.arange(14200/1e3,16300/1e3),'k--')
# ax.set_xlim(14100/1e3, 16300/1e3)
# ax.set_ylim(14100/1e3, 16300/1e3)
# plt.hlines(10**7/675,min(omegas_ges_arr[0,:]),max(omegas_ges_arr[0,:]),color='k',linestyle='--')
# plt.vlines(10**7/675,min(omegas_efs_arr[0,:]),max(omegas_efs_arr[0,:]),color='k',linestyle='--')
# laser_loc = [10**7/675/1e3, 10**7/675/1e3]
# ax.scatter(laser_loc[0],laser_loc[1],color='k',marker='x')
# circle=plt.Circle((laser_loc[0],laser_loc[1]),(10**7/(675-15) - 10**7/(675+15))/1e3,color='r',alpha=0.1,clip_on=True)
# ax.add_patch(circle);
ax.set_xlabel(r"$\omega_{ee'}$",fontsize=14)
ax.set_ylabel(r'$\omega_{gf}$',fontsize=14)



# =============================================================================
# %% Simulate absorption & CD spectra from these energies
# =============================================================================
# =============================================================================
# # How to calculate rotational strength for monomer??
# =============================================================================
# define dipole vector for monomer
muA = np.array([0.1,0.2,0.3]) # what should this be??
R12 = 1 # what should this be?


# muTot = np.array([muA[i]*kr(muOp, Ivib) for i in range(3)]) # from above: muOp = cD + c
# matrix_plotter(omega0*bDb, alpha, alpha, title='w/vibrational levels ',frac=0.96)                                                 # 3 because muA is a 3D vector
muTot = np.array([muA[i]*kr(muOp, omega0*bDb) for i in range(3)]) # from above: muOp = cD + c
# using omega0*bDb to try to make muTot a function of the vibrational level (instead of Ivib where it is independent of vib level)
# matrix_plotter(muTot[2,:,:], alpha, alpha, title='muTot',frac=0.96)

# I don't think I'll need R or R12 for the monomer case, but leaving it here for now until I know for sure
unitR = np.array([0, 0, 1])
Rvec = R12*unitR

# magVecA = np.cross(unitR, muA) # this is currently perpendicular to R and mu...
# 20240116: Need a new way to define magVecA so the dot product between magVecA and muA is nonzero
magVecA = np.array([-0.1,0.2,0.3]) # for not set some arbitrary magVec so that np.dot(magVecA,muA) =/= 0
# magVecB = np.cross(-unitR, muB) 

magA = [magVecA[i]*muOp for i in (0,1,2)] # shape: (3,3,3), muOp = c + cd proportional to x operator
# magB = [magVecB[i]*muOp for i in (0,1,2)]
magA = np.array(magA)

# magB = magA # making monomers identical!

# how to make the vibrational levels optically active?
# op1 = kr(magA[0], Ivib) + kr(Iel,Ivib) # CSA - do i need the second term?
# op2 = kr(magA[1], Ivib) + kr(Iel,Ivib)
# op3 = kr(magA[2], Ivib) + kr(Iel,Ivib)
op1 = kr(magA[0], omega0*bDb) + kr(Iel,Ivib)  #* # 20240116: try using bDb*omega0 instead of identity operator to make the operator depend on vibrational level
op2 = kr(magA[1], omega0*bDb) + kr(Iel,Ivib) #omega0*
op3 = kr(magA[2], omega0*bDb) + kr(Iel,Ivib) #omega0*
# op1 = kr4(magA[0], Iel, Ivib, Ivib) + kr4(Iel, magB[0], Ivib, Ivib)
# op2 = kr4(magA[1], Iel, Ivib, Ivib) + kr4(Iel, magB[1], Ivib, Ivib)
# op3 = kr4(magA[2], Iel, Ivib, Ivib) + kr4(Iel, magB[2], Ivib, Ivib)

  
magOps = [op1, op2, op3]

mVecA = muA*mumonomer  # electric dipole moment scaled by correct units (mumonomer)
# mVecB = muB*mumonomer

# how to define rotational strength for monomer?  hertzberg-teller??
# look at eqs on 2491 of Nooijen_Int J of Quantum Chemistry_2006.pdf
# RS = (H*nubar2nu*epsilon0/(4*Hbar)) * dot(np.cross(mVecA, mVecB), Rvec)
RS = (H*nubar2nu*epsilon0/(4*Hbar)) * dot(magA, mVecA) # this is a 3x3 matrix...
# 20240116: actually a rotational strength for each transition maybe makes sense? 
# RS = (H*nubar2nu*epsilon0/(4*Hbar)) * dot(magVecA, mVecA) # should be a number? 
# check constants out front... I am using a different form of Rosenfeld equation than is used for the dimer

Area = RS*epsilon0*nubar2nu/(7.659e-54)
sigma=100 # inhomogenous linewidth (placeholder for now)
Height = Area/(sigma * ma.sqrt(2 * Pi) * nubar2nu) / 2 

#%%

eps, vecs = epsA, vecsA

# absorbtion intensities
# Ix = dot(muTot[0], vecs)[0] # why do we need the eigenvectors here? What is this doing for us physically?
# Iy = dot(muTot[1], vecs)[0] # are the eigenvectors setting up the collective modes?
# Iz = dot(muTot[2], vecs)[0]
Ix = dot(muTot[0], vecs)#[0]
Iy = dot(muTot[1], vecs)#[0]
Iz = dot(muTot[2], vecs)#[0]
SimI = (Ix**2 + Iy**2 + Iz**2)*(2/3)
# does this still make sense? I guess since we are assuming e||f that maybe this is ok?

AbsData = np.transpose([eps, SimI]) # simulated absorption

# extend range of data to cover virtual states for the time being
xvals2append = np.linspace(0,np.min(cAbsSpectrum[:,0]),num=int(np.ceil(np.min(cAbsSpectrum[:,0])/(cAbsSpectrum[1,0]-cAbsSpectrum[0,0]))))
zeros2append = np.zeros(xvals2append.shape)
cAbs2append = np.vstack([xvals2append, zeros2append]).T
cAbsSpectrum = np.vstack([cAbs2append, cAbsSpectrum])

gamma=40 # homogeneous linewidth (placeholder for now)
normAbs = PseudoVoigtDistribution(epsilon0, gamma, sigma, epsilon0)
simAbs = SimData(AbsData, cAbsSpectrum, gamma, sigma, normAbs*1.1)

plt.figure()
plt.scatter(AbsData[:,0],AbsData[:,1])
# plt.plot(cAbsSpectrum[:,0], cAbsSpectrum[:,1])
plt.plot(simAbs[:,0],simAbs[:,1],'r')
plt.xlim(10000,max(cAbsSpectrum[:,0]))
plt.xlim(14000,16000)


# CD intensities
cdk1 = - dot(muTot[0], vecs)[0] * dot(magOps[0], vecs)[0]
cdk2 = - dot(muTot[1], vecs)[0] * dot(magOps[1], vecs)[0]
cdk3 = - dot(muTot[2], vecs)[0] * dot(magOps[2], vecs)[0]

cdTot = Height*(cdk1 + cdk2 + cdk3)
# np.set_printoptions(threshold=1000)

CDdata = np.transpose([eps, cdTot])# simulated CD

xvals2append = np.linspace(0,np.min(cCDSpectrum[:,0]),num=int(np.ceil(np.min(cCDSpectrum[:,0])/(cCDSpectrum[1,0]-cCDSpectrum[0,0]))))
zeros2append = np.zeros(xvals2append.shape)
cCD2append = np.vstack([xvals2append, zeros2append]).T
cCDSpectrum = np.vstack([cCD2append, cCDSpectrum])

simCD = SimData(CDdata, cCDSpectrum, gamma, sigma, normAbs)

plt.figure()
plt.scatter(CDdata[:,0],CDdata[:,1])
# plt.plot(cCDSpectrum[:,0], cCDSpectrum[:,1])
plt.plot(simCD[:,0],simCD[:,1],'r')
plt.xlim(10000,max(cCDSpectrum[:,0]))

#%%

# Start by trying to define mu_if and mag_fi as a function of vibrational level
muA = np.array([0.1,0.2,0.3]) # orientation of electric dipole in lab space (will integrate over angles later)
muA = muA / np.sqrt(muA[0]**2 + muA[1]**2 + muA[2]**2)
muA = omega0 * kr(muA, bDb) # each direction has vibrational levels... is this what I want?
plt.matshow(muA)

# 20240116: project this dipole onto the vibrational states (used to use Ivib instead of bDb)
# muTot = np.array([muA[i]*kr(muOp, 10*omega0*bDb) for i in range(3)]) ; print('multiplying omega0 by 10 for now')
muTot = np.array([muA[i]*kr(muOp, bDb) for i in range(3)])  
matrix_plotter(100*muTot[2],alpha,alpha,title='muTot',frac=0.99); print('multiplying omega0 by 100 for now')

magVecA = np.array([-0.1,0.2,0.3]) # for now set some arbitrary magVec so that np.dot(magVecA,muA) =/= 0
# magVecB = np.cross(-unitR, muB) 

magA = [magVecA[i]*muOp for i in (0,1,2)] # shape: (3,3,3), muOp = c + cd proportional to x operator
# magB = [magVecB[i]*muOp for i in (0,1,2)]
magA = np.array(magA)






#%%
# =============================================================================
# generate the full system hamiltonian (dimer)
# =============================================================================
# # Look at eq 17 & 18 in Kringle et al. 2018
# h1 = epsilon0*kr4(cDc, Iel, Ivib, Ivib)  # H_A: term 1 (electronic excitation)
# h2 = epsilon0*kr4(Iel, cDc, Ivib, Ivib) # H_B: term 1  (electronic excitation)
# h3 = J*kr(kr(kr(cD, c) + kr(c, cD), Ivib), Ivib) # H_coupling (eq 18)
# h4 = omega0*kr4(Iel, Iel, bDb, Ivib) # H_A: term 2  (vibrational excitation)
# h5 = omega0*kr4(Iel, Iel, Ivib, bDb) # H_B: term 2 (vibrational excitation)
# h6 = omega0*kr4(cDc, Iel, lam * (bD + b) + (lam**2) * Ivib, Ivib) # H_A: term 3 (vibronic coupling)
# h7 = omega0*kr4(Iel, cDc, Ivib, lam * (bD + b) + (lam**2 * Ivib)) # H_B: term 3 (vibronic coupling)
# ham = h1 + h2 + h3 + h4 + h5 + h6 + h7
# plt.matshow(ham)

# # Diagonalize Hamiltonian
# eps, vecs = la.eig(ham)
    
# idx = eps.argsort()[::-1]   
# eps = eps[idx]
# vecs = vecs[:,idx]
# #print(vecs)           # debugging
# eps = np.flip(eps, 0)
# vecs = np.fliplr(vecs)

# # ele_state_labels = []
# # for i in range(nEle):
# #     ele_state_labels.append([i]*nVib)
# # ele_state_labels = np.array(ele_state_labels).flatten()

# plt.figure()
# plt.scatter(np.arange(len(eps)),eps/epsilon0)
# plt.ylabel(r'$\lambda (E)$ / $\epsilon_0$')
# # plt.scatter(np.array([0,0,1,1,2,2]),epsA)#/epsilon0)
# # plt.scatter(np.array([0,0,1,1,2,2]),eps/epsilon0) # eigenvalues normalized by F0 energy



