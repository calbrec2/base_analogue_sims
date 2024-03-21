#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  7 10:09:38 2024

@author: calbrec2
"""

# Updated version from hamiltonian_fot_6MI_v4.py
# trying to streamline the hamiltonian calculation so we can optimize to Abs/CD and 2PE-2DFS data
# getting rid of stuff not used...

import scipy as sp
from scipy.optimize import minimize, minimize_scalar, brute , differential_evolution, fmin, basinhopping

#import ext_dip as ed  # This file contains the definitions for 
                      #  the dipole orientation angles and the
                      #  extended dipole J calculation
import os 
print(os.getcwd())

import numpy as np
import math as ma
import matplotlib.pyplot as plt
import scipy.linalg as la
import pandas as pd

# Time/date stamp runs:
from datetime import datetime as dt
import time

# Stop Warning Messages
import warnings
warnings.filterwarnings('ignore')

import os 
cwd = os.path.dirname(os.path.realpath(__file__))
print(cwd)

# import data .mat file
import os
import glob

import scipy.optimize as opt
#%% Matrix plotter & grab data
###############################################
# Plotting function for looking at matrices
def matrix_plotter(omegas, alpha_x, alpha_y, title, frac=0.8, figsize=[6,6], fontsize=14,title_fontsize=18,label_fontsize=12):
    fig = plt.figure(figsize=figsize)
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
    
    terminalID = '/Users/calbrec2/'
    
    # os.chdir('/Users/clairealbrecht/Dropbox/MATLAB_programs/claire_programs/from_Lulu/20230726')
    # os.chdir('/Users/calbrecht/Dropbox/MATLAB_programs/claire_programs/from_Lulu/20230726')
    # os.chdir(terminalID+'Dropbox/MATLAB_programs/claire_programs/from_Lulu/20230726')
    # file_name = 'DNTDP_10perc_20230306_finescan_10nm_min.txt'
    # file_name = '20230726_MNS_4uM_20230726_finescan.txt'
    
    
    import os.path
    from pathlib import Path

    home = str(Path.home())
    path = os.path.join(home,'Documents','github_base','Data','CD_Abs','20230726')
    os.chdir(path)
    
    buf_file_name = '20230726_buffer2.txt'
    buf_data = np.loadtxt(buf_file_name, skiprows=21)
    file_name = '20230726_MNS_4uM_20230726_finescan.txt'
    # file_name = 'DNTDP_10perc_20230306_finescan_10nm_min.txt'
    # file_name = 'MNS_4uM_20230726_finescan_10nm_min_smoothed.txt'
    # file_name = 'DNTDP_10perc_20230306_finescan_10nm_min_smoothed.txt'   
    
    
    files = os.listdir()
    # file_name = files[1]
    print('...loading: '+file_name)
    data = np.loadtxt(file_name, skiprows=21)
    # data = np.loadtxt(file_name).T
    

    
    wavelengths = data[:,0]
    extra_buf_scaling = 1.18 # for MNS... need different for DNTDP
    # extra_buf_scaling = 1.1 # for DNTDP
    CDSpec = data[:,1] - extra_buf_scaling*buf_data[:,1]
    AbsSpec = data[:,3] - extra_buf_scaling*buf_data[:,3]
    
    CDSpec = CDSpec - np.mean(CDSpec[:10])
    AbsSpec = AbsSpec - np.mean(AbsSpec[:10])
    print('...forcing CD And ABS to go to zero at ~398-400nm')
    
    
    def moving_average(x, w):
        # return np.convolve(x, np.ones(w), 'valid') / w
        return np.convolve(x, np.ones(w)/w, 'valid')
    
    window = 12
    print('!! Smoothing over '+str(window)+' data points!!')
    CDSpec = np.array(moving_average(CDSpec, window))
    AbsSpec = np.array(moving_average(AbsSpec,window))
    wavelengths = np.array(moving_average(wavelengths,window))

    AbsSpec = np.vstack([10**7/wavelengths, AbsSpec]).T
    CDSpec = np.vstack([10**7/wavelengths, CDSpec]).T
    
    low = 25000
    high = 36000#2500 
    
    return [Closeup(AbsSpec, low, high), Closeup(CDSpec, low, high)]


cAbsSpectrum, cCDSpectrum = get_c_spectra()
# fig,ax = plt.subplots(2,1,figsize=[10,8],sharex=True)
# ax[0].plot(cAbsSpectrum[:,0],cAbsSpectrum[:,1])
# ax[0].axhline(0,color='k',linestyle='--')
# ax[0].set_xlim(25500,36000)
# ax[1].plot(cCDSpectrum[:,0],cCDSpectrum[:,1])
# ax[1].axhline(0,color='k',linestyle='--')
#%% variables and simple definitions
########### VARIABLE DEFINITIONS ##############
Pi = np.pi

D = 3.33e-30  #C*m per Debye
#D = 2.99e-29 #Debeye per C*m
mumonomer = 12.8 * D  #EDTM for Cy-3
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
def kr(a,b): return sp.kron(a,b)
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
                PseudoVoigtDistribution(point[0], gamma, sigma, stick[0])
                # PseudoVoigtDistribution(point, gamma, sigma, stick[0])
                #CauchyDist(point[0],stick[0],gamma)
                #NormalDist(point[0],stick[0], sigma)
        output.append([point[0], simval])
        # output.append([point, simval])
    return np.array(output)

###################################################
#%% Set up state space and operators

# =============================================================================
# define energy parameters (...later optimize)
# =============================================================================
lam = 2.5           # huang-rhys parameter: displacement of excited state well w.r.t. ground state well
epsilon0 = 29000    # energy gap from ground state (g) to real excited state (f)
omega0=180          # spacing of vibrational levels 
# epsilon0/2 # virtual state energy spacing... this is defined in the 2PE hamiltonian

# =============================================================================
# set number of electronic and vibrational states in hamiltonian
# =============================================================================
nVib = 4 #6 #4 
nEle = 3 # 0 = ground state, 1 = virtual state, 2 = real excited state
# create labels for the nEle x nVib matrices
# electronic states: g, e, f      vibrational states: 1,2,3,...
alpha = [['g'] * nVib , ['e']*nVib, ['f'] * nVib] 
alpha_tex = []
for i in range(len(alpha)):
    alpha_list = alpha[i]
    for j in range(len(alpha_list)):
        alpha_tex.append(r'$'+alpha_list[j]+'_'+str(j)+'$')
alpha=alpha_tex
# =============================================================================
# writing in J for now... actually comes from orientational parameters in dimer case
J = 0

################## OPERATORS ###################
# ----------- Electronic operators ----------- #
#### For 2PE ####
# electronic raising and lowering operators: c, cD
c_2PE = sp.zeros((nEle,nEle))       # nxn where n is the number of electronic states
for i in range(nEle-1):
    c_2PE[i,i+1] = sp.sqrt(i+1)     # 2PE requires you to go through the e-states
cD_2PE = c_2PE.T    
cDc_2PE = sp.dot(cD_2PE,c_2PE)      # electronic number operator
# number ops (need to be same size as corresponding identity ops)
muOp_2PE = cD_2PE + c_2PE           # proportional to position operator (x = sqrt(hbar/ 2m omega) (c + cD))
#################
# matrix_plotter(cDc_2PE*1e3,alpha_x=['g','e','f'],alpha_y=['g','e','f'], title='ladder operator')
# matrix_plotter(cD_2PE*1e3,alpha_x=['g','e','f'],alpha_y=['g','e','f'], title='cD')
# matrix_plotter(c_2PE*1e3,alpha_x=['g','e','f'],alpha_y=['g','e','f'], title='c')
# matrix_plotter(muOp_2PE*1e3,alpha_x=['g','e','f'],alpha_y=['g','e','f'], title='muOp')

#### For 1PE ####
# electronic raising and lowering operators: c, cD
c_1PE = sp.zeros((nEle,nEle))       # nxn where n is the number of electronic states
c_1PE[0,nEle-1] = 1                 # 1PE takes you all the way to the f-state
cD_1PE = c_1PE.T
cDc_1PE = sp.dot(cD_1PE,c_1PE)      # electronic number operator
# number ops (need to be same size as corresponding identity ops)
muOp_1PE = cD_1PE + c_1PE           # proportional to position operator (x = sqrt(hbar/ 2m omega) (c + cD))
#################
# matrix_plotter(cDc_1PE*1e3,alpha_x=['g','e','f'],alpha_y=['g','e','f'], title='ladder operator')

# ----------- Vibrational operators ----------- # Vibrational Modes   #***#   6 (for Cy3)
# vibrational raising and lowering operators: b, bD
b = sp.zeros((nVib,nVib)) # need to be mxm where m is the number of vibrational states
for i in range(nVib-1):
    b[i,i+1] = sp.sqrt(i+1)  # vibrational raising and lowering ops
bD = b.T

# number ops (need to be same size and corresponding identity ops)
bDb = sp.dot(bD,b)  # vibrational number operator

# matrix_plotter(b*1e3,alpha_x=['n0','n1','n2','n3'],alpha_y=['n0','n1','n2','n3'], title='b')
# matrix_plotter(bD*1e3,alpha_x=['n0','n1','n2','n3'],alpha_y=['n0','n1','n2','n3'], title='bdD')
# matrix_plotter(bDb*1e3,alpha_x=['n0','n1','n2','n3'],alpha_y=['n0','n1','n2','n3'], title='c')

# identity ops
Iel = sp.eye(nEle)  # g, e, fg,        ( from selection rules model: e1, f0 )
Ivib = sp.eye(nVib) # 0, 1             ( from selection rules model: e3, f2 )
# matrix_plotter(Iel*1e3,alpha_x=['g','e','f'],alpha_y=['g','e','f'], title='Iele')
# matrix_plotter(Ivib*1e3,alpha_x=['n0','n1','n2','n3'],alpha_y=['n0','n1','n2','n3'], title='Iele')

#################################################

#%% Hamiltonians
# =============================================================================
# Generate Hamiltonian for monomer A (see eq 17 from Kringle et al)
# =============================================================================
def Ham_2PE(epsilon0, omega0, lam, plot_mode=0):
    # ----------- 2PE Hamiltonian ----------- #
    h1 = (epsilon0/2) * kr(cDc_2PE, Ivib)   # electronic levels
    h4 = omega0 * kr(Iel, bDb)              # vibrational levels
    h6 = omega0 * kr(cDc_2PE, lam * (bD + b) + (lam**2)*Ivib) # coupling between electronic and vibrational
    ham_2PE = h1 + h4 + h6
    if plot_mode == 1:
        # matrix_plotter(ham_2PE, alpha, alpha, title=r'Hamiltonian of monomer $(cm^{-1} x10^{3})$',size=nEle*nVib,title_fontsize=20,label_fontsize=16,fontsize=22)#14)
        matrix_plotter(ham_2PE, alpha, alpha, title=r'Hamiltonian of monomer $(cm^{-1} x10^{3})$',frac=0.8,figsize=[nEle*nVib,nEle*nVib],title_fontsize=20,label_fontsize=16,fontsize=22)#14)
    
    # Diagonalize Hamiltonian
    eigs_2PE, vecs_2PE = la.eig(ham_2PE)
    # sort eigenvaluess and maintain corresponding order of eigenvectors
    idxA = eigs_2PE.argsort()[::-1]   
    eigs_2PE = eigs_2PE[idxA]
    vecs_2PE = vecs_2PE[:,idxA]
    eigs_2PE = np.flip(eigs_2PE, 0)
    vecs_2PE = np.fliplr(vecs_2PE)
    
    diag_ham_2PE = np.diag(np.real(eigs_2PE))
    # plot diagonalized hamiltonian
    if plot_mode == 1:
        matrix_plotter(diag_ham_2PE, alpha, alpha, title=r'Diagonalized Hamiltonian of monomer $(cm^{-1} x10^{3})$' ,frac=0.8,figsize=[nEle*nVib,nEle*nVib],title_fontsize=20,label_fontsize=16,fontsize=22)
    return eigs_2PE, vecs_2PE#, ham_2PE #, diag_ham_2PE
def Ham_1PE(epsilon0, omega0, lam, plot_mode =0):
    # ----------- 1PE Hamiltonian ----------- #
    h1 = epsilon0 * kr(cDc_1PE, Ivib)       # electronic levels
    h4 = omega0 * kr(Iel, bDb)              # vibrational levels
    h6 = omega0 * kr(cDc_1PE, lam * (bD + b) + (lam**2)*Ivib) # coupling between electronic and vibrational
    ham_1PE = h1 + h4 + h6
    if plot_mode == 1:
        matrix_plotter(ham_1PE, alpha, alpha, title=r'Hamiltonian of monomer $(cm^{-1} x10^{3})$',size=nEle*nVib,title_fontsize=20,label_fontsize=16,fontsize=22)#14)
    
    # Diagonalize Hamiltonian
    eigs_1PE, vecs_1PE = la.eig(ham_1PE)
    # sort eigenvaluess and maintain corresponding order of eigenvectors
    idxA = eigs_1PE.argsort()[::-1]   
    eigs_1PE = eigs_1PE[idxA]
    vecs_1PE = vecs_1PE[:,idxA]
    eigs_1PE = np.flip(eigs_1PE, 0)
    vecs_1PE = np.fliplr(vecs_1PE)
    
    diag_ham_1PE = np.diag(np.real(eigs_1PE))
    # plot diagonalized hamiltonian
    if plot_mode == 1:
        matrix_plotter(diag_ham_1PE, alpha, alpha, title=r'Diagonalized Hamiltonian of monomer $(cm^{-1} x10^{3})$' ,frac=0.8,size=nEle*nVib,title_fontsize=20,label_fontsize=16,fontsize=22)
    return eigs_1PE, vecs_1PE #, ham_1PE, diag_ham_1PE
# =============================================================================
#%% #### Calculate absorption ####
def calc_Abs(vecs_1PE, eigs_1PE, sigma=20, gamma=200, plot_mode=0):
    #  ----------- dipole operators  ----------- #
    muA = np.array([-0.1,0.2,-0.2]) # some dipole orientation... do I need a orientational average?
    muA /= np.linalg.norm(muA)
    # muTot_2PE = np.array([muA[i]*kr(muOp_2PE, Ivib) for i in range(3)]) # from above: muOp = cD + c ~ x operator... 
    muTot_1PE = np.array([muA[i]*kr(muOp_1PE,Ivib) for i in range(3)])
    # =============================================================================
    # Calculate absorbtion intensities for 1PE
    # =============================================================================
    Ix = dot(muTot_1PE[0], vecs_1PE)[0] # I think the eigenvectors are setting up the collective modes 
    Iy = dot(muTot_1PE[1], vecs_1PE)[0] # select 0th row so we overlap <n|0> the nth vibrational state with the ground vibrational (first row)
    Iz = dot(muTot_1PE[2], vecs_1PE)[0] # using eigenvecs from 1PE because this is linear abs
    SimI = (Ix**2 + Iy**2 + Iz**2)*(2/3)
    
    AbsData = np.transpose([eigs_1PE, SimI]) # simulated 1PE absorption
    
    # RS = (H*nubar2nu*epsilon0/(4*Hbar))  # this is going to just be a scaling factor now
    # # check constants out front... I am using a different form of Rosenfeld equation than is used for the dimer
    
    # Area = RS*epsilon0*nubar2nu/(7.659e-54)
    # Height = 1 #Area/(sigma * ma.sqrt(2 * Pi) * nubar2nu) / 2 
    
    # sigma=20 # inhomogenous linewidth (placeholder for now)
    # gamma=200 # homogeneous linewidth (placeholder for now)
    normAbs = PseudoVoigtDistribution(epsilon0, gamma, sigma, epsilon0)
    simAbs = SimData(AbsData, cAbsSpectrum, gamma, sigma, normAbs) # *1.1
    simAbs[:,1] = (simAbs[:,1]/np.max(simAbs[:,1])) * np.max(np.abs(AbsData[:,1]))
    # normalizing abs for now... sort out units later
    
    def gauss(x, lam1, sig1, amp1):
        return amp1 * np.exp(-(x-lam1)**2 / (2 *sig1**2))
        
    # plot CD and Abs
    if plot_mode == 1:
        fig,ax = plt.subplots(2,1,figsize=[10,8],sharex=True)
        ax[0].plot(cAbsSpectrum[:,0], (cAbsSpectrum[:,1]/np.max(cAbsSpectrum[:,1]))*np.max(AbsData[:,1]),'k',linewidth=2.5)
        ax[0].scatter(AbsData[:,0],AbsData[:,1],color='b')
        ax[0].plot(simAbs[:,0],simAbs[:,1],'r',linewidth=2.5)
        laser_mu = 675; laser_fwhm = 30
        laser_fwhm = 10**7/(675-(laser_fwhm/2)) - 10**7/(675+(laser_fwhm/2))
        laser_sig =  laser_fwhm / (2 * np.sqrt(2 * np.log(2)))
        ax[0].fill_between(simAbs[:,0],gauss(simAbs[:,0], 10**7/(laser_mu/2),laser_sig,np.max(AbsData[:,1])),color='gray',alpha=0.25)
        ax[0].axhline(0,color='k',linestyle='--')
        ax[0].set_title('Abs sim w/ exp Abs',fontsize=14)
        # plt.set_xlim(10000,max(cAbsSpectrum[:,0]))
        ax[0].set_xlim(27500,32000)
        
        ax[1].plot(cCDSpectrum[:,0],(cCDSpectrum[:,1]/np.max(np.abs(cCDSpectrum[:,1]))) * np.max(np.abs(AbsData[:,1])),'k',linewidth=2.5)
        ax[1].scatter(AbsData[:,0],AbsData[:,1],color='b')#*1e-38 )
        ax[1].plot(simAbs[:,0],simAbs[:,1],'g',linewidth=2.5)
        # ax[1].plot(simCD[:,0],(simCD[:,1]/np.max(np.abs(simCD[:,1])))*np.max(np.abs(CDdata[:,1])),'g')
        ax[1].fill_between(simAbs[:,0],gauss(simAbs[:,0], 10**7/(675/2),laser_sig,np.max(AbsData[:,1])),color='gray',alpha=0.25)
        ax[1].axhline(0,color='k',linestyle='--')
        ax[1].set_title('Abs sim w/ exp CD',fontsize=14)
        ax[1].set_xlabel(r'Energy $(cm^{-1})$',fontsize=14)
        ax[1].plot(simAbs[:,0],(simAbs[:,1] + 1.2*(simAbs[0,1])) )#*1e-40,'g')
        ax[1].set_xlim(10000,max(cCDSpectrum[:,0]))
        ax[1].set_xlim(27500,32000)
        # ax[1].set_ylim(-0.1,0.1)

    return AbsData, simAbs

# AbsData, simAbs = calc_Abs(vecs_1PE, eigs_1PE, sigma=20, gamma=200, plot_mode=1)
#%%
# =============================================================================
# Optimize calculated Abs to experimental CD
# =============================================================================
# def monomerCD_chisq_opt(params):
#     epsilon0, omega0, lam = params
#     eigs_1PE, vecs_1PE = Ham_1PE(epsilon0, omega0, lam)
#     AbsData, simAbs = calc_Abs(vecs_1PE, eigs_1PE, sigma=20, gamma=200, plot_mode=0)
    
#     mask_lower_transition = (simAbs[:,0]>27500) * (simAbs[:,0] < 32000) # put zeros outsize of this range
#     chisq = np.sum( (simAbs[:,1] - cCDSpectrum[:,1])**2 * mask_lower_transition)
#     return chisq

# lam = 2.4
# epsilon0 = 29500 
# omega0=120
# eigs_1PE, vecs_1PE = Ham_1PE(epsilon0, omega0, lam)
# AbsData, simAbs = calc_Abs(vecs_1PE, eigs_1PE, sigma=20, gamma=200, plot_mode=1)

# optimize values assuming CD is calculated like Abs...
# x0 = np.array([epsilon0, omega0, lam])
# lam_bounds = [1,2.8] #[0,4]
# epsilon0_bounds = [29000, 30000]
# omega0_bounds = [80, 150]
# bounds = np.vstack([epsilon0_bounds, omega0_bounds,lam_bounds])
# if __name__ == '__main__':
#     res = opt.differential_evolution(func=monomerCD_chisq_opt, 
#                                       bounds=bounds,
#                                       x0=x0,
#                                       disp=True,
#                                       workers=1,
#                                       maxiter=1000,
#                                       polish=True)#,
#                                       # atol=1e-8, #1e-6, 1e-10,
#                                       # tol = 1e-8, #1e-6, 10,
#                                       # mutation=(0,1.9),
#                                       # popsize=30,
#                                       # updating='immediate',
#                                       # strategy = 'best1exp') 
#     epsilon0, omega0, lam = res.x
# print('epsilon0 = '+str(epsilon0))
# print('omega0 = '+str(omega0))
# print('lam = '+str(lam))
# eigs_1PE, vecs_1PE = Ham_1PE(epsilon0, omega0, lam)
# AbsData, simAbs = calc_Abs(vecs_1PE, eigs_1PE, sigma=20, gamma=200, plot_mode=1)

# # From 20240226 fit
# epsilon0 = 29010.62480689866
# omega0 = 149.93304096216116
# lam = 2.694887318141411

#%% plot2Dspectra function from simple2Dcalc_fromRbcode_CSA_v14a
import matplotlib as mpl

# Makes it so the color bar of the 2D spec always is centered about zero (green)
class MidpointNormalize(mpl.colors.Normalize):
    def __init__(self, vmin, vmax, midpoint=0, clip=False):
        self.midpoint = midpoint
        mpl.colors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        denom_min = (self.midpoint - self.vmax)
        if denom_min == 0:
            denom_min = 1
        normalized_min = max(0, 1 / 2 * (1 - abs((self.midpoint - self.vmin) / denom_min)))
        denom_max = (self.midpoint - self.vmin)
        if denom_max == 0:
            denom_max = 1
        normalized_max = min(1, 1 / 2 * (1 + abs((self.vmax - self.midpoint) / denom_max)))
        normalized_mid = 0.5
        x, y = [self.vmin, self.midpoint, self.vmax], [normalized_min, normalized_mid, normalized_max]
        return np.ma.masked_array(np.interp(value, x, y))

# plotting function that creates aspect ratio 1 plots with the correct axes and labels, etc.
def plot2Dspectra(ax1, ax2, data, n_cont,ax_lim, timing_mode,title = '', domain = 'time',save_mode = 0,file_name=' ',scan_folder = ' '):
    #%
    scan_params = scan_folder[len('20230101-120000-'):len(scan_folder)-len('_2DFPGA')]
    stages = scan_params[len(scan_params)-2:]
    scan_type = scan_params[:len(scan_params)-3]
    axes_fontsize = 14
    cmap = 'jet' 
    diag_line_width = 4.5
    # print('scan_folder = '+scan_folder)
    # scan_params = scan_folder[len('20230101-120000-'):len(scan_folder)-len('_2DFPGA_FFT')]
    # scan_params = scan_folder[len('20230101-120000-'):len(scan_folder)-len('_2DFPGA')]
    # # stages = scan_params[len(scan_params)-2:]
    # scan_type = scan_params[:len(scan_params)-3]
    # print('scan_type = '+scan_type)
    if domain == 'time':
        x_lim = y_lim = ax_lim
        if timing_mode == 't32 = 0':
            xlabel = r'$\tau_{21}$ (fs)'
            ylabel = r'$\tau_{43}$ (fs)'
        elif timing_mode == 't21 = 0':
            xlabel = r'$\tau_{32}$ (fs)'
            ylabel = r'$\tau_{43}$ (fs)'
        elif timing_mode == 't43 = 0':
            xlabel = r'$\tau_{21}$ (fs)'
            ylabel = r'$\tau_{32}$ (fs)'
    elif domain == 'freq':
        if timing_mode == 't32 = 0':
            xlabel = r'$\omega_{21} (x10^3 cm^{-1})$'
            ylabel = r'$\omega_{43} (x10^3 cm^{-1})$'
            x_lim = y_lim = ax_lim
        elif timing_mode == 't21 = 0':
            y_lim = ax_lim
            if FT2D_mode == 0:
                xlabel = r'$\tau_{32} (fs)$'
                ylabel = r'$\omega_{43} (x10^3 cm^{-1})$' 
                x_lim = [min(ax1), max(ax1)]
            elif FT2D_mode == 1:
                xlabel = r'$\omega_{32} (x10^3 cm^{-1})$'
                ylabel = r'$\omega_{43} (x10^3 cm^{-1})$' 
                if scan_type == 'NRP_RP':
                    x_lim = np.array(ax_lim) - (10**7/700/10**3) - 0.5
                elif scan_type == 'DQC':
                    x_lim = 2 * np.array(ax_lim)
                    lim_span = ax_lim[1] - ax_lim[0]
                    x_lim = [np.mean(x_lim) - (lim_span/2), np.mean(x_lim) + (lim_span/2)]
        elif timing_mode == 't43 = 0':
            x_lim = ax_lim
            if FT2D_mode == 0:
                xlabel = r'$\omega_{21} (x10^3 cm^{-1})$'
                ylabel = r'$\tau_{32} (fs)$' 
                y_lim = [min(ax2), max(ax2)]
            else: # FT2d_mode = 1
                xlabel = r'$\omega_{21} (x10^3 cm^{-1})$'
                ylabel = r'$\omega_{32} (x10^3 cm^{-1})$' 
                if scan_type == 'NRP_RP':
                    y_lim = np.array(ax_lim) - (10**7/700/10**3) - 0.5
                else: # scan_type = DQC
                    y_lim = 2*np.array(x_lim)
                    lim_span = ax_lim[1] - ax_lim[0]
                    y_lim = [np.mean(y_lim) - (lim_span/2), np.mean(y_lim) + (lim_span/2)]
    elif domain == 'none':
        xlabel = ''
        ylabel = xlabel
    colormap = 'jet'
    plt.rcParams['contour.negative_linestyle']= 'solid'
    
    # print('x_lim = '+str(x_lim))
    # print('y_lim = '+str(y_lim))
    #%
    # fig = plt.figure(figsize=(16,16))
    fig = plt.figure(figsize=(16,4))
    fig.suptitle(title, fontsize = 16)
    plt.subplots_adjust(wspace=0.35)
    plt.subplot(131)
    if timing_mode == 't32 = 0':
        plt.gca().set_aspect('equal', adjustable='box')
    plt.title("Real")
    plt.xlabel(xlabel,fontsize=axes_fontsize)
    plt.ylabel(ylabel,fontsize=axes_fontsize)
    vals = np.real(data) #np.array([[-5., 0], [5, 10]]) 
    vmin = vals.min()
    vmax = vals.max()
    norm = MidpointNormalize(vmin=vmin, vmax=vmax, midpoint=0)
    cf = plt.contourf(ax1, ax2, np.real(data), n_cont, cmap=cmap, norm=norm)#"hsv") #20221121: CSA modification
    # if timing_mode == 't32 = 0':
    if ax1.shape == (len(ax1),len(ax1)):
        plt.plot(ax1[0,:],ax2[:,0],linestyle='--',color='w',linewidth=diag_line_width)
    else:
        plt.plot(ax1,ax2,linestyle='--',color='w',linewidth=diag_line_width)    
    fig.colorbar(cf)
    # plt.contour(self.axes[0], self.axes[1], np.real(self.dat), levels = cf.levels, cmap = 'binary')#colors='black')
    # plt.contour(ax1, ax2, np.real(data), levels = cf.levels[cf.levels >= 0], colors='black',linestyle='-')
    # plt.contour(ax1, ax2, np.real(data), levels = cf.levels[cf.levels < 0], colors='white',linestyle='.')
    plt.contour(ax1, ax2, np.real(data), levels = cf.levels[cf.levels >= 0], colors='black')
    plt.contour(ax1, ax2, np.real(data), levels = cf.levels[cf.levels < 0], colors='white')
    plt.tick_params(axis='both', which='major', labelsize=15)
    plt.xlim(x_lim)
    plt.ylim(y_lim)
    
    plt.subplot(132)
    if timing_mode == 't32 = 0':
        plt.gca().set_aspect('equal', adjustable='box')
    plt.title("Imag")
    plt.xlabel(xlabel,fontsize=axes_fontsize)
    plt.ylabel(ylabel,fontsize=axes_fontsize)
    vals = np.imag(data) #np.array([[-5., 0], [5, 10]]) 
    vmin = vals.min()
    vmax = vals.max()
    norm = MidpointNormalize(vmin=vmin, vmax=vmax, midpoint=0)
    cf = plt.contourf(ax1, ax2, np.imag(data), n_cont, cmap=cmap, norm=norm)#"hsv") #20221121: CSA modification
    # plt.contour(ax1, ax2,  np.imag(data), levels = cf.levels[cf.levels <= 0], colors='white',linestyle='-')
    plt.contour(ax1, ax2,  np.imag(data), levels = cf.levels[cf.levels <= 0], colors='white')
    # if timing_mode == 't32 = 0':
    if ax1.shape == (len(ax1),len(ax1)):
        plt.plot(ax1[0,:],ax2[:,0],linestyle='--',color='w',linewidth=diag_line_width)
    else:
        plt.plot(ax1,ax2,linestyle='--',color='w',linewidth=diag_line_width)
    fig.colorbar(cf)
    plt.contour(ax1, ax2, np.imag(data), levels = cf.levels[cf.levels > 0], colors='black')
    plt.tick_params(axis='both', which='major', labelsize=15)
    plt.xlim(x_lim)
    plt.ylim(y_lim)
    plt.subplot(133)
    if timing_mode == 't32 = 0':
        plt.gca().set_aspect('equal', adjustable='box')
    plt.title("Total")
    plt.xlabel(xlabel,fontsize=axes_fontsize)
    plt.ylabel(ylabel,fontsize=axes_fontsize)
    vals = np.abs(data) #np.array([[-5., 0], [5, 10]]) 
    vmin = vals.min()
    vmax = vals.max()
    norm = MidpointNormalize(vmin=vmin, vmax=vmax, midpoint=0)
    cf = plt.contourf(ax1, ax2, np.abs(data), n_cont, cmap=cmap, norm=norm)#"hsv") #20221121: CSA modification
    # if timing_mode == 't32 = 0':
    if ax1.shape == (len(ax1),len(ax1)):
        plt.plot(ax1[0,:],ax2[:,0],linestyle='--',color='w',linewidth=diag_line_width)
    else:
        plt.plot(ax1,ax2,linestyle='--',color='w',linewidth=diag_line_width)    
    fig.colorbar(cf)
    plt.contour(ax1, ax2, (np.abs(data)), levels = cf.levels[cf.levels > 0], colors='black')
    plt.contour(ax1, ax2, (np.abs(data)), levels = cf.levels[cf.levels <= 0], colors='white')#,linestyle='-')
    plt.tick_params(axis='both', which='major', labelsize=15)
    plt.xlim(x_lim)
    plt.ylim(y_lim)
    plt.show()
    
    # if save_mode == 1:
    #     # file_path = os.path.join('/Users/calbrecht/Dropbox/Claire_Dropbox/Data/6MI MNS/2D scan/',date_folder,scan_folder)
    #     file_path = os.path.join('/Users/clairealbrecht/Dropbox/Claire_Dropbox/Data/6MI MNS/2D scan/',date_folder,scan_folder)
    #     date_str_len = len('20230101-120000')
    #     file_name_str = file_name #+ '_' + scan_folder[date_str_len:]
    #     fig.savefig(file_path+'/'+file_name_str+'.pdf')
    #     print('...saving plot as: '+file_name_str)
    #     print('in location: '+file_path)

# plot data and sim and difference comparisons
scan_folder = ' ' # temporary definition, will be overwritten when load data
def plot_comparer(ax1,ax2, data, sim, phase_cond, compare_mode = 'real', domain='freq',figsize=(16,4), ax_lim=[28,30],n_cont=15, save_mode = 0, file_name = '',scan_folder=scan_folder,weight_func_mode=1,plot_resid_mode=1):
    timing_mode = globals()['timing_mode']
    if timing_mode =='t32 = 0':
        timing_mode_str = r'($\tau_{32}$ = 0)'
    elif timing_mode == 't43 = 0':
        timing_mode_str = r'($\tau_{43}$ = 0)'
    elif timing_mode == 't21 = 0':
        timing_mode_str = r'($\tau_{21}$ = 0)'

    FT2D_mode = globals()['FT2D_mode']
    scan_params = scan_folder[len('20230101-120000-'):len(scan_folder)-len('_2DFPGA')]
    stages = scan_params[len(scan_params)-2:]
    scan_type = scan_params[:len(scan_params)-3]
    axes_fontsize = 14
    title=phase_cond + ' Experiment vs Simulation '+timing_mode_str
    # ax2 = ax1.T
    
    # if domain == 'time':
    #     xlabel = r'$\tau_{21}$'
    #     ylabel = r'$\tau_{43}$'
    # elif domain == 'freq':
    #     xlabel = r'$\omega_{21} (x10^3 cm^{-1})$'
    #     ylabel = r'$\omega_{43} (x10^3 cm^{-1})$'
    # elif domain == 'none':
    #     xlabel = ''
    #     ylabel = xlabel
        
    # scan_params = scan_folder[len('20230101-120000-'):len(scan_folder)-len('_2DFPGA_FFT')]
    # scan_params = scan_folder[len('20230101-120000-'):len(scan_folder)-len('_2DFPGA')]
    # # stages = scan_params[len(scan_params)-2:]
    # scan_type = scan_params[:len(scan_params)-3]
    # print('scan_type = '+scan_type)
    if domain == 'time':
        x_lim = y_lim = ax_lim
        if timing_mode == 't32 = 0':
            xlabel = r'$\tau_{21}$ (fs)'
            ylabel = r'$\tau_{43}$ (fs)'
        elif timing_mode == 't21 = 0':
            xlabel = r'$\tau_{32}$ (fs)'
            ylabel = r'$\tau_{43}$ (fs)'
        elif timing_mode == 't43 = 0':
            xlabel = r'$\tau_{21}$ (fs)'
            ylabel = r'$\tau_{32}$ (fs)'
    elif domain == 'freq':
        if timing_mode == 't32 = 0':
            xlabel = r'$\omega_{21} (x10^3 cm^{-1})$'
            ylabel = r'$\omega_{43} (x10^3 cm^{-1})$'
            x_lim = y_lim = ax_lim
        elif timing_mode == 't21 = 0':
            y_lim = ax_lim
            if FT2D_mode == 0:
                xlabel = r'$\tau_{32} (fs)$'
                ylabel = r'$\omega_{43} (x10^3 cm^{-1})$' 
                x_lim = [min(ax1), max(ax1)]
            elif FT2D_mode == 1:
                xlabel = r'$\omega_{32} (x10^3 cm^{-1})$'
                ylabel = r'$\omega_{43} (x10^3 cm^{-1})$' 
                if scan_type == 'NRP_RP':
                    x_lim = np.array(ax_lim) - (10**7/700/10**3) - 0.5
                elif scan_type == 'DQC':
                    x_lim = 2 * np.array(ax_lim)
                    lim_span = ax_lim[1] - ax_lim[0]
                    x_lim = [np.mean(x_lim) - (lim_span/2), np.mean(x_lim) + (lim_span/2)]
        elif timing_mode == 't43 = 0':
            x_lim = ax_lim
            if FT2D_mode == 0:
                xlabel = r'$\omega_{21} (x10^3 cm^{-1})$'
                ylabel = r'$\tau_{32} (fs)$' 
                y_lim = [min(ax2), max(ax2)]
            else: # FT2d_mode = 1
                xlabel = r'$\omega_{21} (x10^3 cm^{-1})$'
                ylabel = r'$\omega_{32} (x10^3 cm^{-1})$' 
                if scan_type == 'NRP_RP':
                    y_lim = np.array(ax_lim) - (10**7/700/10**3) - 0.5
                else: # scan_type = DQC
                    y_lim = 2*np.array(x_lim)
                    lim_span = ax_lim[1] - ax_lim[0]
                    y_lim = [np.mean(y_lim) - (lim_span/2), np.mean(y_lim) + (lim_span/2)]
        
        
    colormap = 'jet'
    plt.rcParams['contour.negative_linestyle']= 'solid'
    diag_line_width = 4.5
            
    loc = 521
    
    if weight_func_mode == 1:
        xx = np.linspace(0,400,400)
        xx = np.tile(xx, (len(xx),1))
        yy = xx.T
        wind = delayedGaussian(np.sqrt(xx**2 + yy**2),4, 6); #70e-15,10e-15); 
        w,h=plt.figaspect(1.)
        wind2d = np.hstack([np.rot90(wind,2),np.rot90(wind)])
        wind2d = np.vstack([wind2d,np.rot90(wind2d,2)])        
        wind2d_len = len(wind2d)
        zeros2add_left = int(loc - (wind2d_len/2))
        zeros2add_right = len(ax1) - (zeros2add_left + wind2d_len)
        downward_shift = 3
        wind2d_full = np.hstack([np.zeros([wind2d_len,zeros2add_left]),wind2d, np.zeros([wind2d_len,zeros2add_right])])
        wind2d_full = np.vstack([np.zeros([zeros2add_left-downward_shift,len(ax1)]), wind2d_full, np.zeros([zeros2add_right+downward_shift,len(ax1)])])
        weight_func = wind2d_full
    else:
        weight_func = np.ones(RP_exp.shape)
    
    if compare_mode == 'real':
        # fig = plt.figure(figsize=(16,16))
        # fig = plt.figure(figsize=(16,4))
        fig = plt.figure(figsize=figsize)
        fig.suptitle(title, fontsize = 16)
        plt.subplots_adjust(wspace=0.35)
        if plot_resid_mode == 1:
            plt.subplot(131)
        elif plot_resid_mode == 0:
            plt.subplot(121)
        plt.gca().set_aspect('equal', adjustable='box')
        plt.title("Real Exp")
        plt.xlabel(xlabel,fontsize=axes_fontsize)
        plt.ylabel(ylabel,fontsize=axes_fontsize)
        vals = np.real(data/np.max(np.max(np.real(data)))) #np.array([[-5., 0], [5, 10]]) 
        vmin = vals.min()
        vmax = vals.max()
        norm = MidpointNormalize(vmin=vmin, vmax=vmax, midpoint=0)
        cf = plt.contourf(ax1, ax2, np.real(data/np.max(np.max(np.real(data)))), n_cont, cmap = colormap, norm=norm)#"hsv") #20221121: CSA modification
        if ax1.shape == (len(ax1),len(ax1)):
            plt.plot(ax1[0,:],ax2[:,0],linestyle='--',color='w',linewidth = diag_line_width,zorder=1e3)
        else:
            plt.plot(ax1,ax2,linestyle='--',color='w',linewidth = diag_line_width,zorder=1e3)    
        fig.colorbar(cf)
        # plt.contour(self.axes[0], self.axes[1], np.real(self.dat), levels = cf.levels, cmap = 'binary')#colors='black')
        # plt.contour(ax1, ax2, np.real(data), levels = cf.levels[cf.levels >= 0], colors='black',linestyle='-')
        # plt.contour(ax1, ax2, np.real(data), levels = cf.levels[cf.levels < 0], colors='white',linestyle='.')
        plt.contour(ax1, ax2, np.real(data)/np.max(np.max(np.real(data))), levels = cf.levels[cf.levels >= 0], colors='black')
        plt.contour(ax1, ax2, np.real(data)/np.max(np.max(np.real(data))), levels = cf.levels[cf.levels < 0], colors='white')
        plt.tick_params(axis='both', which='major', labelsize=15)
        plt.xlim(x_lim)
        plt.ylim(y_lim)
        
        # plt.subplot(132)
        if plot_resid_mode == 1:
            plt.subplot(132)
        elif plot_resid_mode == 0:
            plt.subplot(122)
        plt.gca().set_aspect('equal', adjustable='box')
        plt.title("Real Sim")
        plt.xlabel(xlabel,fontsize=axes_fontsize)
        plt.ylabel(ylabel,fontsize=axes_fontsize)
        vals = np.real(sim)/np.max(np.max(np.real(sim))) #np.array([[-5., 0], [5, 10]]) 
        vmin = vals.min()
        vmax = vals.max()
        norm = MidpointNormalize(vmin=vmin, vmax=vmax, midpoint=0)
        cf = plt.contourf(ax1, ax2, np.real(sim)/np.max(np.max(np.real(sim))), n_cont, cmap = colormap,norm=norm)#"hsv") #20221121: CSA modification
        if ax1.shape == (len(ax1),len(ax1)):
            plt.plot(ax1[0,:],ax2[:,0],linestyle='--',color='w',linewidth = diag_line_width,zorder=1e3)
        else:
            plt.plot(ax1,ax2,linestyle='--',color='w',linewidth = diag_line_width,zorder=1e3)    
        fig.colorbar(cf)
        # plt.contour(self.axes[0], self.axes[1], np.real(self.dat), levels = cf.levels, cmap = 'binary')#colors='black')
        # plt.contour(ax1, ax2, np.real(data), levels = cf.levels[cf.levels >= 0], colors='black',linestyle='-')
        # plt.contour(ax1, ax2, np.real(data), levels = cf.levels[cf.levels < 0], colors='white',linestyle='.')
        plt.contour(ax1, ax2, np.real(sim)/np.max(np.max(np.real(sim))), levels = cf.levels[cf.levels >= 0], colors='black')
        plt.contour(ax1, ax2, np.real(sim)/np.max(np.max(np.real(sim))), levels = cf.levels[cf.levels < 0], colors='white')
        plt.tick_params(axis='both', which='major', labelsize=15)
        plt.xlim(x_lim)
        plt.ylim(y_lim)
        
        if plot_resid_mode == 1:
            plt.subplot(133)
            plt.gca().set_aspect('equal', adjustable='box')
            if weight_func_mode == 1:
                plt.title("Squared residuals (w/weight func)")
            else:
                plt.title("Squared residuals (w/o weight func)")
            plt.xlabel(xlabel,fontsize=axes_fontsize)
            plt.ylabel(ylabel,fontsize=axes_fontsize)
            # ax1_2d = np.tile(ax1, (len(ax1),1))
            # ax2_2d = ax1_2d.T
            # wx = gaussian2d(ax1_2d, 29, 2)
            # wy = gaussian2d(ax2_2d, 29, 2)
            # weight_func = wx*wy
            # weight_func = np.ones(sim.shape)
                   
            diff = (np.abs(np.real(sim))/np.max(np.max(np.abs(np.real(sim)))) - np.abs(np.real(data))/np.max(np.max(np.abs(np.real(data)))))**2 * weight_func
            vals = diff #np.array([[-5., 0], [5, 10]]) 
            vmin = vals.min()
            vmax = vals.max()
            norm = MidpointNormalize(vmin=vmin, vmax=vmax, midpoint=0)
            cf = plt.contourf(ax1, ax2, diff, n_cont, cmap = colormap,norm=norm)#"hsv") #20221121: CSA modification
            if ax1.shape == (len(ax1),len(ax1)):
                plt.plot(ax1[0,:],ax2[:,0],linestyle='--',color='w',linewidth = diag_line_width,zorder=1e3)
            else:
                plt.plot(ax1,ax2,linestyle='--',color='w',linewidth = diag_line_width,zorder=1e3)    
            fig.colorbar(cf)
            # plt.contour(self.axes[0], self.axes[1], np.real(self.dat), levels = cf.levels, cmap = 'binary')#colors='black')
            # plt.contour(ax1, ax2, np.real(data), levels = cf.levels[cf.levels >= 0], colors='black',linestyle='-')
            # plt.contour(ax1, ax2, np.real(data), levels = cf.levels[cf.levels < 0], colors='white',linestyle='.')
            plt.contour(ax1, ax2, diff, levels = cf.levels[cf.levels >= 0], colors='black')
            if len(cf.levels[cf.levels < 0]) > 0:
                plt.contour(ax1, ax2, diff, levels = cf.levels[cf.levels < 0], colors='white')
            plt.tick_params(axis='both', which='major', labelsize=15)
            plt.xlim(x_lim)
            plt.ylim(y_lim)
    
        plt.show()
        
        if save_mode == 1:
            # file_path = os.path.join('/Users/calbrecht/Dropbox/Claire_Dropbox/Data/6MI MNS/2D scan/',date_folder,scan_folder)
            # file_path = os.path.join('/Users/clairealbrecht/Dropbox/Claire_Dropbox/Data/6MI MNS/2D scan/',date_folder,scan_folder)
            scan_folder_split = scan_folder.split('_2D')
            file_path = os.path.join('/Users/calbrec2/Documents/github_base/Data/2PE2DFS/Data_lock_in/MNS_4uM/2D scan/',date_folder,scan_folder_split[0])#'20221202/20221202-142926_DQC_xz')
            date_str_len = len('20230101-120000')
            file_name_str = file_name #+ '_real_' + '_' + scan_folder[date_str_len:]
            fig.savefig(file_path+'/'+file_name_str+'.pdf',transparent=True)
            print('...saving plot as: '+file_name_str)
            print('in location: '+file_path)
        
    elif compare_mode == 'imag':
                # fig = plt.figure(figsize=(16,16))
        # fig = plt.figure(figsize=(16,4))
        fig = plt.figure(figsize=figsize)
        fig.suptitle(title, fontsize = 16)
        plt.subplots_adjust(wspace=0.35)
        plt.subplot(131)
        plt.gca().set_aspect('equal', adjustable='box')
        plt.title("Imag Exp")
        plt.xlabel(xlabel,fontsize=axes_fontsize)
        plt.ylabel(ylabel,fontsize=axes_fontsize)
        vals = np.imag(data/np.max(np.max(np.imag(data)))) #np.array([[-5., 0], [5, 10]]) 
        vmin = vals.min()
        vmax = vals.max()
        norm = MidpointNormalize(vmin=vmin, vmax=vmax, midpoint=0)
        cf = plt.contourf(ax1, ax2, vals, n_cont, cmap = colormap,norm=norm)#"hsv") #20221121: CSA modification
        if ax1.shape == (len(ax1),len(ax1)):
            plt.plot(ax1[0,:],ax2[:,0],linestyle='--',color='w',linewidth = diag_line_width,zorder=1e3)
        else:
            plt.plot(ax1,ax2,linestyle='--',color='w',linewidth = diag_line_width,zorder=1e3)    
        fig.colorbar(cf)
        # plt.contour(self.axes[0], self.axes[1], np.real(self.dat), levels = cf.levels, cmap = 'binary')#colors='black')
        # plt.contour(ax1, ax2, np.real(data), levels = cf.levels[cf.levels >= 0], colors='black',linestyle='-')
        # plt.contour(ax1, ax2, np.real(data), levels = cf.levels[cf.levels < 0], colors='white',linestyle='.')
        plt.contour(ax1, ax2, vals, levels = cf.levels[cf.levels >= 0], colors='black')
        plt.contour(ax1, ax2, vals, levels = cf.levels[cf.levels < 0], colors='white')
        plt.xlim(x_lim)
        plt.ylim(y_lim)
        
        plt.subplot(132)
        plt.gca().set_aspect('equal', adjustable='box')
        plt.title("Imag Sim")
        plt.xlabel(xlabel,fontsize=axes_fontsize)
        plt.ylabel(ylabel,fontsize=axes_fontsize)
        vals = np.imag(sim)/np.max(np.max(np.imag(sim))) #np.array([[-5., 0], [5, 10]]) 
        vmin = vals.min()
        vmax = vals.max()
        norm = MidpointNormalize(vmin=vmin, vmax=vmax, midpoint=0)
        cf = plt.contourf(ax1, ax2, vals, n_cont, cmap = colormap,norm=norm)#"hsv") #20221121: CSA modification
        if ax1.shape == (len(ax1),len(ax1)):
            plt.plot(ax1[0,:],ax2[:,0],linestyle='--',color='w',linewidth = diag_line_width,zorder=1e3)
        else:
            plt.plot(ax1,ax2,linestyle='--',color='w',linewidth = diag_line_width,zorder=1e3)    
        fig.colorbar(cf)
        # plt.contour(self.axes[0], self.axes[1], np.real(self.dat), levels = cf.levels, cmap = 'binary')#colors='black')
        # plt.contour(ax1, ax2, np.real(data), levels = cf.levels[cf.levels >= 0], colors='black',linestyle='-')
        # plt.contour(ax1, ax2, np.real(data), levels = cf.levels[cf.levels < 0], colors='white',linestyle='.')
        plt.contour(ax1, ax2, vals, levels = cf.levels[cf.levels >= 0], colors='black')
        plt.contour(ax1, ax2, vals, levels = cf.levels[cf.levels < 0], colors='white')
        plt.tick_params(axis='both', which='major', labelsize=15)
        plt.xlim(x_lim)
        plt.ylim(y_lim)
        
        plt.subplot(133)
        plt.gca().set_aspect('equal', adjustable='box')
        if weight_func_mode == 1:
            plt.title("Squared residuals (w/weight func)")
        else:
            plt.title("Squared residuals (w/o weight func)")
        plt.xlabel(xlabel,fontsize=axes_fontsize)
        plt.ylabel(ylabel,fontsize=axes_fontsize)
        ax1_2d = np.tile(ax1, (len(ax1),1))
        ax2_2d = ax1_2d.T
        # wx = gaussian2d(ax1_2d, 29, 2)
        # wy = gaussian2d(ax2_2d, 29, 2)
        # weight_func = wx*wy
        # weight_func = np.ones(sim.shape)
        diff = (np.abs(np.imag(sim))/np.max(np.max(np.abs(np.imag(sim)))) - np.abs(np.imag(data))/np.max(np.max(np.abs(np.imag(data)))))**2 * weight_func
        vals = diff #np.array([[-5., 0], [5, 10]]) 
        vmin = vals.min()
        vmax = vals.max()
        cf = plt.contourf(ax1, ax2, diff, n_cont, cmap = colormap,norm=norm)#"hsv") #20221121: CSA modification
        if ax1.shape == (len(ax1),len(ax1)):
            plt.plot(ax1[0,:],ax2[:,0],linestyle='--',color='w',linewidth = diag_line_width,zorder=1e3)
        else:
            plt.plot(ax1,ax2,linestyle='--',color='w',linewidth = diag_line_width,zorder=1e3)    
        fig.colorbar(cf)
        # plt.contour(self.axes[0], self.axes[1], np.real(self.dat), levels = cf.levels, cmap = 'binary')#colors='black')
        # plt.contour(ax1, ax2, np.real(data), levels = cf.levels[cf.levels >= 0], colors='black',linestyle='-')
        # plt.contour(ax1, ax2, np.real(data), levels = cf.levels[cf.levels < 0], colors='white',linestyle='.')
        plt.contour(ax1, ax2, diff, levels = cf.levels[cf.levels >= 0], colors='black')
        plt.contour(ax1, ax2, diff, levels = cf.levels[cf.levels < 0], colors='white')
        plt.tick_params(axis='both', which='major', labelsize=15)
        plt.xlim(x_lim)
        plt.ylim(y_lim)
    
        plt.show()
        
        if save_mode == 1:
            # file_path = os.path.join('/Users/calbrecht/Dropbox/Claire_Dropbox/Data/6MI MNS/2D scan/',date_folder,scan_folder)
            # file_path = os.path.join('/Users/clairealbrecht/Dropbox/Claire_Dropbox/Data/6MI MNS/2D scan/',date_folder,scan_folder)
            scan_folder_split = scan_folder.split('_2D')
            file_path = os.path.join('/Users/calbrec2/Documents/github_base/Data/2PE2DFS/Data_lock_in/MNS_4uM/2D scan/',date_folder,scan_folder_split[0])#'20221202/20221202-142926_DQC_xz')
            date_str_len = len('20230101-120000')
            file_name_str = file_name #+ scan_folder[date_str_len:]
            fig.savefig(file_path+'/'+file_name_str+'.pdf',transparent=True)
            print('...saving plot as: '+file_name_str)
            print('in location: '+file_path)

#%% Function to load 2D data

import os.path
from pathlib import Path

home = str(Path.home())
# print(os.path.join(home, "CodeProjects","WebSafer"))

# grabs data to be optimized to
# 20231215: going to change file paths to take files from github folder
# create a computer mode function to be able to take files regardless of what machine this is run on?
def data_file_grabber(date_folder, scan_folder,sample_name, load_tau_domain =0):
    # file_path = '/Users/calbrecht/Dropbox/Claire_Dropbox/Data/6MI nucleoside/2D scan/20221202/20221202-135005_NRP_RP_xz'
    # file_path = os.path.join('/Users/calbrecht/Dropbox/Claire_Dropbox/Data/6MI MNS/2D scan/',date_folder,scan_folder)
    # file_path = os.path.join('/Users/clairealbrecht/Dropbox/Claire_Dropbox/Data/6MI MNS/2D scan/',date_folder,scan_folder)
    # file_path = os.path.join('/Users/clairealbrecht/Dropbox/Claire_Dropbox/Data_FPGA/MNS_4uM/2D_scans/',date_folder,scan_folder)
    # file_path = os.path.join('/Users/calbrecht/Dropbox/Claire_Dropbox/Data_FPGA/MNS_4uM/2D_scans/',date_folder,scan_folder)
    
    # if FPGA_mode == 1:
    #     file_path = os.path.join('/Users/calbrec2/Dropbox/Claire_Dropbox/Data_FPGA/MNS_4uM/2D_scans/',date_folder,scan_folder)
    # else:
    #     file_path = os.path.join('/Users/calbrec2/Dropbox/Claire_Dropbox/Data/6MI MNS/2D scan/',date_folder,scan_folder)
        
    # if FPGA_mode == 1:
    #     # file_path = os.path.join(os.path.join('/Users/calbrec2/Dropbox/Claire_Dropbox/Data_FPGA/', sample_name), '/2D_scans/') ,date_folder,scan_folder)
    #     file_path = '/Users/calbrec2/Dropbox/Claire_Dropbox/Data_FPGA/'+ sample_name+ '/2D_scans/' + date_folder #+ '/' + scan_folder
    # else:
    #     file_path = os.path.join('/Users/calbrec2/Dropbox/Claire_Dropbox/Data/6MI MNS/2D scan/',date_folder,scan_folder)
            
    home = str(Path.home())
    if len(glob.glob(home+'*')) > 0:
        if FPGA_mode == 1:
            file_path = os.path.join(home,'Documents','github_base','Data','2PE2DFS','Data_FPGA',sample_name,'2D_scans', date_folder, scan_folder)
        else:
            file_path = os.path.join(home, 'Documents','github_base','Data','2PE2DFS','Data_lock_in',sample_name,'2D scan',date_folder, scan_folder)
    
    os.chdir(file_path)
    
    
    # scan_params = scan_folder[len('20230101-120000-'):len(scan_folder)-len('_2DFPGA_FFT')]
    if FPGA_mode == 1:
        scan_params = scan_folder[len('20230101-120000-'):len(scan_folder)-len('_2DFPGA')]
    else:
        scan_params = scan_folder[len('20230101-120000-'):]

    stages = scan_params[len(scan_params)-2:]
    scan_type = scan_params[:len(scan_params)-3]
    
    # Update 20231116
    if stages == 'xz':
        FT2D_mode = 0
    else:
        FT2D_mode = 1 # forcing FT2D for now... 20231215 CSA fix this eventually
    # files_mat = glob.glob('*.mat')

    if FT2D_mode == 1:
        file_FFT = glob.glob('*'+scan_type+'*'+stages+'*FFT2.mat')[0]
    else: 
        file_FFT = glob.glob('*'+scan_type+'*'+stages+'*FFT.mat')[0]
    file_RF_raw = glob.glob('*'+scan_type+'*'+stages+'*RF_raw*.mat')[0]


    if FPGA_mode == 0:
        file_RF_rt = glob.glob('*'+scan_type+'*'+stages+'*RF.mat')[0]
    else:
        if len(glob.glob('*'+scan_type+'*'+stages+'*RF_rt*.mat')) == 0:
            print('cant find retimed data, loading raw twice for now... fix later') # 20231215 CSA - need to fix
            file_RF_rt = file_RF_raw
        else:
            file_RF_rt = glob.glob('*'+scan_type+'*'+stages+'*RF_rt*.mat')[0]

    import scipy.io
    
    # =============================================================================
    # Load retimed & rephased time domain
    # =============================================================================
    
    # if load_tau_domain == 1:
    file = file_RF_rt
    mat = scipy.io.loadmat(file)
    # mat = scipy.io.loadmat(file[np.char.find(file,'_RF.mat')>0][0]) # used prior to 20231116
    
    # load the time domain data
    SHGcenter = mat['SHGcenter'][0][0]
    Tsn = mat['Tsn'][0][0]
    dqc_mode = mat['dqc_mode'][0][0]

    
    if FPGA_mode == 0:
        dmatIntPx2 = mat['dmatIntPx2']
        smatIntPx2 = mat['smatIntPx2']
        t21ax_rt = mat['t21ax_rt'][0]
        t43ax_rt = mat['t43ax_rt'][0]
    else:
        if len(glob.glob('*'+scan_type+'*'+stages+'*RF_rt*.mat')) == 0:
            dmatIntPx2 = mat['RPmat']               # 20231215 CSA - need to fix with comment above
            smatIntPx2 = mat['NRPmat']
            t21ax_rt = mat['tb1'].flatten() #[0]
            if t21ax_rt[2] < 0:
                t21ax_rt = - t21ax_rt
            t43ax_rt = mat['raw_ax2'].flatten() #[0]
        else:
            dmatIntPx2 = mat['dmat']
            smatIntPx2 = mat['smat']
            t21ax_rt = mat['t21ax'][0]
            t43ax_rt = mat['t43ax'][0]
    
    
    
    # timingMode = mat['timingMode']
    n_cont = 15
    tau_ax_lim = [0, 200] #104]
    if dqc_mode == 0:
        smatIntPx2 = smatIntPx2 / np.max(np.max(smatIntPx2))
        plot2Dspectra(t21ax_rt, t43ax_rt, smatIntPx2, n_cont,  ax_lim=tau_ax_lim, timing_mode=timing_mode,title=r'Experimental NRP($\tau$) with ($\tau_{32}$ = 0) w/re-timing & re-phasing')
        dmatIntPx2 = dmatIntPx2 / np.max(np.max(dmatIntPx2))
        plot2Dspectra(t21ax_rt, t43ax_rt, dmatIntPx2, n_cont,  ax_lim=tau_ax_lim,timing_mode=timing_mode, title=r'Experimental RP($\tau$) with ($\tau_{32}$ = 0) w/re-timing & re-phasing')
    else:
        smatIntPx2 = smatIntPx2 / np.max(np.max(smatIntPx2))
        plot2Dspectra(t21ax_rt, t43ax_rt, smatIntPx2, n_cont, ax_lim=tau_ax_lim,timing_mode=timing_mode, title=r'Experimental DQC($\tau$) with ($\tau_{32}$ = 0) w/re-timing & re-phasing')
    
    
    # =============================================================================
    # Load raw time domain
    # =============================================================================
    # mat = scipy.io.loadmat(file[1])  
    
    # mat = scipy.io.loadmat(file[np.char.find(file,'_RF_raw.mat')>0][0]) # used prior to 20231116
    file = file_RF_raw
    mat = scipy.io.loadmat(file)
    
    if FPGA_mode == 0:
        Ts = mat['Ts'][0][0]
        dmatW = mat['dmatW']
        smatW = mat['smatW']
        t21ax = mat['t21ax'].flatten() #[0]
        t43ax = mat['t43ax'].flatten() #[0]
    else:
        dmatW = mat['RPmat']
        smatW = mat['NRPmat']
        t21ax = mat['tb1'].flatten() #[0]
        if t21ax[2] < 0:
            t21ax = - t21ax
        t43ax = mat['raw_ax2'].flatten() #[0]
    
    
    dqc_mode = mat['dqc_mode'][0][0]
    
    # timingMode = mat['timingMode']
    tau_ax_lim = [min(t21ax), max(t21ax)]
    n_cont = 15
    if dqc_mode == 0:
        smatW = smatW / np.max(np.max(smatW))
        plot2Dspectra(t21ax, t43ax, smatW, n_cont, ax_lim=tau_ax_lim, timing_mode=timing_mode,title=r'Experimental NRP($\tau$) with ($\tau_{32}$ = 0)', scan_folder = scan_folder)
        dmatW = dmatW / np.max(np.max(dmatW))
        plot2Dspectra(t21ax, t43ax, dmatW, n_cont,ax_lim=tau_ax_lim,  timing_mode=timing_mode,title=r'Experimental RP($\tau$) with ($\tau_{32}$ = 0)', scan_folder = scan_folder)
    else:
        smatW = smatW / np.max(np.max(smatW))
        plot2Dspectra(t21ax, t43ax, smatW, n_cont, ax_lim=tau_ax_lim, timing_mode=timing_mode,title=r'Experimental DQC($\tau$) with ($\tau_{32}$ = 0)', scan_folder = scan_folder)
    
    
    # =============================================================================
    # Load spectral domain
    # =============================================================================

    file = file_FFT
    mat = scipy.io.loadmat(file)
    
    # load the frequency domain data
    SHGcenter = mat['SHGcenter'][0][0]
    xaxis = mat['xaxis'][0]
    yaxis = mat['yaxis'][0]
    if FPGA_mode == 0:
        sumFunc_RT = mat['sumFunc_RT']
        difFunc_RT = mat['difFunc_RT']
    else:
        sumFunc_RT = mat['sumFunc']
        difFunc_RT = mat['diffFunc']
    xbounds = mat['xbounds'][0]
    dqc_mode = mat['dqc_mode'][0]

    if scan_folder == '20230208-141423_DQC_xz':
        sumFunc_RT = np.rot90(sumFunc_RT,2)
    
    n_cont = 15
    if dqc_mode == 0:
        if date_folder == '20221122':
            sumFunc_RT = np.rot90(sumFunc_RT,-1)
            difFunc_RT = np.rot90(difFunc_RT,2)
        plot2Dspectra(xaxis, yaxis, sumFunc_RT, n_cont, ax_lim=xbounds,timing_mode=timing_mode, title=r'Experimental NRP($\omega$) with ($\tau_{32}$ = 0)', domain='freq',scan_folder = scan_folder_nrprp)
        plot2Dspectra(xaxis, yaxis, difFunc_RT, n_cont, ax_lim=xbounds,timing_mode=timing_mode, title=r'Experimental RP($\omega$) with ($\tau_{32}$ = 0)',domain='freq',  scan_folder = scan_folder_nrprp)
    else:
        if date_folder == '20221122':
            sumFunc_RT = np.rot90(sumFunc_RT,1)
        plot2Dspectra(xaxis, yaxis, sumFunc_RT, n_cont, ax_lim=xbounds, timing_mode=timing_mode,title=r'Experimental DQC($\omega$) with ($\tau_{32}$ = 0)',domain='freq', scan_folder = scan_folder_dqc)

    return xaxis, yaxis, sumFunc_RT, difFunc_RT, xbounds, dqc_mode, t21ax_rt, t43ax_rt, smatIntPx2, dmatIntPx2, Tsn, t43ax, t21ax, dmatW, smatW


global xaxis, yaxis, DQC_exp, NRP_exp, RP_exp , NRP_tau_exp, RP_tau_exp, DQC_tau_exp, t43ax, t21ax, dmatW, smatW
global timing_mode, FPGA_mode, sample_name, FT2D_mode, ET
FT2D_mode = 1

#%% LOAD 2PE-2DFS data for particular sample (unfold section to see all options)
# =============================================================================
# # sample_name = 'MNS_4um' # all samples before ~20231030
# =============================================================================

# FPGA_mode = 0
# sample_name = 'MNS_4uM'

# date_folder = '20221202'
# scan_folder_nrprp = '20221202-135005_NRP_RP_xz' # first set of data optimized 
# scan_folder_dqc = '20221202-142926_DQC_xz'
# parameters saved here: '2023-07-21_optimized_params.npy'

# date_folder = '20230208'
# # scan_folder_nrprp = '20230208-104734_NRP_RP_xz'
# # scan_folder_nrprp = '20230208-115102_NRP_RP_xz'
# scan_folder_nrprp = '20230208-130045_NRP_RP_xz'
# scan_folder_dqc = '20230208-141423_DQC_xz'
# =============================================================================
# # # Not good data after all....
# =============================================================================

# date_folder = '20230214'
# scan_folder_nrprp = '20230214-102219_NRP_RP_xz'
# scan_folder_dqc = '20230214-113706_DQC_xz'
# =============================================================================
# 
# =============================================================================

# date_folder = '20230316'
# scan_folder_nrprp = '20230316-111544_NRP_RP_xz'
# scan_folder_dqc = '20230316-121356_DQC_xz'
# # scan_folder_dqc = '20230316-133621_DQC_yz'
# # scan_folder_nrprp = '20230316-143244_NRP_RP_yz'
# =============================================================================
# 
# =============================================================================



FPGA_mode = 1
sample_name = 'MNS_4uM'
# date_folder = '20230728' # all six data sets
# scan_folder_nrprp = '20230728-115041-NRP_RP_xz_2DFPGA'
# scan_folder_dqc = '20230728-130413-DQC_xz_2DFPGA'
# scan_folder_dqc = '20230728-141633-DQC_xy_2DFPGA'
# scan_folder_nrprp = '20230728-153548-NRP_RP_xy_2DFPGA'
# =============================================================================
# saved results w/ alpha fix in: 20231101_071354_optimized_params
# =============================================================================


# date_folder = '20230729' # two good data sets
# scan_folder_nrprp ='20230729-105217-NRP_RP_xz_2DFPGA'
# scan_folder_dqc ='20230729-121444-DQC_xz_2DFPGA'
# =============================================================================
# saved results w/ alpha fix in: 20231101_083207_optimized_params
# =============================================================================

# date_folder = '20230801' # two good data sets
# scan_folder_nrprp = '20230801-115033-NRP_RP_xz_2DFPGA'
# scan_folder_dqc = '20230801-130235-DQC_xz_2DFPGA'
# # scan_folder_nrprp = '20230801-144625-NRP_RP_yz_2DFPGA' #THESE HAD STAGES MOVING INCORRECTLY!!! DO NOT USE!
# # scan_folder_dqc = '20230801-160023-DQC_yz_2DFPGA'
# =============================================================================
# saved results w/ alpha fix: 20231101_102817_optimized_params
# =============================================================================

# date_folder = '20230802' # only NRP,RP
# scan_folder_nrprp = '20230802-105049-NRP_RP_xz_2DFPGA'
# # scan_folder_dqc =

# date_folder = '20230803' # two good data sets
# scan_folder_nrprp ='20230803-092506-NRP_RP_xz_2DFPGA'
# scan_folder_dqc ='20230803-103918-DQC_xz_2DFPGA'
# =============================================================================
# saved results w/ alpha fix: 20231106_135431_optimized_params (mnight be one from 20231101 also)
# =============================================================================

date_folder = '20231129'
# scan_folder_nrprp = '20231129-092422-NRP_RP_xz_2DFPGA'
# scan_folder_dqc = '20231129-102354-DQC_xz_2DFPGA'
scan_folder_nrprp ='20231129-113400-NRP_RP_yz_2DFPGA'
scan_folder_dqc = '20231129-123606-DQC_yz_2DFPGA'
# scan_folder_nrprp ='20231129-134634-NRP_RP_xy_2DFPGA'
# scan_folder_dqc = '20231129-144844-DQC_xy_2DFPGA'

# =============================================================================
# Let's look at MNT data
# =============================================================================
# sample_name = 'MNT_5perc'
# date_folder = '20231030'
# scan_folder_nrprp = '20231030-111439-NRP_RP_xz_2DFPGA_FFT'
# scan_folder_dqc = '220231030-121729-DQC_xz_2DFPGA_FFT'
# scan_folder_nrprp = '20231030-132010-NRP_RP_yz_2DFPGA_FFT'
# scan_folder_dqc = '20231030-141546-DQC_yz_2DFPGA_FFT'
# scan_folder_nrprp = '20231030-151811-NRP_RP_xy_2DFPGA_FFT'
# scan_folder_dqc = '20231030-161416-DQC_xy_2DFPGA_FFT'

# =============================================================================
# Let's look at oreg0112 monomer ssDNA data
# =============================================================================

#%
# =============================================================================
# 
# =============================================================================
# scan_folder = scan_folder_nrprp
# # scan_params = scan_folder[len('20230101-120000-'):len(scan_folder)-len('_2DFPGA_FFT')]
# scan_params = scan_folder[len('20230101-120000-'):len(scan_folder)-len('_2DFPGA')]
# stages = scan_params[len(scan_params)-2:]
# scan_type = scan_params[:len(scan_params)-3]
# # timing_mode = 't32 = 0'
# timing_mode ='t21 = 0'
# # timing_mode ='t43 = 0'
# FT2D_mode = 1
# ax_lim = [14, 15.5]
scan_folder = scan_folder_nrprp
scan_params = scan_folder[len('20230101-120000-'):len(scan_folder)-len('_2DFPGA')]
stages = scan_params[len(scan_params)-2:]
if stages == 'xz':
    timing_mode ='t32 = 0'
elif stages == 'yz':
    timing_mode ='t21 = 0'
elif stages == 'xy':
    timing_mode ='t43 = 0'

xaxis, yaxis, sumFunc_RT, difFunc_RT, xbounds, dqc_mode, t21ax_rt, t43ax_rt, smatIntPx2, dmatIntPx2, Tsn, t43ax, t21ax, dmatW, smatW = data_file_grabber(date_folder, scan_folder_nrprp,sample_name,load_tau_domain=1)
NRP_tau_raw_exp, RP_tau_raw_exp = smatW, dmatW
NRP_tau_exp, RP_tau_exp = smatIntPx2, dmatIntPx2
NRP_exp, RP_exp = sumFunc_RT, difFunc_RT

xaxis, yaxis, sumFunc_RT, difFunc_RT, xbounds, dqc_mode, t21ax_rt, t43ax_rt, smatIntPx2, dmatIntPx2, Tsn, t43ax, t21ax, dmatW, smatW = data_file_grabber(date_folder, scan_folder_dqc,sample_name,load_tau_domain=1)
DQC_tau_raw_exp = smatW
DQC_tau_exp = sumFunc_RT
DQC_exp = sumFunc_RT
scan_folder = ' '


#%%
# =============================================================================
# Now perform a corresponding simulation of the 2PE-2DFS data
# =============================================================================
# Calculate the 2PE hamiltonian
eigs_2PE, vecs_2PE= Ham_2PE(epsilon0, omega0, lam)


#%%
    # % notes for omega123 indexing pattern 
    # =============================================================================
    # Build omega1, omega2, omega2 arrays for each pathway
    # =============================================================================
    # from simple2Dcode... 
    #                 1         2           3           4           5           6              7            8
        # omega1 = [omega_ge, omega_gep,   omega_gep,  omega_ge,   omega_gep,   omega_ge,     omega_ge,   omega_gep]
    
    # NRP omega2 = [omega_ee, omega_epep,  omega_eep,   omega_eep,   omega_eep,   omega_eep, omega_ee, omega_epep] 
    # DQC omega2 = [omega_gf, omega_gfp,   omega_gf,   omega_gf,   omega_gfp,   omega_gfp,    omega_gfp,  omega_gf]        
        
        # omega3 = [omega_ef, omega_epfp,  omega_ef,   omega_epf,  omega_efp,   omega_epfp,   omega_efp,  omega_epf]
        
    
    # omega1: selecting from omegas_ges
    # omega2: selecting from omegas_eeps or omegas_gfs
    # omega3: selecting from omegas_efs
    
    # Or is it simpler to generate omegas_ij = [omega1, omega2, omega3]
    # omegas_ij = np.zeros([3,1])
    # omegas_ij[0,0] = omegas_ges[0]
    # omegas_ij[1,0] = omegas_eeps[0,0]
    # omegas_ij[2,0] = omegas_efs[0,0]
    # omegas_ij[0,1] = omegas_ges[0]
    # omegas_ij[1,1] = omegas_eeps[0,1]
    # omegas_ij[2,1] = omegas_efs[0,1]
    # plt.matshow(omegas_ij)
    
    # omegas_ij = []
    # omegas_ij = np.array([omegas_ges[0,0],omegas_eeps[0,0],omegas_efs[0,0]])
    # omegas_ij.append(np.array([omegas_ges[1,0],omegas_eeps[1,0],omegas_efs[1,0]]))
    # omegas_ij.append(np.array([omegas_ges[0,0],omegas_eeps[1,0],omegas_efs[1,0]]))
    # omegas_ij.append(np.array([
    
    #                 1              2              3                   4               5                   6              7*                8*
    # omega1 = [omegas_ges[0,0], omegas_ges[1,0], omegas_ges[1,0], omegas_ges[0,0], omegas_ges[1,0], omegas_ges[0,0], omegas_ges[0,0], omegas_ges[1,0]]   #  [e1,g0]  row<-col
    # omega2 = [omegas_ees[0,0], omegas_ees[1,1], omegas_ees[0,1], omegas_ees[1,0], omegas_ees[0,1], omegas_ees[1,0], omegas_ees[0,0], omegas_ees[1,1]]   #  [e2,e1]
    # omega3 = [omegas_efs[0,0], omegas_efs[1,1], omegas_efs[0,0], omegas_efs[0,1], omegas_efs[1,0], omegas_efs[1,1], omegas_efs[1,0], omegas_efs[0,1]]   #  [f1,e2]
    
    # omega2 = [omegas_gfs[0,0], omegas_gfs[1,0], omegas_gfs[0,0], omegas_gfs[0,0], omegas_gfs[1,0], omegas_gfs[1,0], omegas_gfs[1,0], omegas_gfs[0,0]]        
     
    
              #       1               7                   4               6              3              5                 8              2 
              # omegas_ges[0,0], omegas_ges[0,0], omegas_ges[0,0], omegas_ges[0,0], omegas_ges[1,0], omegas_ges[1,0], omegas_ges[1,0], omegas_ges[1,0]
              # omegas_ees[0,0], omegas_ees[0,0], omegas_ees[1,0], omegas_ees[1,0], omegas_ees[0,1], omegas_ees[0,1], omegas_ees[1,1], omegas_ees[1,1]
              # omegas_efs[0,0], omegas_efs[1,0], omegas_efs[0,1], omegas_efs[1,1], omegas_efs[0,0], omegas_efs[1,0], omegas_efs[0,1], omegas_efs[1,1]
              
              # omegas_gfs[0,0], omegas_gfs[1,0], omegas_gfs[0,0], omegas_gfs[1,0], omegas_gfs[0,0], omegas_gfs[1,0], omegas_gfs[0,0], omegas_gfs[1,0]
    
    
              # omegas_ges[i,0]
              # omegas_ees[j,i]
              # omegas_efs[k,j]
              # iterate over i, j, k should get them all! 
    
    # dqc
    # plt.scatter(omegas_123[:,0],omegas_123[:,2])
    # plt.plot(np.linspace(12e3,18e3,100),np.linspace(12e3,18e3,100),'k--')
    # plt.xlim(14e3,18e3)
    # plt.ylim(12e3,18e3)
    # matrix_plotter(omegas123.T,alpha_x=['1','7','4','6','3','5','8','2'],alpha_y=[r'$\omega_1$',r'$\omega_2$',r'$\omega_3$'],title='DQC pathways',figsize=[12,5])
    
    
    #nrprp
    # plt.scatter(omegas_123[:,0],omegas_123[:,2])
    # plt.plot(np.linspace(12e3,18e3,100),np.linspace(12e3,18e3,100),'k--')
    # plt.xlim(14e3,18e3)
    # plt.ylim(12e3,18e3)       
    # matrix_plotter(omegas123.T,alpha_x=['1','7','4','6','3','5','8','2'],alpha_y=[r'$\omega_1$',r'$\omega_2$',r'$\omega_3$'],title='NRP & RP pathways',figsize=[12,5])
    
    #%
    # =============================================================================
    # Build omega1, omega2, omega2 arrays for each pathway
    # =============================================================================
    # general scheme for building omegas: (loop over i,j,k)
        # omegas_ges[i,0]
        # omegas_ees[j,i]
        # omegas_efs[k,j]
    # =============================================================================
    # NOTE: currently these pathways are specific to the experiments that end inf |f><f|...
    # will need to adapt for it to be general for all possible 2DFS pathways
    
    
    #%
# old notes from overlap omega and alpha calculations
# shift omegas based on overlap with laser spectrum
# use inhomogenous linewidth as width of molecular absorption peak to calculate shifts (variable parameter)
# omega1 = (laser_omega * sigI**2 + np.array(omega1) * laser_sig_omega**2) / (sigI**2 + laser_sig_omega**2)
# omega2 = ((2*laser_omega) * sigI**2 + np.array(omega2) * (laser_sig_omega/2)**2) / (sigI**2 + (laser_sig_omega/2)**2)
# # need 2*laser omega for omega2 in the DQC calculation because omega2 is during t32 which has an |g><f| coherence = 2x energy of laser
# omega3 = (laser_omega * sigI**2 + np.array(omega3) * laser_sig_omega**2) / (sigI**2 + laser_sig_omega**2)

# alpha1 = gauss(np.array(omega1), laser_omega, laser_sig_omega,1)
# # alpha2 = gauss(np.array(omega2), laser_omega, laser_sig_omega,1) # things are funky about omega2... sort this out
# # alpha2 = gauss(np.array(omega2), 2*laser_omega - 2*monoC_omega, laser_sig_omega,1)
# alpha2 = gauss(np.array(omega2), 2*laser_omega, laser_sig_omega**2,1)
# alpha3 = gauss(np.array(omega3), laser_omega, laser_sig_omega,1)
# # alpha1 = alpha1/np.max(alpha1)
# # alpha2 = alpha2/np.max(alpha2) # things are funky about omega2... sort this out
# # alpha3 = alpha3/np.max(alpha3)
# overlap_alpha = alpha1 * alpha2 * alpha3 

# 20240304 CSA - omega2 needs to be shifted via omega_ge and omega_gep before calculating omega_eep
# also calculate alpha this way
# alpha2 = 1#gauss(np.array(omega2), laser_omega, laser_sig_omega,1)
# is this how to deal with alpha2?
# overlap_alpha = alpha1 * alpha2 * alpha3

#%%

def gauss(x, lam1, sig1, amp1):
    return amp1 * np.exp(-(x-lam1)**2 / (2 *sig1**2))


# =============================================================================
#  Calculate all possible energy differences given the eigenenergies 
#  => these give the omega21, omega_32, omega_43 values
# =============================================================================
def eigs2omegas(eigs_2PE, sigI, laser_omega, laser_sig_omega, selection_rules = 1, plot_mode = 0):
    #%
    # apply shift to energies based on overlap with laser at THIS POINT. And don't do it in the sim2Dcalc
    eigs = eigs_2PE # could remove the gn levels here? but no I'll leave them for now
    eigs_gs = eigs[:nVib]
    eigs_ges = eigs[nVib:2*nVib]
    eigs_gfs = eigs[2*nVib:]
    # shift eigenvalues based on overlap with laser
    eigs_ges_shifted = (laser_omega * sigI**2 + eigs_ges * laser_sig_omega**2)/(sigI**2 + laser_sig_omega**2)
    eigs_gfs_shifted = (2*laser_omega * sigI**2 + eigs_gfs * laser_sig_omega**2)/(sigI**2 + laser_sig_omega**2)
    
    # reconstruct the array of shifted eigenvalues
    eigs = np.array([eigs_gs, eigs_ges_shifted, eigs_gfs_shifted]).flatten()
    
    # omegas= np.real(np.subtract.outer(eigs_2PE,eigs_2PE))
    omegas= np.real(np.subtract.outer(eigs,eigs))
    if plot_mode == 1:
        matrix_plotter(omegas, alpha, alpha, title=r'Differences between eigenenergies ($cm^{-1}$ x$10^{3}$)',frac=0.8,figsize=[nEle*nVib,nEle*nVib],title_fontsize=20,label_fontsize=16,fontsize=22)

    # select the regions of the matrix corresponding to each set of transitions
    omegas_ges = omegas[nVib:nVib*(nEle-1),0:nVib]
    omegas_efs = omegas[(nEle-1)*nVib:nEle*nVib, nVib:2*nVib]
    omegas_eeps = omegas[nVib:2*nVib, nVib:(nEle-1)*nVib]
    omegas_gfs = omegas[(nEle-1)*nVib:nVib*nEle,0:nVib]
    # extract corresponding labels
    alpha_gs = alpha[0:nVib]
    alpha_es = alpha[nVib:nVib*(nEle-1)]
    alpha_fs = alpha[nVib*(nEle-1):nVib*nEle]
    # look at sub matrices for transitions (if you want)
    # matrix_plotter(omegas_ges, alpha_gs, alpha_es,title=r'Energies for $\Sigma_i \omega_{ge_i}$',frac=0.99,label_fontsize=18)
    # matrix_plotter(omegas_efs, alpha_es, alpha_fs,title=r'Energies for $\Sigma_{i,j} \omega_{e_if_j}$',frac=0.99,label_fontsize=18)
    # matrix_plotter(omegas_eeps*1e3, alpha_es, alpha_es,title=r"Energies for $\Sigma_{i,j} \omega_{e_ie'_j}$",frac=0.99,label_fontsize=18)
    # matrix_plotter(omegas_gfs*1e3, alpha_gs, alpha_fs,title=r'Energies for $\Sigma_{i} \omega_{gf_i}$',figsize=[12,12],frac=0.99,label_fontsize=18)
    
    # for t32 =/= 0 allow e->f to access all e states but still need to go odd->even or even->odd
    omegas_efs_all = omegas_efs 
    omegas_eeps_all = omegas_eeps
    
    
    # calculate corresponding alphas -- how to set these up for the beat freqs? 20240304 CSA
    # overlap_alphas_ges = np.real(gauss(omegas_ges, laser_omega, laser_sig_omega,1))
    # overlap_alphas_gfs = np.real(gauss(omegas_gfs, 2*laser_omega, laser_sig_omega**2,1))
    # overlap_alphas_efs = np.real(gauss(omegas_efs, laser_omega, laser_sig_omega,1))
    # 20240305 do I need to square the overlap alphas gfs?

    
    # =============================================================================
    # Impose selection rules
    # =============================================================================
    if selection_rules == 1:
        omegas_ges = omegas_ges[1::2,::2] # only select the omegas_ges (g -> odd e's)
        omegas_ges = omegas_ges[:,0].reshape(omegas_ges[:,0].shape[0],1) # we actually dont want any g other than g0
        if plot_mode == 1:
            matrix_plotter(omegas_ges, [alpha_gs[0]], np.array(alpha_es[1::2]),title=r'Energies for $\Sigma_i \omega_{ge_i}$',frac=0.99,label_fontsize=18)
        omegas_eeps = omegas_eeps[1::2,1::2] # (odd -> odd) ... is this right?
        # 20240304 CSA - am I allowing e3-e1 = (+) AND e1-e3 = (-)... yes
        if plot_mode == 1:
            matrix_plotter(omegas_eeps*1e3, alpha_es[1::2],alpha_es[1::2],title=r'Energies for $\Sigma_{i,j} \omega_{e_ie_j}$',frac=0.99,label_fontsize=18)
        # how to take care of selection rules for eep? should it only be the odd e's?
        omegas_gfs = omegas_gfs[::2, ::2] # even g -> even f
        omegas_gfs = omegas_gfs[:,0].reshape(omegas_ges[:,0].shape[0],1) # we actually dont want any g other than g0
        if plot_mode == 1:
            matrix_plotter(omegas_gfs*1e3, [alpha_gs[0]], alpha_fs[::2],title=r'Energies for $\Sigma_{i,j} \omega_{gf_j}$',frac=0.99,label_fontsize=18)
        omegas_efs = omegas_efs[::2,1::2] # select omegas_efs that we want (odd e's -> even f's)
        if plot_mode == 1:
            matrix_plotter(omegas_efs, alpha_es[1::2], alpha_fs[::2],title=r'Energies for $\Sigma_{i,j} \omega_{e_if_j}$',frac=0.99,label_fontsize=18)
    
        # allow for evens through energy transfer... initial rough approach
        omegas_eeps_evens = omegas_eeps_all[::2,::2] # (even e -> even e) ... use this for possible energy transfer?
        omegas_efs_evens = omegas_efs_all[1::2,::2] # (even e's -> odd f's)
        if plot_mode == 1:    
            matrix_plotter(omegas_eeps_evens*1e3, alpha_es[::2],alpha_es[::2],title=r'Energies for evens $\Sigma_{i,j} \omega_{e_ie_j}$',frac=0.99,label_fontsize=18)
            matrix_plotter(omegas_efs_evens*1e3, alpha_es[::2],alpha_fs[1::2],title=r'Energies for evens->odds $\Sigma_{i,j} \omega_{e_if_j}$',frac=0.99,label_fontsize=18)

    
    #%
    # omegas for DQC
    omegas123 = []
    for i in range(len(omegas_ges)):
        for j in range(len(omegas_eeps)):
            for k in range(len(omegas_efs)):
                # omegas123.append([omegas_ges[i,0], omegas_gfs[k,0], omegas_efs[k,j]])
                omegas123.append([omegas_ges[i,0], omegas_gfs[k,0], omegas_efs[k,j]])
    #                                omega1              omega2          omega3
    omegas123_dqc = np.array(omegas123)
    
    # ET = 1
    ET = globals()['ET']
    if ET == 1:
        eigs_2PE, vecs_2PE = Ham_2PE(epsilon0, omega0, lam) 
        omegas= np.real(np.subtract.outer(eigs_2PE,eigs_2PE))
        # select the regions of the matrix corresponding to each set of transitions
        omegas_ges_full = omegas[nVib:nVib*(nEle-1),0:nVib]
        omegas_ges_full = omegas_ges_full[:,0]
        omegas_efs_full = omegas[(nEle-1)*nVib:nEle*nVib, nVib:2*nVib]
        omegas_eeps_full = omegas[nVib:2*nVib, nVib:(nEle-1)*nVib]
        omegas_gfs_full = omegas[(nEle-1)*nVib:nVib*nEle,0:nVib]
        omegas_gfs_full = omegas_gfs_full[:,0]
        
        # allow for evens through energy transfer... initial rough approach
        omegas_ges_evens = omegas_ges_full[::2]#,::2]
        omegas_eeps_evens = omegas_eeps_full[::2,::2] # (even e -> even e) ... use this for possible energy transfer?
        omegas_efs_evenToOdd = omegas_efs_full[1::2,::2] # (even e's -> odd f's)
        omegas_gfs_evenToOdd = omegas_gfs_full[1::2]#,::2] # (even g's -> off f's) ... must proceed through virtual state who decays to even e
        # if plot_mode == 1:  
            # matrix_plotter(omegas_ges_evens*1e3, alpha_gs[::2],alpha_es[::2],title=r'Energies for evens $\Sigma_{i,j} \omega_{g_ie_j}$',frac=0.99,label_fontsize=18)
        #     matrix_plotter(omegas_eeps_evens*1e3, alpha_es[::2],alpha_es[::2],title=r'Energies for evens $\Sigma_{i,j} \omega_{e_ie_j}$',frac=0.99,label_fontsize=18)
            # matrix_plotter(omegas_efs_evens*1e3, alpha_es[::2],alpha_fs[1::2],title=r'Energies for evens->odds $\Sigma_{i,j} \omega_{e_if_j}$',frac=0.99,label_fontsize=18)
    
    
        ETarr = np.zeros(len(omegas123_dqc))
        # allow for pathways that undergo energy transfer during t32
        # if ET == 1:
        omegas123_dqc_wET = []
        # pathway_ges_evenStart = []
        nterms = int(nVib/2)
        # nterms =10
        for i in range(nterms):
            for j in range(nterms):
                for k in range(nterms):
                    # omegas123.append([omegas_ges[i,0], omegas_gfs[k,0], omegas_efs[k,j]])
        #                                omega1                     omega2                     omega3
                    omegas123_dqc_wET.append([omegas_ges[i,0], omegas_gfs_evenToOdd[k], omegas_efs_evenToOdd[k,j]])

        ETarr = np.hstack([ETarr, np.ones(len(omegas123_dqc_wET))])
        omegas123_dqc_wET = np.array(omegas123_dqc_wET)
        omegas123_dqc = np.vstack([omegas123_dqc, omegas123_dqc_wET])
    
    
    # plt.figure(figsize=[30,7])
    # xmax = 217
    # # xmax=450
    # spacing = np.linspace(0,xmax,xmax)
    # plt.hlines(0,0,xmax,'k')
    # plt.title('Pathways for DQC',fontsize=14)
    
    # for i in range(len(omegas_ges)):
    #     plt.hlines(omegas_ges[i],0,xmax,'k',linewidth=1.5,zorder=-1)
    #     plt.hlines(omegas_ges[i]+omegas_efs[i,i],0,xmax,'k',linewidth=1.5,zorder=-1)
    #     # plt.hlines(omegas_gfs[i],0,10,'gray','--')
    # for j in range(len(omegas_ges_full)):
    #     plt.hlines(omegas_ges_full[j],0,xmax,'gray','--',zorder=-1,linewidth=0.75)
    #     plt.hlines(omegas_gfs_full[j],0,xmax,'gray','--',zorder=-1, linewidth=0.75)
    # plt.ylim(0,max(omegas_gfs_full)*1.1)
    # j=0
    # for m in range(len(pathway_omegas)):
    #     j += 2
    #     plt.vlines(spacing[j], 0, omegas123_dqc[m,0],color='r',linestyle='-',linewidth=linewidths)
    #     j+=1
    #     plt.vlines(spacing[j], 0, omegas123_dqc[m,1],color='b',linestyle='-',linewidth=linewidths)
    #     j+=1
    #     plt.vlines(spacing[j],  omegas123_dqc[m,1] - omegas123_dqc[m,2],omegas123_dqc[m,1],color='c',linestyle='-',linewidth=linewidths)
    #     j+=2
    #     plt.vlines(spacing[j],-10,35000,'w',linewidth=10)
    #     j += 2
    
    
    # omegas for NRP & RP
    omegas123 = []
    for i in range(len(omegas_ges)):
        for j in range(len(omegas_eeps)):
            for k in range(len(omegas_efs)):
                omegas123.append([omegas_ges[i,0], omegas_eeps[j,i], omegas_efs[k,j]])
    #                                omega1              omega2          omega3
                # alphas123.append([overlap_alphas_ges[i,0], overlap_alphas_ges[j,0],overlap_alphas_efs[k,j]])
    #                               omega1           overlap for other omega ge brining to omega_ee',  omega3
    omegas123_nrprp = np.array(omegas123)
    
    
    ETarr = np.zeros(len(omegas123_nrprp))
    # allow for pathways that undergo energy transfer during t32
    if ET == 1:
        omegas123_nrprp_wET = []
        # pathway_ges_evenStart = []
        nterms = int(nVib/2)
        # nterms =10
        for i in range(nterms):
            for j in range(nterms):
                for k in range(nterms):
                    # omegas123.append([omegas_ges[i,0], omegas_gfs[k,0], omegas_efs[k,j]])
        #                                omega1                     omega2                     omega3
                    omegas123_nrprp_wET.append([omegas_ges[i,0], omegas_eeps_evens[j,i], omegas_efs_evenToOdd[k,j]])

        ETarr = np.hstack([ETarr, np.ones(len(omegas123_nrprp_wET))])
        omegas123_nrprp_wET = np.array(omegas123_nrprp_wET)
        omegas123_nrprp = np.vstack([omegas123_nrprp, omegas123_nrprp_wET])

    return omegas123_nrprp, omegas123_dqc, omegas_ges, omegas_efs, omegas_eeps, omegas_gfs, omegas, ETarr #, alphas123_dqc, alphas123_nrprp

ET = 1
laser_lam = 675
laser_fwhm = 30
sigI = 50
laser_omega = 10**7 / laser_lam
laser_fwhm_omega = 10**7/(laser_lam - (laser_fwhm/2)) - 10**7/(laser_lam + (laser_fwhm/2)) # fwhm in cm^-1
laser_sig_omega = laser_fwhm_omega / (2 * np.sqrt(2 * np.log(2))) # FWHM -> sigma (gaussian)
# omegas123_nrprp, omegas123_dqc,omegas_ges, omegas_efs, omegas_eeps, omegas_gfs, alphas123_dqc, alphas123_nrprp = eigs2omegas(eigs_2PE,sigI,laser_omega, laser_sig_omega)
omegas123_nrprp, omegas123_dqc,omegas_ges, omegas_efs, omegas_eeps, omegas_gfs, omegas, ET_arr = eigs2omegas(eigs_2PE,sigI,laser_omega, laser_sig_omega)

#%% notes on building mu arrays
# =============================================================================
# Need mu arrays
# =============================================================================

# Dipoles to multiply for all 8 pathways: put orientation factor in the orientation avg arr
# the order of the mu's sets up the interaction path for: 
#    |ket><bra|   |ket><bra|   |ket><bra|   |ket><bra|       |ket><bra|      |ket><bra|    |ket><bra|    |ket><bra|
    # gef,gef  gepfp, gepfp     gef, gepf    gepf,gef      gefp,gepfp       gepfp,gefp    gefp, gefp    gefp, gefp
# mu1 = [mu_ef,   mu_epfp,        mu_ef,       mu_epf,       mu_efp,          mu_epfp,      mu_efp,     mu_epf]
# mu2 = [mu_ge,   mu_gep,         mu_ge,       mu_gep,       mu_ge,           mu_gep,       mu_ge,      mu_gep]
# mu3 = [mu_ge,   mu_gep,         mu_gep,      mu_ge,        mu_gep,          mu_ge,        mu_ge,      mu_gep] 
# mu4 = [mu_ef,   mu_epfp,        mu_epf,      mu_ef,        mu_epfp,         mu_efp,       mu_efp,     mu_epf]
      
# mus = [mu_epf, mu_gep, mu_ge, mu_ef]

# mus = np.array([[mu_ge, mu_gep], # i=0, i=1
#                 [mu_ef, mu_efp], # j=0, k=0, k=1
#                 [mu_epf, mu_epfp]]) # j=1: k=0, k=1

# i=0, j=0, k=0
# mus_efs[0,0] * mus_ges[0] * mus_ges[0] * mus_efs[0,0]
#   mu_ef           mu_ge       mu_ge       mu_ef 

# i=0, j=1, k=0
# mus_efs[0,1] * mus_ges[1] * mus_ges[0] * mus_efs[0,0]
#   mu_epf           mu_gep       mu_ge       mu_ef 

# i=0, j=1, k=1
# mus_efs[1,1] * mus_ges[1] * mus_ges[0] * mus_efs[1,0]
#  mu_epfp          mu_gep      mu_ge       mu_efp

# i=0, j=0, k=1
# mus_efs[1,0] * mus_ges[0] * mus_ges[0] * mus_efs[1,0]
#  mu_efp           mu_ge       mu_ge       mu_efp

# i=1, j=0, k=1
# mus_efs[1,0] * mus_ges[0] * mus_ges[1] * mus_efs[1,1]
# mu_efp            mu_ge        mu_gep      mu_epfp


# General form: use test values for dipoles for now
# mu_ge = mu_ef = 1
# mu_gep = mu_epfp = 1.2
# mu_epf = (mu_ge + mu_ef) - mu_gep
# mu_efp = (mu_gep + mu_epfp) - mu_ge

# # need to generalize this part?
# mus_ges = np.array([mu_ge, mu_gep]) # i=0, i=1
# #           j,k =     0,0    0,1
# mus_efs = np.array([[mu_ef, mu_epf],        
#                     [mu_efp,mu_epfp]])
# #           j,k =     1,0    1,1

# mu_prods = []
# for i in range(len(omegas_ges)):
#     for j in range(len(omegas_eeps)):
#         for k in range(len(omegas_efs)):
#             mu_prods.append(mus_efs[k,j] * mus_ges[j] * mus_ges[i] * mus_efs[k,i])
# mu_prods = np.array(mu_prods)
#%%
# =============================================================================
# Generate mus for each pathway
# =============================================================================
# general form for mu arrays
# mus_efs[k,j] * mus_ges[j] * mus_ges[i] * mus_efs[k,i]
# =============================================================================
# def gen_mu_prods(mu1, mu2, mu3, omegas_ges, omegas_eeps, omegas_efs):
def gen_mu_prods(mus, omegas_ges, omegas_eeps, omegas_efs):
    # use test values for dipoles for now
    # mu_ge = mu_ef = mu1/2
    # mu_gep = mu_epfp = mu2/2
    # mu_epf = (mu_ge + mu_ef) - mu_gep
    # mu_efp = (mu_gep + mu_epfp) - mu_ge
    
    # # for nVib = 4
    # # need to generalize this part?
    # mus_ges = np.array([mu_ge, mu_gep]) # i=0, i=1
    # #           j,k =     0,0    0,1
    # mus_efs = np.array([[mu_ef, mu_epf],        
    #                     [mu_efp,mu_epfp]])
    # #           j,k =     1,0    1,1
    
    # # for nVib = 6
    # mus_ges = np.array([mu_ge, mu_gep, mu_ge]) # i=0, i=1, i=2
    
    #                                                 # j,k
    # mus_efs = np.array([[mu_ef, mu_epf, mu_ge],     # 0,0  0,1  0,2       
    #                     [mu_efp,mu_epfp, mu_ge],    # 1,0  1,1  1,2
    #                     [mu_efp,mu_epfp, mu_ge]])   # 2,0  2,1  2,2

    # mus = np.array([mu1, mu2, mu3])
    mu_ges = mus / 2
    mu_efs = np.zeros([len(mus), len(mus)])
    for n in range(len(mu_ges)):
        for m in range(len(mu_ges)):
            mu_efs[n,m] = mus[n] - mu_ges[m]
            
            
    # # calculate corresponding alphas -- how to set these up for the beat freqs? 20240304 CSA
    # overlap_alphas_ges = np.real(gauss(omegas_ges, laser_omega, laser_sig_omega,1)).flatten()
    # # overlap_alphas_gfs = np.real(gauss(omegas_gfs, 2*laser_omega, laser_sig_omega**2,1))
    # overlap_alphas_efs = np.real(gauss(omegas_efs, laser_omega, laser_sig_omega,1))
    # # 20240305 do I need to square the overlap alphas gfs?

    # overlap_alphas = []
    mu_prods = []
    for i in range(len(omegas_ges)):
        for j in range(len(omegas_eeps)):
            for k in range(len(omegas_efs)):
                # mu_prods.append(mu_efs[k,j] * mu_ges[j] * mu_ges[i] * mu_efs[k,i])
                mu_prods.append([mu_efs[j,k], mu_ges[k], mu_ges[i], mu_efs[j,i]])
         # pathway_omegas.append([omegas_efs[j,k], omegas_ges[k,0], omegas_ges[i,0], omegas_efs[j,i]])
                # overlap_alphas.append([overlap_alphas_efs[j,k], overlap_alphas_ges[k], overlap_alphas_ges[i], overlap_alphas_efs[j,i]])
                # print(i, j, k)
    mu_prods = np.array(mu_prods)
       
    # overlap_alphas = np.array(overlap_alphas)
    # ET =1
    ET = ET = globals()['ET']
    if ET == 1:
        mu_prods_wET = []
        for i in range(len(omegas_ges)):
            for j in range(len(omegas_eeps)):
                for k in range(len(omegas_efs)):
                    # mu_prods.append(mu_efs[k,j] * mu_ges[j] * mu_ges[i] * mu_efs[k,i])
                    mu_prods_wET.append([mu_efs[j,k], mu_ges[k], mu_ges[i], mu_efs[j,i]])
             # pathway_omegas.append([omegas_efs[j,k], omegas_ges[k,0], omegas_ges[i,0], omegas_efs[j,i]])
                    # overlap_alphas.append([overlap_alphas_efs[j,k], overlap_alphas_ges[k], overlap_alphas_ges[i], overlap_alphas_efs[j,i]])
                    # print(i, j, k)
        mu_prods_wET = np.array(mu_prods_wET)
        mu_prods = np.vstack([mu_prods, mu_prods_wET])
    
    
    
# =============================================================================
#     plotting diagnostics for looking at mu_prods
# =============================================================================
    # plt.figure(figsize=[30,7])
    # xmax = 217
    # spacing = np.linspace(0,xmax,xmax)
    # plt.hlines(0,0,xmax,'k')
    # plt.title('Pathways for RP',fontsize=14)
    # # for i in range(len(omegas_ges)):
    # plt.hlines(mu_ges[:],0,xmax,'k',linewidth=1.5)
    #     # plt.hlines(mu_ges[i]+mu_efs[i,i],0,xmax,'k',linewidth=1.5)
    # for i in range(len(omegas_efs)):
    #     for j in range(len(omegas_efs)):
    #         plt.hlines(mu_ges[i]+mu_efs[j,i],0,xmax,'k',linewidth=1.5)
    #     # plt.hlines(omegas_gfs[i],0,10,'gray','--')
    # j=0
    # for m in range(len(mu_prods)):
    #     j+=1                        # t1, idx2
    #     plt.vlines(spacing[j], 0, mu_prods[m,2],'r',linewidth=linewidths)
    #     j+=1                                             # t2, idx3
    #     plt.vlines(spacing[j], mu_prods[m,2], mu_prods[m,3] + mu_prods[m,2],'b',linewidth=linewidths)
    #     j += 1                      # t3, idx1
    #     plt.vlines(spacing[j], 0, mu_prods[m,1],'m',linewidth=linewidths)
    #     j += 1                                          # t4, idx0
    #     plt.vlines(spacing[j], mu_prods[m,1], mu_prods[m,0] + mu_prods[m,1],'c',linewidth=linewidths)
    #     j += 2
    #     plt.vlines(spacing[j],-0.1,4,'w',linewidth=10)
    #     j += 2
    
    mu_prods = np.product(mu_prods,axis=1)
    # overlap_alphas = np.product(overlap_alphas,axis=1)
    
    return mu_prods #, overlap_alphas
# generating the mu_ge, mu_gep, mu_efp etc. arrays only currently works for nEle=3, nVib=4 (with selection rules = 1) or nEle=3, nVib=2 (with selection rules = 0)
# need to figure out how to generalize this


#%
def gen_alpha_overlaps(laser_omega,laser_sig_omega,omegas_ges,omegas_eeps,omegas_efs, omegas):
    
    # =============================================================================
    # Calculate via overlap with beats
    # =============================================================================
    # calculate corresponding alphas -- how to set these up for the beat freqs? 20240304 CSA
    # overlap_alphas_ges = np.real(gauss(omegas_ges, laser_omega, laser_sig_omega,1))
    # overlap_alphas_gfs = np.real(gauss(omegas_gfs, 2*laser_omega, laser_sig_omega**2,1))
    # overlap_alphas_efs = np.real(gauss(omegas_efs, laser_omega, laser_sig_omega,1))
    # # 20240305 do I need to square the overlap alphas gfs?
    # # omegas for DQC
    # alphas_123 = []
    # for i in range(len(omegas_ges)):
    #     for j in range(len(omegas_eeps)):
    #         for k in range(len(omegas_efs)):
    #             # omegas123.append([omegas_ges[i,0], omegas_gfs[k,0], omegas_efs[k,j]])
    #             # omegas123.append([omegas_ges[i,0], omegas_gfs[k,0], omegas_efs[k,j]])
    # #                                omega1              omega2          omega3
    #             alphas_123.append([overlap_alphas_ges[i,0], overlap_alphas_gfs[k,0],overlap_alphas_efs[k,j]])
    # #                                   alpha1                     alpha2              alpha3
    # alphas123_dqc = np.array(alphas_123)
    
    # # omegas for NRP & RP
    # alphas123 = []
    # for i in range(len(omegas_ges)):
    #     for j in range(len(omegas_eeps)):
    #         for k in range(len(omegas_efs)):
    #             # omegas123.append([omegas_ges[i,0], omegas_eeps[j,i], omegas_efs[k,j]])
    # #                                omega1              omega2          omega3
    #             alphas123.append([overlap_alphas_ges[i,0], overlap_alphas_ges[j,0],overlap_alphas_efs[k,j]])
    # #                               omega1           overlap for other omega ge brining to omega_ee',  omega3
    # alphas123_nrprp = np.array(alphas123)
    
    
    # =============================================================================
    # Calculate via laser overlap with each of the four transitions involved in the interaction
    # =============================================================================
    # calculate corresponding alphas -- how to set these up for the beat freqs? 20240304 CSA
    overlap_alphas_ges = np.real(gauss(omegas_ges, laser_omega, 2*laser_sig_omega,1)).flatten()
    # matrix_plotter(1e6*overlap_alphas_ges/np.max(overlap_alphas_ges), alpha_gs, alpha_es,'overlap_alphas_ges')

    # overlap_alphas_gfs = np.real(gauss(omegas_gfs, 2*laser_omega, laser_sig_omega**2,1))
    overlap_alphas_efs = np.real(gauss(omegas_efs, laser_omega, 2*laser_sig_omega,1))
    # matrix_plotter(1e6*overlap_alphas_efs/np.max(overlap_alphas_efs), alpha_es, alpha_fs,'overlap_alphas_efs')
    # 20240305 do I need to square the overlap alphas gfs?

    overlap_alphas = []
    for i in range(len(omegas_ges)):
        for j in range(len(omegas_eeps)):
            for k in range(len(omegas_efs)):
                # mu_prods.append(mu_efs[k,j] * mu_ges[j] * mu_ges[i] * mu_efs[k,i])
                # mu_prods.append([mu_efs[j,k], mu_ges[k], mu_ges[i], mu_efs[j,i]])
         # pathway_omegas.append([omegas_efs[j,k], omegas_ges[k,0], omegas_ges[i,0], omegas_efs[j,i]])
                overlap_alphas.append([overlap_alphas_efs[k,j], overlap_alphas_ges[j], overlap_alphas_ges[i], overlap_alphas_efs[k,i]])
                # print(i, j, k)
    #             omegas123.append([omegas_ges[i,0], omegas_gfs[k,0], omegas_efs[k,j]])
    # #                                omega1              omega2          omega3
    #             pathway_omegas.append([omegas_efs[k,j], omegas_ges[j,0], omegas_ges[i,0], omegas_efs[k,i]])
    overlap_alphas = np.array(overlap_alphas)

    # ET = 1
    ET = globals()['ET']
    if ET == 1:
        # select the regions of the matrix corresponding to each set of transitions
        omegas_efs_full = omegas[(nEle-1)*nVib:nEle*nVib, nVib:2*nVib]
        omegas_efs_evens = omegas_efs_full[1::2,::2] # (even e's -> odd f's)

        overlap_alphas_efs_evens = np.real(gauss(omegas_efs_evens, laser_omega, 2*laser_sig_omega,1))
        overlap_alphas_wET = []
        for i in range(len(omegas_ges)):
            for j in range(len(omegas_eeps)):
                for k in range(len(omegas_efs)):
                    overlap_alphas_wET.append([overlap_alphas_efs_evens[k,j], overlap_alphas_ges[j], overlap_alphas_ges[i], overlap_alphas_efs_evens[k,i]])
                    # pathway_omegas_wET.append([omegas_efs_evens[k,j], omegas_ges[j,0], omegas_ges[i,0], omegas_efs_evens[k,i]])
        # pathway_omegas = np.vstack([pathway_omegas,pathway_omegas_wET])
        overlap_alphas_wET = np.array(overlap_alphas_wET)
        overlap_alphas = np.vstack([overlap_alphas, overlap_alphas_wET])


    overlap_alphas = np.product(overlap_alphas,axis=1)
    overlap_alphas /= np.max(overlap_alphas)
    
    # identify terms that have large alpha
    # alpha_idxs = np.where(overlap_alphas > 1e-3)[0]
    # anser: array([ 0,  3,  4, 12, 13]) ... look at those terms, are they the ones we see in data?
    
    return overlap_alphas
    
    
#%%

# =============================================================================
# Calculate 2D FFT
# =============================================================================
def FFT_2d(data, t21ax_rt, time_ax, monoC_lam, scan_type): # do the zeropadding more explicitly than above...
    data_0 = np.zeros([len(t21ax_rt), len(t21ax_rt)], dtype='complex')
    data_0[:len(data),:len(data)] = data
    # test_time = np.linspace(0,len(data_0),len(data_0))
    # plot2Dspectra(test_time, test_time, data_0, n_cont,ax_lim=[min(test_time),max(test_time)],timing_mode=timing_mode, title=r'test($\tau$) with '+timing_mode_str,domain='time')
    
    NFFT = len(data_0)
    dataFT = np.fft.fftshift(np.fft.fft2(data_0,[NFFT,NFFT]))
    # dataFT = np.rot90(dataFT,-1)
    # test_freq = np.linspace(0,len(dataFT),len(dataFT))
    # ax_lim = [450,550]
    # plot2Dspectra(test_freq, test_freq, dataFT, n_cont,ax_lim=ax_lim,timing_mode='t32 = 0', title=r'test($\tau$) with '+timing_mode_str,domain='freq')
    
    
    Fmono = 10**7 / (monoC_lam)
    c0 = 0.000299792458 # speed of light in mm/fs
    Ts = t21ax_rt[1] - t21ax_rt[0]
    # nfft = 256
    # nifft = 2048
    # Tsn = (Ts * nfft) / nifft
    Fs = 10 / (c0 * Ts)
    Faxis = Fs/2 * np.linspace(-1,1,num=NFFT)
    ax1 = (Faxis + Fmono) * 1e-3 
    ax2 = ax1
    
    if FT2D_mode == 1: # always take 2D transform
        if timing_mode == 't21 = 0':
            if scan_type == 'NRP_RP':
                Fmono = 0
                ax1 = (Faxis + Fmono) * 1e-3
            else: # => scan_type = 'DQC'
                Fmono = 2 * (10**7/monoC_lam)
                ax1 = (Faxis + Fmono) * 1e-3
        elif timing_mode == 't43 = 0':
            if scan_type == 'NRP_RP':
                Fmono = 0
                ax2 = (Faxis + Fmono) * 1e-3
                # print('t43=0 & NRP_RP => Fmono = '+str(Fmono))
            else: # => scan_type = 'DQC'
                Fmono = 2 * (10**7/monoC_lam)
                # print('t43=0 & DQC => Fmono = '+str(Fmono))
                ax2 = (Faxis + Fmono) * 1e-3
    
    
    return dataFT, ax1, ax2
# =============================================================================
# FFT for only one axis of a 2d plot
# =============================================================================
def FFT_1d(data, t21ax_rt, time_ax,monoC_lam): # do the zeropadding more explicitly than above...
    
    if timing_mode == 't32 = 0':
        data_0 = np.zeros([len(t21ax_rt), len(t21ax_rt)], dtype='complex')
    elif timing_mode == 't43 = 0':
        axis_num = 1
        data_0 = np.zeros([len(data), len(t21ax_rt)], dtype='complex')
    elif timing_mode == 't21 = 0':
        axis_num = 0
        data_0 = np.zeros([len(t21ax_rt),len(data)], dtype='complex')
    data_0[:len(data),:len(data)] = data
    NFFT = len(t21ax_rt)
    
    dataFT = np.fft.fftshift(np.fft.fft(data_0,NFFT,axis_num),axis_num)
    
    Fmono = 10**7 / (monoC_lam)
    c0 = 0.000299792458 # speed of light in mm/fs
    Ts = t21ax_rt[1] - t21ax_rt[0]
    # nfft = 256
    # nifft = 2048
    # Tsn = (Ts * nfft) / nifft
    Fs = 10 / (c0 * Ts)
    Faxis = Fs/2 * np.linspace(-1,1,num=NFFT)
    ax1 = (Faxis + Fmono) * 1e-3 
    
    
    if timing_mode == 't32 = 0':
        ax2 = ax1
    elif timing_mode == 't43 = 0':
        ax2 = time_ax
    elif timing_mode == 't21 = 0':
        ax2 = ax1
        ax1 = time_ax
    return dataFT, ax1, ax2
    
# used for windowing the 2D sim plots to make sure they go to zero at edges (avoid ringing in freq domain)
def delayedGaussian(x,c,s): 
    w = np.ones(np.shape(x));
    shifted = x-c;
    index = shifted > 0;
    w[index] = np.exp(-4.5*(shifted[index]**2)/(s**2));
    return w    

#%% Now we're ready to set up the function for the 2D calc!

# def sim2Dspec3(t21, laser_lam, laser_fwhm, mu1, mu2, Gam, sigI, monoC_lam, epsilon0, omega0, lam): # optimize these if using CD results doesn't work
# def sim2Dspec3(t21, laser_lam, laser_fwhm, mus, Gam, sigI, monoC_lam, epsilon0, omega0, lam): # optimize these if using CD results doesn't work
# def sim2Dspec3(t21, laser_lam, laser_fwhm, mu1, mu2, Gam, sigI, monoC_lam, epsilon0, omega0, lam): # optimize these if using CD results doesn't work
# def sim2Dspec3(t21, laser_lam, laser_fwhm, mu1, mu2, mu3, Gam, sigI, monoC_lam, epsilon0, omega0, lam): # optimize these if using CD results doesn't work
def sim2Dspec3(t21, laser_lam, laser_fwhm, mu1, mu2, Gam, sigI, monoC_lam, epsilon0, omega0, lam, ET_prob): # optimize these if using CD results doesn't work
# def sim2Dspec3(t21, laser_lam, laser_fwhm, mu1, mu2, mu3, Gam, sigI, monoC_lam, epsilon0, omega0, lam, ET_prob): # optimize these if using CD results doesn't work

    # mus = np.array([mu1, mu2])
    mus = np.array([mu1, mu2, mu3])
    timing_mode = globals()['timing_mode']
    FT2D_mode = globals()['FT2D_mode']
    ET = globals()['ET']
    if ET == 0:
        ET_prob = 0 # if turn ET off, then also don't allow terms with ET prob to contribute
    # else -> scale ET terms by ET prob (below)
    
    c0 = 0.000299792458 # mm / fs
    # t21 = 2 * np.pi * 10 * c0 * t21
    nubar2omega = 1/ ((10) / (2 * np.pi * c0)) # where c0 is in mm/fs
    #  multiplying by nubar2omega converts cm^-1 to fs^-1 
    # ==> use this conversion in the exponential of the response functions
    
    ##### Laser & monochromator parameters #####
    laser_sig =  laser_fwhm / (2 * np.sqrt(2 * np.log(2))) # fwhm -> sigma (gaussian)
    laser_fwhm_omega = 10**7/(laser_lam - (laser_fwhm/2)) - 10**7/(laser_lam + (laser_fwhm/2)) # fwhm in cm^-1
    laser_sig_omega = laser_fwhm_omega / (2 * np.sqrt(2 * np.log(2))) # FWHM -> sigma (gaussian)
    laser_omega = (10**7 / laser_lam) #/ 10**3 # convert wavelength to wavenumber
    
    monoC_omega = (10**7 / monoC_lam) #/ 10**3 # set monochromator wavelength and convert to wavenumber
    #############################################
    
    # calculate eigenvalues from 2PE hamiltonian
    eigs_2PE, vecs_2PE = Ham_2PE(epsilon0, omega0, lam, plot_mode=0)  
    
    # use eigenvalues to generate omegas for dqc, nrprp and each of the transitions
    # and calculate energy and intensity shifts due to laser overlap
    # omegas123_nrprp, omegas123_dqc,omegas_ges, omegas_efs, omegas_eeps, omegas_gfs, alphas123_dqc, alphas123_nrprp = eigs2omegas(eigs_2PE,sigI,laser_omega, laser_sig_omega) #,selection_rules = 1, plot_mode = 0)
    omegas123_nrprp, omegas123_dqc,omegas_ges, omegas_efs, omegas_eeps, omegas_gfs, omegas, ET_arr= eigs2omegas(eigs_2PE,sigI,laser_omega, laser_sig_omega) #,selection_rules = 1, plot_mode = 0)
    # 20240307 CSA - trying to calculate alphas with mu's instead (below)

    # generate mu products for each pathway (can this be further generalized?)
    # mu_prods = gen_mu_prods(mu1, mu2, omegas_ges, omegas_eeps, omegas_efs)
    # mu_prods = gen_mu_prods(mus, omegas_ges, omegas_eeps, omegas_efs)
    # mu_prods, overlap_alphas = gen_mu_prods(mus, omegas_ges, omegas_eeps, omegas_efs)
    mu_prods= gen_mu_prods(mus, omegas_ges, omegas_eeps, omegas_efs)
    # overlap_alphas /= np.max(overlap_alphas)
    # overlap_alphas = np.ones(overlap_alphas.shape)
    
    overlap_alphas = gen_alpha_overlaps(laser_omega,laser_sig_omega,omegas_ges,omegas_eeps,omegas_efs, omegas)
    # overlap_alphas = np.ones(overlap_alphas.shape)

    # =============================================================================
    # Write down the response functions
    # =============================================================================   
    # set up time arrays
    # Ntimesteps = int(np.max(np.abs(t21ax))/(t21ax_rt[1] - t21ax_rt[0]))
    # t21 = np.linspace(0,116.0805,num=Ntimesteps)
    ax_lim=[np.min(t21),np.max(t21)]
    t21 = np.tile(t21, (len(t21),1))
    time_size = len(t21)
    
    # set up the axes depending on which type of experiment we are doing (timing_mode)
    if timing_mode =='t32 = 0':
        t32 = 0 
        t43 = t21.T
        timing_mode_str = r'($\tau_{32}$ = 0)'
    elif timing_mode == 't43 = 0':
        t32 = t21.T
        t43 = 0
        timing_mode_str = r'($\tau_{43}$ = 0)'
    elif timing_mode == 't21 = 0':
        t32 = t21
        t43 = t21.T
        t21 = 0
        timing_mode_str = r'($\tau_{21}$ = 0)'
    
    # Assuming we are only probing one electronic dipole transition (EDTM) moment in the molecule (and its virtual and vibrational states)
    orient_avg_arr = np.ones(len(omegas123_nrprp)) * (1/5) 
    # 1/5 comes from the orientational average of the angle between the molecule EDTM   and the laser polarization (horizontal)


    cm_DQC = np.zeros([time_size,time_size],dtype='complex')
    cm_NRP = np.zeros([time_size,time_size],dtype='complex')
    cm_RP =  np.zeros([time_size,time_size],dtype='complex')
    nterms = len(omegas123_dqc) #len(mu_prods)
    for i in range(nterms):
    # for i in [ 0,  3,  4, 12, 13]:
        # start with DQC energies
        omega1, omega2, omega3 = omegas123_dqc[i,:]

        # omegas are shifted by laser overlap in eigs2omegas
        
        # alphas are: what is the amplitude of the overlap between molecule abs and laser at the newly shifted energies (directly above) 
        # overlap alphas now calculated in eigs2omegas
        # overlap_alpha = np.prod(alphas123_dqc[i,:]) # calculated with 3 delays with omegas
        overlap_alpha = overlap_alphas[i] # calculated via 4 interactions along with mu's
        # product of the alphas will scale this peak intensity
        
        # if ET is allowed, scale the ET peaks by ET_prob_arr value
        if ET_arr[i] != 0:
            overlap_alpha *= ET_prob
        # applying additional scaling to alpha with ET_prob because it essentially has the same cause but for a different purpose
        
        # subtract off monochromator reference frequency (because we are downsampling as explained in Tekavec 2006 & 2007)
        omega1 = (np.array(omega1) - monoC_omega)
        omega2 = (np.array(omega2) - (2 * monoC_omega) ) # factor of 2 came out of calculations
        omega3 = (np.array(omega3) - monoC_omega)

        # calculate response function for DQC
        cm_DQC += overlap_alpha * orient_avg_arr[i] * mu_prods[i] * np.exp(1j*nubar2omega*(omega3*t43 + omega2*t32 + omega1*t21)) \
                                                            * np.exp(-Gam*nubar2omega*(t43 + t32 + t21)) * np.exp(-(1/2)*(sigI*nubar2omega)**2*(t21 + t32 + t43)**2)
        
        # now for NRP & RP energies
        test, omega2, test = omegas123_nrprp[i,:] # "test" for 1 and 3 because same as dqc so we don't have to re-do the above calculations

        # overlap alphas now calculated in eigs2omegas
        # overlap_alpha = np.prod(alphas123_nrprp[i,:])

        ##### omega2 doesn't get monoC_omega subtracted off for NRP & RP... comes out of calculations
        
        # calculate response functions for NRP & RP
        cm_NRP += overlap_alpha * orient_avg_arr[i] * mu_prods[i] * np.exp(1j*nubar2omega*(omega3*t43 + omega2*t32 + omega1*t21)) \
                                                            * np.exp(-Gam*nubar2omega*(t43 + t32 + t21)) * np.exp(-(1/2)*(sigI*nubar2omega)**2*(t21 + t32 + t43)**2)
        
        cm_RP += overlap_alpha * orient_avg_arr[i] * mu_prods[i] * np.exp(1j*nubar2omega*(omega3*(-1*t43) + omega2*t32 + omega1*t21)) \
                                                            * np.exp(-Gam*nubar2omega*(t43 + t32 + t21)) * np.exp(-(1/2)*(sigI*nubar2omega)**2*(t21 + t32 + (-1*t43))**2)

    
    # set up the times to send out based on experiment being simulated
    if timing_mode =='t32 = 0':
        # t21 = array of times, t32 = 0, t43 = array of times
        t1_out = t21
        t2_out = t43
    elif timing_mode =='t43 = 0':
        # t21 = array of times, t32 = array of times, t43 = 0
        t1_out = t21
        t2_out = t32
    elif timing_mode =='t21 = 0':
        # t21 = 0, t32 = array of times, t43 = array of times
        t1_out = t32
        t2_out = t43
    
    ### this commented block is for testing time domain calculations
    # n_cont = 15
    # ax_lim = [np.min(t1_out), np.max(t1_out)]
    # plot2Dspectra(t1_out, t2_out, cm_DQC, n_cont,ax_lim=ax_lim, timing_mode=timing_mode,title=r'DQC($\tau$) with '+timing_mode_str,domain='time')#'($\tau_{32}$ = 0)',domain = 'time')
    # plot2Dspectra(t1_out, t2_out, cm_NRP, n_cont,ax_lim=ax_lim, timing_mode=timing_mode,title=r'NRP($\tau$) with '+timing_mode_str,domain='time')#'($\tau_{32}$ = 0)',domain = 'time')
    # plot2Dspectra(t1_out, t2_out, cm_RP, n_cont,ax_lim=ax_lim,timing_mode=timing_mode, title=r'RP($\tau$) with '+timing_mode_str,domain='time')#'($\tau_{32}$ = 0)',domain = 'time')
    # colormap = 'jet'


    ### this commented block is for testing frequency domain calculations
    # if timing_mode == 't32 = 0':
    #     time_ax = t21
    #     FT_dqc_temp, ax1_dqc, ax2_dqc = FFT_2d(cm_DQC, t21ax_rt, time_ax, monoC_lam,scan_type='DQC')
    #     FT_nrp_temp, ax1, ax2 = FFT_2d(cm_NRP, t21ax_rt, time_ax, monoC_lam,scan_type='NRP_RP')
    #     FT_rp_temp,  ax1, ax2 = FFT_2d(cm_RP,  t21ax_rt, time_ax, monoC_lam,scan_type='NRP_RP')
    #     FT_rp_temp = np.flipud(FT_rp_temp)
    #     FT_dqc_temp = np.flipud(FT_dqc_temp)
    #     ax_lim = [13.75,15.75]#[13.6, 15.75]
    #     n_cont=15
    #     plot2Dspectra(ax1_dqc, ax2_dqc, FT_dqc_temp, n_cont, ax_lim=ax_lim, timing_mode=timing_mode,title=r'DQC($\omega$) with '+timing_mode_str,domain='freq',scan_folder=scan_folder_dqc)#'($\tau_{32}$ = 0)',domain = 'freq')
    #     plot2Dspectra(ax1, ax2, FT_nrp_temp, n_cont, ax_lim=ax_lim, timing_mode=timing_mode,title=r'NRP($\omega$) with '+timing_mode_str,domain='freq',scan_folder=scan_folder_nrprp)#'($\tau_{32}$ = 0)',domain = 'freq')
    #     plot2Dspectra(ax1, ax2, FT_rp_temp,n_cont, ax_lim=ax_lim, timing_mode=timing_mode,title=r'RP($\omega$) with '+timing_mode_str,domain='freq',scan_folder=scan_folder_nrprp)#'($\tau_{32}$ = 0)',domain = 'freq')
    #     plt.show()

    

    # create a window to apply to time domain before taking FFT (adjust window time and steepness depending on delay space in experiment)
    xx = t21
    yy = t43
    # w = delayedGaussian(np.sqrt(xx**2 + yy**2),80, 10);
    # w = delayedGaussian(np.sqrt(xx**2 + yy**2),100, 10);
    w = delayedGaussian(np.sqrt(xx**2 + yy**2),np.max(t1_out)*0.8, 10); 
    #                                          onset,              steepness
    # plt.contourf(xx, yy, w, cmap='jet') # test window
    # plt.axis('scaled')
    # plt.colorbar()
    # plt.show
    
    cm_DQC = cm_DQC * w
    cm_NRP = cm_NRP * w
    cm_RP = cm_RP * w
    
    ### this commented block is for testing windowed time domain calculations
    # ax_lim = [np.min(t21),np.max(t21)]
    # ax_lim = [0, 116]
    # ax_lim = [0, np.max(t1_out)]
    # plot2Dspectra(t21, t43, cm_DQC, n_cont, ax_lim=ax_lim,timing_mode=timing_mode,title=r'DQC($\tau$) with ($\tau_{32}$ = 0)',domain = 'time')
    # plot2Dspectra(t21, t43, cm_NRP, n_cont, ax_lim=ax_lim,timing_mode=timing_mode,title=r'NRP($\tau$) with ($\tau_{32}$ = 0)',domain = 'time')
    # plot2Dspectra(t21, t43, cm_RP, n_cont, ax_lim=ax_lim,timing_mode=timing_mode,title=r'RP($\tau$) with ($\tau_{32}$ = 0)',domain = 'time')
    
    # take the FFT of the windowed time domain based on the experiment of interest
    if timing_mode == 't32 = 0':
        time_ax = t21
        FT_dqc, ax1, ax2 = FFT_2d(cm_DQC, t21ax_rt, time_ax, monoC_lam, scan_type='DQC')
        FT_nrp, ax1, ax2 = FFT_2d(cm_NRP, t21ax_rt, time_ax, monoC_lam, scan_type='NRP_RP')
        FT_rp,  ax1, ax2 = FFT_2d(cm_RP,  t21ax_rt, time_ax, monoC_lam, scan_type='NRP_RP')
        FT_rp = np.flipud(FT_rp)
        ax1_nrprp, ax2_nrprp = ax1_dqc, ax2_dqc = ax1, ax2
    elif timing_mode == 't43 = 0':
        if FT2D_mode == 0: # DON'T FFT along t32
            time_ax = t2_out[:,0]#t32[0,:]
            FT_dqc, ax1, ax2 = FFT_1d(cm_DQC, t21ax_rt, time_ax, monoC_lam)
            FT_nrp, ax1, ax2 = FFT_1d(cm_NRP, t21ax_rt, time_ax, monoC_lam)
            FT_rp,  ax1, ax2 = FFT_1d(cm_RP,  t21ax_rt, time_ax, monoC_lam)
            # FT_rp = np.flipud(FT_rp)
            ax1_nrprp, ax2_nrprp = ax1_dqc, ax2_dqc = ax1, ax2
        elif FT2D_mode == 1: # DO FFT along t32
            time_ax = t2_out[:,0] # I don't think this is used when FT2D_mode = 1, but the function needs it anyway
            FT_dqc, ax1, ax2 = FFT_2d(cm_DQC, t21ax_rt, time_ax, monoC_lam, scan_type='DQC') # the 2x shift is done in the function
            ax1_dqc, ax2_dqc = ax1, ax2
            FT_nrp, ax1, ax2 = FFT_2d(cm_NRP, t21ax_rt, time_ax, monoC_lam, scan_type='NRP_RP')
            FT_rp,  ax1, ax2 = FFT_2d(cm_RP,  t21ax_rt, time_ax, monoC_lam, scan_type='NRP_RP')
            FT_rp = np.flipud(FT_rp)
            ax1_nrprp, ax2_nrprp= ax1, ax2
    elif timing_mode == 't21 = 0':
        if FT2D_mode == 0: # DON'T FFT along t32
            time_ax = t32[0,:] 
            FT_dqc, ax1, ax2 = FFT_1d(cm_DQC, t21ax_rt, time_ax,monoC_lam)
            FT_nrp, ax1, ax2 = FFT_1d(cm_NRP, t21ax_rt, time_ax,monoC_lam)
            FT_rp,  ax1, ax2 = FFT_1d(cm_RP,  t21ax_rt, time_ax,monoC_lam)
            # FT_rp = np.flipud(FT_rp)
            ax1_nrprp, ax2_nrprp = ax1_dqc, ax2_dqc = ax1, ax2
        else: # DO FFT along t32
            time_ax = t32[0,:] # I don't think this is used when FT2D_mode = 1, but the function needs it anyway
            FT_dqc, ax1, ax2 = FFT_2d(cm_DQC, t21ax_rt, time_ax, monoC_lam, scan_type='DQC') # the 2x shift is done in the function
            ax1_dqc, ax2_dqc = ax1, ax2
            FT_nrp, ax1, ax2 = FFT_2d(cm_NRP, t21ax_rt, time_ax, monoC_lam, scan_type='NRP_RP')
            FT_rp,  ax1, ax2 = FFT_2d(cm_RP,  t21ax_rt, time_ax, monoC_lam, scan_type='NRP_RP')
            FT_rp = np.flipud(FT_rp)
            ax1_nrprp, ax2_nrprp= ax1, ax2
        
    return t1_out, t2_out, cm_DQC, cm_NRP, cm_RP, ax1_dqc, ax2_dqc, ax1_nrprp, ax2_nrprp, FT_dqc, FT_nrp, FT_rp

#%%
#%% Define chisquared function
# do we need to add a window function? (see example in simple2Dcalc_fromRbcode_CSA_v14a)
def chisq_calc(params):
    t21 = np.linspace(0,tmax,num=Ntimesteps) # generalize this so that when data coming in is over a different range this doesn't cause problems...
    # laser_lam, laser_fwhm, mu1, mu2, Gam, sigI, monoC_lam = params
    # laser_lam, laser_fwhm, mu1, mu2, Gam, sigI, monoC_lam, epsilon0, omega0, lam = params
    # laser_lam, laser_fwhm, mu1, mu2,mu3, Gam, sigI, monoC_lam, epsilon0, omega0, lam = params
    
    # nvib = 4
    laser_lam, laser_fwhm, mu1, mu2, Gam, sigI, monoC_lam, epsilon0, omega0, lam, ET_prob = params
    # nvib = 6
    laser_lam, laser_fwhm, mu1, mu2,mu3, Gam, sigI, monoC_lam, epsilon0, omega0, lam, ET_prob = params
    
    # epsilon0 = 29010.62480689866
    # omega0 = 149.93304096216116
    # lam = 2.69488731814141
    
    # t1_out, t2_out, cm_DQC, cm_NRP, cm_RP, ax1_dqc, ax2_dqc, ax1_nrprp, ax2_nrprp, FT_dqc, FT_nrp, FT_rp = sim2Dspec3(t21, laser_lam, laser_fwhm, mu1, mu2, Gam, sigI, monoC_lam, epsilon0, omega0, lam)
    t1_out, t2_out, cm_DQC, cm_NRP, cm_RP, ax1_dqc, ax2_dqc, ax1_nrprp, ax2_nrprp, FT_dqc, FT_nrp, FT_rp = sim2Dspec3(t21, laser_lam, laser_fwhm, mu1, mu2, mu3, Gam, sigI, monoC_lam, epsilon0, omega0, lam)

    # real part RP
    sim_denom = np.max(np.max(np.abs(np.real(FT_rp))))
    if sim_denom == 0: sim_denom = 1
    exp_denom = np.max(np.max(np.abs(np.real(RP_exp))))
    chisq_rp_re = (np.abs(np.real(FT_rp))/sim_denom - np.abs(np.real(RP_exp))/exp_denom)**2
    # imag part RP
    sim_denom = np.max(np.max(np.abs(np.imag(FT_rp))))
    if sim_denom == 0: sim_denom = 1
    exp_denom = np.max(np.max(np.abs(np.imag(RP_exp)))) 
    chisq_rp_im = (np.abs(np.imag(FT_rp))/sim_denom - np.abs(np.imag(RP_exp))/exp_denom)**2
    # recombine real + imag and take abs
    chisq_rp = np.abs(chisq_rp_re + 1j * chisq_rp_im)
    
    # real part NRP
    sim_denom = np.max(np.max(np.abs(np.real(FT_nrp))))
    if sim_denom == 0: sim_denom = 1
    exp_denom = np.max(np.max(np.abs(np.real(NRP_exp))))
    chisq_nrp_re =(np.abs(np.real(FT_nrp))/sim_denom - np.real(NRP_exp)/exp_denom)**2 
    # imag part NRP
    sim_denom = np.max(np.max(np.abs(np.imag(FT_nrp))))
    if sim_denom == 0: sim_denom = 1
    exp_denom = np.max(np.max(np.abs(np.imag(NRP_exp))))
    chisq_nrp_im =(np.abs(np.imag(FT_nrp))/sim_denom- np.abs(np.imag(NRP_exp))/exp_denom)**2
    # recombine real + imag and take abs
    chisq_nrp = np.abs(chisq_nrp_re + 1j * chisq_nrp_im)
    
    # real part DQC
    sim_denom = np.max(np.max(np.abs(np.real(FT_dqc))))
    if sim_denom == 0: sim_denom = 1
    exp_denom = np.max(np.max(np.abs(np.real(DQC_exp))))
    chisq_dqc_re =(np.abs(np.real(FT_dqc))/sim_denom - np.abs(np.real(DQC_exp))/exp_denom)**2
    # imag part DQC
    sim_denom = np.max(np.max(np.abs(np.imag(FT_dqc))))
    if sim_denom == 0: sim_denom = 1
    exp_denom = np.max(np.max(np.abs(np.imag(DQC_exp))))
    chisq_dqc_im =(np.abs(np.imag(FT_dqc))/sim_denom - np.abs(np.imag(DQC_exp))/exp_denom)**2
    # recombine real + imag and take abs
    chisq_dqc = np.abs(chisq_dqc_re + 1j * chisq_dqc_im)
    
    chisq_rp = np.mean(np.mean(chisq_rp))
    chisq_nrp = np.mean(np.mean(chisq_nrp))
    chisq_dqc = np.mean(np.mean(chisq_dqc))  
    
    # print('RP: '+str(np.round(chisq_rp,6))+' NRP: '+str(np.round(chisq_nrp,6))+' DQC: '+str(np.round(chisq_dqc,6)))
    # print(chisq_rp + chisq_nrp + chisq_dqc)
    return chisq_rp + chisq_nrp + chisq_dqc

    chisq_tot = chisq_nrp + chisq_dqc + chisq_rp
    
    return chisq_tot



#%%
# manually set timing mode instead of from data
# timing_mode ='t32 = 0'
# timing_mode ='t21 = 0'
# timing_mode ='t43 = 0'
print(timing_mode)

tmax = np.max(t21ax)

FT_2D_mode = 1 # If =1, always take 2D FT, if = 0, don't FT along t32 (so 1DFT for xy and yz experiments and 2DFT for xz experiments)
Ntimesteps = int(np.max(np.abs(t21ax))/(t21ax_rt[1] - t21ax_rt[0]))
# t21 = np.linspace(0,116.0805,num=Ntimesteps) # simulate at the retimed timesteps
t21 = np.linspace(0,tmax,num=Ntimesteps) # simulate at the retimed timesteps

#%
laser_lam = 675
laser_fwhm = 30
mu1 =4.2 #3.5
mu2 =3.8 
Gam =85 #85
sigI =45 #65 
monoC_lam = 700#701.9994
epsilon0 = 28500#28867 #28877 #27900 #29023
omega0 = 40 #105 #65 #81 #70 # 149.8
lam =3 #1.8 #1.41#1.26#2.6 #2.677

if timing_mode == 't32 = 0':
    ET = 0
    ET_prob = 0.5
    laser_lam = 675
    laser_fwhm = 30
    mu1 = 1
    mu2 = 1
    mu3 = 1
    Gam = 105 #80#70 #69 #85
    sigI = 20#60 #65 
    monoC_lam = 700
    epsilon0 = 28500 #28600 #
    omega0 = 55#50 #50#34 
    lam = 2.2#8 #3 #3.7#3.0 
elif timing_mode == 't21 = 0':
    ET = 1
    ET_prob = 1
    laser_lam = 674  # 675
    laser_fwhm = 30 #30 #100
    mu1 = 1 #3.5
    mu2 = 1
    mu3 = 1
    Gam = 70#85 #45 #85
    sigI = 10 #35#30 #65 
    monoC_lam = 700
    epsilon0 = 28600 #28286
    omega0 = 30
    lam = 2.2
elif timing_mode == 't43 = 0':
    ET = 1
    ET_prob = 1
    laser_lam = 675
    laser_fwhm = 25#30 #100
    mu1 = 1.2 #3.5
    mu2 = 1.2
    mu3 = 1
    Gam = 65#75 #45 #85
    sigI = 30#70#30 #65 
    monoC_lam = 700
    epsilon0 = 28600 #28286
    omega0 = 55 
    lam = 2.2 #3 


if ET == 0:
    ET_prob=0

# pack parameters into param array
# params = laser_lam, laser_fwhm, mu1, mu2, Gam, sigI, monoC_lam 
# params = laser_lam, laser_fwhm, mus, Gam, sigI, monoC_lam 
if nVib == 4:
    params = laser_lam, laser_fwhm, mu1,mu2, Gam, sigI, monoC_lam , epsilon0, omega0, lam, ET_prob
elif nVib == 6:
    params = laser_lam, laser_fwhm, mu1,mu2,mu3, Gam, sigI, monoC_lam , epsilon0, omega0, lam, ET_prob

# t1_out, t2_out, cm_DQC, cm_NRP, cm_RP, ax1_dqc, ax2_dqc, ax1_nrprp, ax2_nrprp, FT_dqc, FT_nrp, FT_rp = sim2Dspec3(t21, laser_lam, laser_fwhm, mu1, mu2, Gam, sigI, monoC_lam, epsilon0, omega0, lam)
# t1_out, t2_out, cm_DQC, cm_NRP, cm_RP, ax1_dqc, ax2_dqc, ax1_nrprp, ax2_nrprp, FT_dqc, FT_nrp, FT_rp = sim2Dspec3(t21, laser_lam, laser_fwhm, mus, Gam, sigI, monoC_lam, epsilon0, omega0, lam)
if nVib == 4:
    t1_out, t2_out, cm_DQC, cm_NRP, cm_RP, ax1_dqc, ax2_dqc, ax1_nrprp, ax2_nrprp, FT_dqc, FT_nrp, FT_rp = sim2Dspec3(t21, laser_lam, laser_fwhm, mu1, mu2, Gam, sigI, monoC_lam, epsilon0, omega0, lam, ET_prob)
elif nVib == 6:
    t1_out, t2_out, cm_DQC, cm_NRP, cm_RP, ax1_dqc, ax2_dqc, ax1_nrprp, ax2_nrprp, FT_dqc, FT_nrp, FT_rp = sim2Dspec3(t21, laser_lam, laser_fwhm, mu1, mu2, mu3, Gam, sigI, monoC_lam, epsilon0, omega0, lam, ET_prob)

#%
n_cont = 15
ax_lim = [14, 15.5]
# ax_lim = [13,16]
# ax_lim = [12.5, 16.5]
# ax_lim = [13.5,15.5]
# ax_lim = [13.75,15.75]
# =============================================================================
save_mode = 0
# =============================================================================
if timing_mode =='t32 = 0':
    timing_mode_str = r'($\tau_{32}$ = 0)'
elif timing_mode == 't43 = 0':
    timing_mode_str = r'($\tau_{43}$ = 0)'
elif timing_mode == 't21 = 0':
    timing_mode_str = r'($\tau_{21}$ = 0)'

# save_name = 'sim_' + scan_folder_dqc + '_tauDQC'
# plot2Dspectra(t1_out, t2_out, cm_DQC, n_cont,ax_lim=[min(t1_out[0,:]), max(t1_out[0,:])],timing_mode=timing_mode, title=r'DQC($\tau$) with '+timing_mode_str, domain='time',save_mode = save_mode, file_name = save_name,scan_folder=scan_folder_dqc)
# save_name = 'sim_' + scan_folder_nrprp + '_tauNRP'
# plot2Dspectra(t1_out, t2_out, cm_NRP, n_cont,ax_lim=[min(t1_out[0,:]), max(t1_out[0,:])],timing_mode=timing_mode, title=r'NRP($\tau$) with '+timing_mode_str, domain='time',save_mode = save_mode, file_name = save_name,scan_folder = scan_folder_nrprp)
# save_name = 'sim_' + scan_folder_nrprp + '_tauRP'
# plot2Dspectra(t1_out, t2_out, cm_RP, n_cont,ax_lim=[min(t1_out[0,:]), max(t1_out[0,:])], timing_mode=timing_mode,title=r'RP($\tau$) with '+timing_mode_str, domain='time',save_mode = save_mode, file_name = save_name,scan_folder=scan_folder_nrprp)

FT_dqc = FT_dqc/ np.max(np.max(FT_dqc))
FT_nrp = FT_nrp/ np.max(np.max(FT_nrp))
FT_rp = FT_rp /  np.max(np.max(FT_rp))

# save_mode = save_mode
# save_name = 'sim_' + scan_folder_dqc+'_FTdqc'
# plot2Dspectra(ax1_dqc, ax2_dqc, FT_dqc, n_cont,ax_lim, timing_mode=timing_mode,title=r'DQC($\omega$) with '+timing_mode_str, domain='freq',save_mode = save_mode, file_name = save_name,scan_folder=scan_folder_dqc)
# save_name = 'sim_' + scan_folder_nrprp+'_FTnrp'
# plot2Dspectra(ax1_nrprp, ax2_nrprp, FT_nrp, n_cont,ax_lim,timing_mode=timing_mode, title=r'NRP($\omega$) with '+timing_mode_str, domain='freq',save_mode = save_mode, file_name = save_name,scan_folder=scan_folder_nrprp)
# save_name = 'sim_' + scan_folder_nrprp+'_FTrp'
# plot2Dspectra(ax1_nrprp, ax2_nrprp, FT_rp, n_cont,ax_lim,timing_mode=timing_mode, title=r'RP($\omega$) with '+timing_mode_str, domain='freq',save_mode = save_mode, file_name = save_name,scan_folder=scan_folder_nrprp)

plot_resid_mode =0
if plot_resid_mode == 1: figsize=(16,4) 
else: figsize=(12,4)
save_mode = 0#save_mode
save_name = 'sim_' + scan_folder_dqc +'_FTdqcReComp'
plot_comparer(ax1_dqc, ax2_dqc, DQC_exp, FT_dqc, 'DQC',figsize=figsize,ax_lim = ax_lim ,save_mode = save_mode, file_name = save_name, scan_folder = scan_folder_dqc,plot_resid_mode=plot_resid_mode) #,weight_func_mode=weight_func_mode)
save_name = 'sim_' + scan_folder_nrprp +'_FTnrpReComp'
plot_comparer(ax1_nrprp,ax2_nrprp, NRP_exp, FT_nrp, 'NRP',figsize=figsize,ax_lim = ax_lim,save_mode = save_mode, file_name = save_name, scan_folder = scan_folder_nrprp,plot_resid_mode=plot_resid_mode)#,weight_func_mode=weight_func_mode)
save_name = 'sim_' + scan_folder_nrprp +'_FTrpReComp'
plot_comparer(ax1_nrprp,ax2_nrprp, RP_exp, FT_rp, 'RP',figsize=figsize,ax_lim = ax_lim,save_mode = save_mode, file_name = save_name,scan_folder = scan_folder_nrprp,plot_resid_mode=plot_resid_mode)#,weight_func_mode=weight_func_mode)




vals = params
if nVib == 4:
    param_labels = ['laser wavelength', 'laser fwhm', '|mu1|', '|mu2|','Gam_H','sigma_I','monoc wavelength', 'epsilon0', 'omega0', 'lam', 'ET_prob']
    param_unit_labels = [' nm', ' nm', ' D', ' D',' cm^(-1)',' cm^(-1)',' nm', ' cm^(-1)', ' cm^(-1)', '','']
elif nVib == 6:
    param_labels = ['laser wavelength', 'laser fwhm', '|mu1|', '|mu2|','|mu3|','Gam_H','sigma_I','monoc wavelength', 'epsilon0', 'omega0', 'lam', 'ET_prob']
    param_unit_labels = [' nm', ' nm', ' D', ' D','D',' cm^(-1)',' cm^(-1)',' nm', ' cm^(-1)', ' cm^(-1)', '', '']
print('  ')
print('**** Input parameters: ****')
print('         '+timing_mode+ '       ')
print('  ')
print('nVib:'+' '*(19-len('nVib'))+str(nVib))
print('ET:'+' '*(19-len('ET'))+str(bool(ET)))
print('  ')
for i in range(len(vals)):
    print(param_labels[i]+':'+' '*(19-len(param_labels[i]))+str(np.round(vals[i],4))+' '*(10-len(str(np.round(vals[i],4))))+param_unit_labels[i])
print('  ')


#%% Optimize 2D specs

# t21 = np.linspace(0,116.0805,num=Ntimesteps)
t21 = np.linspace(0,tmax,num=Ntimesteps)
# MNS bounds params
laser_lam_bounds = [673.5, 677.5] 
laser_fwhm_bounds = [28,32]
mu1_bounds = [1,4.2]
mu2_bounds = [1,4.2]
mu3_bounds = [1,4.2]#[2.8,4.2]
Gam_bounds = [10, 1e2] 
sigI_bounds = [10, 1e2] 
monoC_lam_bounds = [699.5, 700.5]
epsilon0_bounds = [28000, 30000]
omega0_bounds = [20,100]#800] # 81
lam_bounds = [1,5] #[1,4]
# bounds = [laser_lam_bounds, laser_fwhm_bounds, mu1_bounds, mu2_bounds, Gam_bounds, sigI_bounds, monoC_lam_bounds, epsilon0_bounds, omega0_bounds, lam_bounds]
# x0 = [laser_lam, laser_fwhm, mu1, mu2,Gam, sigI, monoC_lam, epsilon0, omega0, lam]
bounds = [laser_lam_bounds, laser_fwhm_bounds, mu1_bounds, mu2_bounds,mu3_bounds, Gam_bounds, sigI_bounds, monoC_lam_bounds, epsilon0_bounds, omega0_bounds, lam_bounds]
x0 = [laser_lam, laser_fwhm, mu1, mu2, mu3,Gam, sigI, monoC_lam, epsilon0, omega0, lam]

# t21 = np.linspace(0,116.0805,num=Ntimesteps) 
t21 = np.linspace(0,tmax,num=Ntimesteps)

if __name__ == '__main__':
    res = opt.differential_evolution(func=chisq_calc, 
                                      bounds=bounds,
                                      x0=x0,
                                      disp=True,
                                      # workers=1,
                                      # maxiter=1000,
                                      polish=True,
                                      # atol=1e-8, #1e-6, 1e-10,
                                      # tol = 1e-8, #1e-6, 10,
                                      # mutation=(0,1.9),
                                      # popsize=30,
                                      # updating='immediate',
                                      # strategy = 'best1exp')
                                      )
    if nVib == 4:
        laser_lam, laser_fwhm, mu1, mu2, Gam, sigI, monoC_lam,epsilon0, omega0, lam = res.x
    elif nVib == 6:
        laser_lam, laser_fwhm, mu1, mu2,mu3, Gam, sigI, monoC_lam,epsilon0, omega0, lam = res.x
    # t21 = t21.flatten()

#%
if nVib == 4:
    t1_out, t2_out, cm_DQC, cm_NRP, cm_RP, ax1_dqc, ax2_dqc, ax1_nrprp, ax2_nrprp, FT_dqc, FT_nrp, FT_rp = sim2Dspec3(t21, laser_lam, laser_fwhm, mu1, mu2, Gam, sigI, monoC_lam, epsilon0, omega0, lam)
elif nVib == 6:
    t1_out, t2_out, cm_DQC, cm_NRP, cm_RP, ax1_dqc, ax2_dqc, ax1_nrprp, ax2_nrprp, FT_dqc, FT_nrp, FT_rp = sim2Dspec3(t21, laser_lam, laser_fwhm, mu1, mu2,mu3, Gam, sigI, monoC_lam, epsilon0, omega0, lam)
#%
n_cont = 15
ax_lim = [14, 15.5]
# ax_lim = [13,16]
# =============================================================================
save_mode = 0
# =============================================================================
if timing_mode =='t32 = 0':
    timing_mode_str = r'($\tau_{32}$ = 0)'
elif timing_mode == 't43 = 0':
    timing_mode_str = r'($\tau_{43}$ = 0)'
elif timing_mode == 't21 = 0':
    timing_mode_str = r'($\tau_{21}$ = 0)'

save_name = 'sim_' + scan_folder_dqc + '_tauDQC'
plot2Dspectra(t1_out, t2_out, cm_DQC, n_cont,ax_lim=[min(t1_out[0,:]), max(t1_out[0,:])],timing_mode=timing_mode, title=r'DQC($\tau$) with '+timing_mode_str, domain='time',save_mode = save_mode, file_name = save_name,scan_folder=scan_folder_dqc)
save_name = 'sim_' + scan_folder_nrprp + '_tauNRP'
plot2Dspectra(t1_out, t2_out, cm_NRP, n_cont,ax_lim=[min(t1_out[0,:]), max(t1_out[0,:])], timing_mode=timing_mode,title=r'NRP($\tau$) with '+timing_mode_str, domain='time',save_mode = save_mode, file_name = save_name,scan_folder = scan_folder_nrprp)
save_name = 'sim_' + scan_folder_nrprp + '_tauRP'
plot2Dspectra(t1_out, t2_out, cm_RP, n_cont,ax_lim=[min(t1_out[0,:]), max(t1_out[0,:])], timing_mode=timing_mode,title=r'RP($\tau$) with '+timing_mode_str, domain='time',save_mode = save_mode, file_name = save_name,scan_folder=scan_folder_nrprp)

FT_dqc = FT_dqc/ np.max(np.max(FT_dqc))
FT_nrp = FT_nrp/ np.max(np.max(FT_nrp))
FT_rp = FT_rp /  np.max(np.max(FT_rp))

save_mode = save_mode
save_name = 'sim_' + scan_folder_dqc+'_FTdqc'
plot2Dspectra(ax1_dqc, ax2_dqc, FT_dqc, n_cont,ax_lim,timing_mode=timing_mode, title=r'DQC($\omega$) with '+timing_mode_str, domain='freq',save_mode = save_mode, file_name = save_name,scan_folder=scan_folder_dqc)
save_name = 'sim_' + scan_folder_nrprp+'_FTnrp'
plot2Dspectra(ax1_nrprp, ax2_nrprp, FT_nrp, n_cont,ax_lim,timing_mode=timing_mode, title=r'NRP($\omega$) with '+timing_mode_str, domain='freq',save_mode = save_mode, file_name = save_name,scan_folder=scan_folder_nrprp)
save_name = 'sim_' + scan_folder_nrprp+'_FTrp'
plot2Dspectra(ax1_nrprp, ax2_nrprp, FT_rp, n_cont,ax_lim,timing_mode=timing_mode, title=r'RP($\omega$) with '+timing_mode_str, domain='freq',save_mode = save_mode, file_name = save_name,scan_folder=scan_folder_nrprp)

save_mode = 0#save_mode
# save_name = 'sim_' + scan_folder_dqc +'_FTdqcReComp'
plot_comparer(ax1_dqc, ax2_dqc, DQC_exp, FT_dqc, 'DQC',figsize=(16,4),ax_lim = ax_lim ,save_mode = save_mode, file_name = save_name, scan_folder = scan_folder_dqc) #,weight_func_mode=weight_func_mode)
save_name = 'sim_' + scan_folder_nrprp +'_FTnrpReComp'
plot_comparer(ax1_nrprp,ax2_nrprp, NRP_exp, FT_nrp, 'NRP',figsize=(16,4),ax_lim = ax_lim,save_mode = save_mode, file_name = save_name, scan_folder = scan_folder_nrprp)#,weight_func_mode=weight_func_mode)
save_name = 'sim_' + scan_folder_nrprp +'_FTrpReComp'
plot_comparer(ax1_nrprp,ax2_nrprp, RP_exp, FT_rp, 'RP',figsize=(16,4),ax_lim = ax_lim,save_mode = save_mode, file_name = save_name,scan_folder = scan_folder_nrprp)#,weight_func_mode=weight_func_mode)

if nVib == 4:
    param_labels = ['laser wavelength', 'laser fwhm', '|mu1|', '|mu2|','Gam_H','sigma_I','monoc wavelength', 'epsilon0', 'omega0', 'lam']
    param_unit_labels = [' nm', ' nm', ' D', ' D',' cm^(-1)',' cm^(-1)',' nm', ' cm^(-1)', ' cm^(-1)', '']
elif nVib == 6:
    param_labels = ['laser wavelength', 'laser fwhm', '|mu1|', '|mu2|','|mu3|','Gam_H','sigma_I','monoc wavelength', 'epsilon0', 'omega0', 'lam']
    param_unit_labels = [' nm', ' nm', ' D', ' D','D',' cm^(-1)',' cm^(-1)',' nm', ' cm^(-1)', ' cm^(-1)', '']

vals = res.x
print('  ')
print('**** Optimized parameters: ****')
print('         '+timing_mode+ '       ')
print('  ')
for i in range(len(vals)):
    print(param_labels[i]+':'+' '*(19-len(param_labels[i]))+str(np.round(vals[i],4))+' '*(10-len(str(np.round(vals[i],4))))+param_unit_labels[i])
print('  ')
print('nVib = '+str(nVib))
print('ET = '+str(ET))


#%% Notes for developing pathway plotter

# =============================================================================
# Work on visualizing the resulting energy levels
# =============================================================================
# eigs_2PE, vecs_2PE = Ham_2PE(epsilon0, omega0, lam) 
# plt.figure(figsize=[3,10]);
# plt.hlines(eigs_2PE[:nVib]*1e-3,-0.5,0.5,linewidth=0.5,color='k');
# plt.hlines(eigs_2PE[nVib:2*nVib]*1e-3,lam-0.5,lam+0.5,linewidth=0.5,color='r');
# plt.hlines(eigs_2PE[2*nVib:]*1e-3,lam-0.5,lam+0.5,linewidth=0.5,color='b');
# arrow_style = {"head_width":0.5, "head_length":0.12,"linewidth":0.5, "color":"k"}
# plt.arrow(0,8,lam,0,**arrow_style)
# plt.text(0,8.2,r'$\lambda$ ='+str(np.round(lam,3)),fontsize=14)
# plt.ylabel(r'eigen-energies ($x10^3 cm^{-1})$',fontsize=14)
# plt.title('energy levels',fontsize=14)


# # np.diff(eigs_2PE.reshape(nEle,nVib))
# matrix_plotter(np.real(np.diff(eigs_2PE.reshape(nEle,nVib)))*1e3,alpha_x=[r'$n_1 - n_0$',r'$n_2 - n_1$',r'$n_3 - n_2$'],alpha_y=[r'$\Delta g$',r'$\Delta e$','$\Delta f$'],title='energy differences within electronic states')


# #%
# alpha_gs = alpha[0:nVib]
# alpha_es = alpha[nVib:nVib*(nEle-1)]
# alpha_fs = alpha[nVib*(nEle-1):nVib*nEle]
# matrix_plotter(omegas_ges, [alpha_gs[0]], np.array(alpha_es[1::2]),title=r'Energies for $\Sigma_i \omega_{ge_i}$',frac=0.99,label_fontsize=18)
# matrix_plotter(omegas_eeps, alpha_es[1::2],alpha_es[1::2],title=r'Energies for $\Sigma_{i,j} \omega_{e_ie_j}$',frac=0.99,label_fontsize=18)
# matrix_plotter(omegas_gfs, [alpha_gs[0]], alpha_fs[::2],title=r'Energies for $\Sigma_{i,j} \omega_{gf_j}$',frac=0.99,label_fontsize=18)
# matrix_plotter(omegas_efs, alpha_es[1::2], alpha_fs[::2],title=r'Energies for $\Sigma_{i,j} \omega_{e_if_j}$',frac=0.99,label_fontsize=18)






# omega0 = 100

# eigs_2PE, vecs_2PE = Ham_2PE(epsilon0, omega0, lam) 
# omegas123_nrprp, omegas123_dqc, omegas_ges, omegas_efs, omegas_eeps, omegas_gfs, alphas123_dqc, alphas123_nrprp = eigs2omegas(eigs_2PE, laser_omega, laser_sig_omega, selection_rules = 1, plot_mode = 0)
# omegas= np.real(np.subtract.outer(eigs_2PE,eigs_2PE))
# # select the regions of the matrix corresponding to each set of transitions
# omegas_ges_full = omegas[nVib:nVib*(nEle-1),0:nVib]
# omegas_ges_full = omegas_ges_full[:,0]
# omegas_efs_full = omegas[(nEle-1)*nVib:nEle*nVib, nVib:2*nVib]
# omegas_eeps_full = omegas[nVib:2*nVib, nVib:(nEle-1)*nVib]
# omegas_gfs_full = omegas[(nEle-1)*nVib:nVib*nEle,0:nVib]
# omegas_gfs_full = omegas_gfs_full[:,0]

# spacing = np.linspace(0,10,20)

# plt.figure(figsize=[10,7])
# plt.hlines(0,0,10,'k')
# for i in range(len(omegas_ges)):
#     plt.hlines(omegas_ges[i],0,10,'k',linewidth=1.5)
#     plt.hlines(omegas_ges[i]+omegas_efs[i,i],0,10,'k',linewidth=1.5)
#     # plt.hlines(omegas_gfs[i],0,10,'gray','--')
# for j in range(len(omegas_ges_full)):
#     plt.hlines(omegas_ges_full[j],0,10,'gray','--',zorder=-1,linewidth=0.75)
#     plt.hlines(omegas_gfs_full[j],0,10,'gray','--',zorder=-1, linewidth=0.75)

# linewidths = 3
# j=0
# for m in range(len(omegas_ges)):
#     j += 1
#     print(omegas_ges[m])
#     plt.vlines(spacing[j] ,0,omegas_ges[m],'r',linewidth=linewidths)
#     for n in range(len(omegas_ges)):
#         j += 1 
#         print(str(omegas_ges[m]) + ' -> ' + str(omegas_efs[n,m] + omegas_ges[m]))
#         plt.vlines(spacing[j]+0.1,omegas_ges[m], omegas_efs[n,m] + omegas_ges[m],'b',linewidth=linewidths)

# for n in range(len(omegas_gfs)):
#     j += 1
#     plt.vlines(spacing[j]+0.1,0, omegas_gfs[n] ,'g',linewidth=linewidths)
    
#%
# =============================================================================
# Can we plot out the pathways for each term? (like we did by hand for 2 accessible vibrational states in the monomer)
# =============================================================================

# # omegas for all four interactions
# omegas123 = []
# pathway_omegas = []
# nterms = int(nVib/2)
# # nterms =10
# for i in range(nterms):
#     for j in range(nterms):
#         for k in range(nterms):
#             # for l in range(nterms):
# #             omegas123.append([omegas_ges[i,0], omegas_gfs[k,0], omegas_efs[k,j]])
# # #                                omega1              omega2          omega3
# #             pathway_omegas.append([omegas_efs[j,k], omegas_ges[k,0], omegas_ges[i,0], omegas_efs[j,i]])
#             omegas123.append([omegas_ges[i,0], omegas_gfs[k,0], omegas_efs[k,j]])
# #                                omega1              omega2          omega3
#             pathway_omegas.append([omegas_efs[k,j], omegas_ges[j,0], omegas_ges[i,0], omegas_efs[k,i]])

#             # alphas_123.append([overlap_alphas_ges[i,0], overlap_alphas_gfs[k,0],overlap_alphas_efs[k,j]])
# #                                   alpha1                     alpha2              alpha3
# pathway_omegas = np.array(pathway_omegas)
# omegas123_dqc = np.array(omegas123)

#%
 
# plt.figure(figsize=[30,7])
# # plt.figure(figsize=[10,7])
# # xmax = 217
# xmax=450
# # xmax=100
# spacing = np.linspace(0,xmax,xmax)
# plt.hlines(0,0,xmax,'k')
# plt.title('Pathways for DQC',fontsize=14)

# for i in range(len(omegas_ges)):
#     plt.hlines(omegas_ges[i],0,xmax,'k',linewidth=1.5,zorder=-1)
#     plt.hlines(omegas_ges[i]+omegas_efs[i,i],0,xmax,'k',linewidth=1.5,zorder=-1)
#     # plt.hlines(omegas_gfs[i],0,10,'gray','--')
# for j in range(len(omegas_ges_full)):
#     plt.hlines(omegas_ges_full[j],0,xmax,'gray','--',zorder=-1,linewidth=0.75)
#     plt.hlines(omegas_gfs_full[j],0,xmax,'gray','--',zorder=-1, linewidth=0.75)
# plt.ylim(0,max(omegas_gfs_full)*1.1)

# # idx  0  1  2  3
# # t_i  4  3  1  2
# # DQC pathways
# j=0
# for m in range(len(pathway_omegas)):
# # for m in range(nterms):
#     j+=1                        # t1, idx2
#     plt.vlines(spacing[j], 0, pathway_omegas[m,2],'r',linewidth=linewidths)
#     j+=1                                             # t2, idx3
#     plt.vlines(spacing[j], pathway_omegas[m,2], pathway_omegas[m,3] + pathway_omegas[m,2],'b',linewidth=linewidths)
#     j += 1                      # t3, idx1
#     plt.vlines(spacing[j], 0, pathway_omegas[m,1],'m',linewidth=linewidths)
#     j += 1                                          # t4, idx0
#     plt.vlines(spacing[j], pathway_omegas[m,1], pathway_omegas[m,0] + pathway_omegas[m,1],'c',linewidth=linewidths)
#     j += 3
#     plt.vlines(spacing[j], 0, omegas123_dqc[m,0],color='r',linestyle='-',linewidth=linewidths*0.2)
#     j+=1
#     plt.vlines(spacing[j], 0, omegas123_dqc[m,1],color='b',linestyle='-',linewidth=linewidths*0.2)
#     j+=1
#     # plt.vlines(spacing[j], omegas123_dqc[m,0], omegas123_dqc[m,2]+omegas123_dqc[m,0],color='c',linestyle=':')
#     plt.vlines(spacing[j],  omegas123_dqc[m,1] - omegas123_dqc[m,2],omegas123_dqc[m,1],color='c',linestyle='-',linewidth=linewidths*0.2)
#     j+=2
#     plt.vlines(spacing[j],-10,35000,'w',linewidth=10)
#     j += 2
# plt.xlim(0,xmax)
# plt.ylim(-10,35000)










#%

# plt.figure(figsize=[30,7])
# # plt.figure(figsize=[10,7])
# # xmax = 217
# xmax=450
# # xmax=100
# xmax = len(pathway_omegas)*13
# spacing = np.linspace(0,len(pathway_omegas),xmax)
# plt.hlines(0,0,len(pathway_omegas),'k')
# plt.title('Pathways for DQC',fontsize=14)
# for i in range(len(omegas_ges)):
#     plt.hlines(omegas_ges[i],0,xmax,'k',linewidth=1.5,zorder=-1)
#     plt.hlines(omegas_ges[i]+omegas_efs[i,i],0,xmax,'k',linewidth=1.5,zorder=-1)
#     # plt.hlines(omegas_gfs[i],0,10,'gray','--')
# for j in range(len(omegas_ges_full)):
#     plt.hlines(omegas_ges_full[j],0,xmax,'gray','--',zorder=-1,linewidth=0.75)
#     plt.hlines(omegas_gfs_full[j],0,xmax,'gray','--',zorder=-1, linewidth=0.75)
# plt.ylim(0,max(omegas_gfs_full)*1.1)
# plt.xlim(0,max(spacing))
# # idx  0  1  2  3
# # t_i  4  3  1  2
# # DQC pathways
# j=0
# for m in range(len(pathway_omegas)):
# # for m in range(nterms):
#     j+=1                        # t1, idx2
#     plt.vlines(spacing[j], 0, pathway_omegas[m,2],'r',linewidth=linewidths)
#     j+=1                                             # t2, idx3
#     plt.vlines(spacing[j], pathway_omegas[m,2], pathway_omegas[m,3] + pathway_omegas[m,2],'b',linewidth=linewidths)
#     j += 1                      # t3, idx1
#     plt.vlines(spacing[j], 0, pathway_omegas[m,1],'m',linewidth=linewidths)
#     j += 1                                          # t4, idx0
#     plt.vlines(spacing[j], pathway_omegas[m,1], pathway_omegas[m,0] + pathway_omegas[m,1],'c',linewidth=linewidths)
#     j += 3
#     plt.vlines(spacing[j], 0, omegas123_dqc[m,0],color='r',linestyle='-',linewidth=linewidths*0.2)
#     j+=1
#     plt.vlines(spacing[j], 0, omegas123_dqc[m,1],color='b',linestyle='-',linewidth=linewidths*0.2)
#     j+=1
#     # plt.vlines(spacing[j], omegas123_dqc[m,0], omegas123_dqc[m,2]+omegas123_dqc[m,0],color='c',linestyle=':')
#     plt.vlines(spacing[j],  omegas123_dqc[m,1] - omegas123_dqc[m,2],omegas123_dqc[m,1],color='c',linestyle='-',linewidth=linewidths*0.2)
#     j+=2
#     plt.vlines(spacing[j],-10,35000,'w',linewidth=10)
#     j += 2
#     print(j)
# plt.xlim(0,len(pathway_omegas))
# plt.ylim(-100,35000)

#%
# plt.figure(figsize=[30,7])
# spacing = np.linspace(0,xmax,xmax)
# plt.hlines(0,0,xmax,'k')
# plt.title('Pathways for RP',fontsize=14)
# for i in range(len(omegas_ges)):
#     plt.hlines(omegas_ges[i],0,xmax,'k',linewidth=1.5)
#     plt.hlines(omegas_ges[i]+omegas_efs[i,i],0,xmax,'k',linewidth=1.5)
#     # plt.hlines(omegas_gfs[i],0,10,'gray','--')
# for j in range(len(omegas_ges_full)):
#     plt.hlines(omegas_ges_full[j],0,xmax,'gray','--',zorder=-1,linewidth=0.75)
#     plt.hlines(omegas_gfs_full[j],0,xmax,'gray','--',zorder=-1, linewidth=0.75)
# # idx  0  1  2  3
# # t_i  4  2  1  3
# # NRP pathways
# j=0
# for m in range(len(pathway_omegas)):
#     j+=1                        # t1, idx2
#     plt.vlines(spacing[j], 0, pathway_omegas[m,2],'r',linewidth=linewidths)
#     j += 1                      # t2, idx1
#     plt.vlines(spacing[j], 0, pathway_omegas[m,1],'m',linewidth=linewidths)
#     j += 1                      # t3, idx0
#     plt.vlines(spacing[j], pathway_omegas[m,1], pathway_omegas[m,0] + pathway_omegas[m,1],'c',linewidth=linewidths)
#     j+=1                        # t4, idx3
#     plt.vlines(spacing[j], pathway_omegas[m,2], pathway_omegas[m,3] + pathway_omegas[m,2],'b',linewidth=linewidths)
#     j += 3
#     plt.vlines(spacing[j], 0, omegas123_nrprp[m,0],color='r',linestyle='-',linewidth=linewidths*0.2)
#     j+=1
#     plt.vlines(spacing[j], omegas123_nrprp[m,0], omegas123_nrprp[m,1] + omegas123_nrprp[m,0] ,color='b',linestyle='-',linewidth=linewidths*0.2)
#     j+=1
#     # plt.vlines(spacing[j], omegas123_dqc[m,0], omegas123_dqc[m,2]+omegas123_dqc[m,0],color='c',linestyle=':')
#     plt.vlines(spacing[j],  omegas123_nrprp[m,1] + omegas123_nrprp[m,0],omegas123_nrprp[m,2]+omegas123_nrprp[m,1] + omegas123_nrprp[m,0],color='c',linestyle='-',linewidth=linewidths*0.2)
#     j += 2
#     plt.vlines(spacing[j],-3,35000,'w',linewidth=10)
#     j += 2
# plt.xlim(0,xmax)

#%
#%%
def pathway_plotter(epsilon0, omega0, lam):
    #%
    omega0 = 300 #150 # for visual purposes make this bigger than the actual value
    print('epsilon0 = '+str(epsilon0)+' omega0 = '+str(omega0)+' lam = '+str(lam))
    ET = globals()['ET']

    eigs_2PE, vecs_2PE = Ham_2PE(epsilon0, omega0, lam) 
    # omegas123_nrprp, omegas123_dqc, omegas_ges, omegas_efs, omegas_eeps, omegas_gfs, alphas123_dqc, alphas123_nrprp = eigs2omegas(eigs_2PE, sigI,laser_omega, laser_sig_omega, selection_rules = 1, plot_mode = 0)
    omegas123_nrprp, omegas123_dqc, omegas_ges, omegas_efs, omegas_eeps, omegas_gfs, omegas, ET_arr = eigs2omegas(eigs_2PE, sigI,laser_omega, laser_sig_omega, selection_rules = 1, plot_mode = 0)
    omegas= np.real(np.subtract.outer(eigs_2PE,eigs_2PE))
    # select the regions of the matrix corresponding to each set of transitions
    omegas_ges_full = omegas[nVib:nVib*(nEle-1),0:nVib]
    omegas_ges_full = omegas_ges_full[:,0]
    omegas_efs_full = omegas[(nEle-1)*nVib:nEle*nVib, nVib:2*nVib]
    omegas_eeps_full = omegas[nVib:2*nVib, nVib:(nEle-1)*nVib]
    omegas_gfs_full = omegas[(nEle-1)*nVib:nVib*nEle,0:nVib]
    omegas_gfs_full = omegas_gfs_full[:,0]

    # omegas for all four interactions
    # omegas123 = []
    pathway_omegas = []
    nterms = int(nVib/2)
    # nterms =10
    for i in range(nterms):
        for j in range(nterms):
            for k in range(nterms):
                # omegas123.append([omegas_ges[i,0], omegas_gfs[k,0], omegas_efs[k,j]])
    #                                omega1              omega2          omega3
                pathway_omegas.append([omegas_efs[k,j], omegas_ges[j,0], omegas_ges[i,0], omegas_efs[k,i]])
    
    
    # allow for evens through energy transfer... initial rough approach
    omegas_ges_evens = omegas_ges_full[::2]#,::2]
    omegas_eeps_evens = omegas_eeps_full[::2,::2] # (even e -> even e) ... use this for possible energy transfer?
    omegas_efs_evens = omegas_efs_full[1::2,::2] # (even e's -> odd f's)
    # if plot_mode == 1:  
        # matrix_plotter(omegas_ges_evens*1e3, alpha_gs[::2],alpha_es[::2],title=r'Energies for evens $\Sigma_{i,j} \omega_{g_ie_j}$',frac=0.99,label_fontsize=18)
        # matrix_plotter(omegas_eeps_evens*1e3, alpha_es[::2],alpha_es[::2],title=r'Energies for evens $\Sigma_{i,j} \omega_{e_ie_j}$',frac=0.99,label_fontsize=18)
        # matrix_plotter(omegas_efs_evens*1e3, alpha_es[::2],alpha_fs[1::2],title=r'Energies for evens->odds $\Sigma_{i,j} \omega_{e_if_j}$',frac=0.99,label_fontsize=18)

    pathway_omegas = np.array(pathway_omegas)
#%
    plt.figure(figsize=[30,8])
    # spacing = np.linspace(0,len(pathway_omegas),Nsteps)
    xmax = len(pathway_omegas)
    
    if ET == 1:
        Nsteps = 2*len(pathway_omegas)*13
        print('Nsteps = '+str(Nsteps))
        spacing = np.linspace(0,2*len(pathway_omegas),Nsteps)
        xmax = len(pathway_omegas) *2
    else:
        Nsteps = len(pathway_omegas)*13
        print('Nsteps = '+str(Nsteps))
        spacing = np.linspace(0,len(pathway_omegas),Nsteps)
        xmax = len(pathway_omegas)
    plt.hlines(0,0,xmax,'k')
    plt.title('Pathways for DQC',fontsize=18)
    for i in range(len(omegas_ges)):
        plt.hlines(omegas_ges[i],0,xmax,'k',linewidth=2.5,zorder=2)
        plt.hlines(omegas_ges[i]+omegas_efs[i,i],0,xmax,'k',linewidth=2.5,zorder=2)
        # plt.hlines(omegas_gfs[i],0,10,'gray','--')
    for j in range(len(omegas_ges_full)):
        plt.hlines(omegas_ges_full[j],0,xmax,'gray','--',zorder=-1,linewidth=1)
        plt.hlines(omegas_gfs_full[j],0,xmax,'gray','--',zorder=-1, linewidth=1)
    plt.ylim(0,max(omegas_gfs_full)*1.1)
    plt.xlim(0,max(spacing))
    arrow_linewidth = 2.5

    ETarr = np.zeros(len(pathway_omegas))
    # allow for pathways that undergo energy transfer during t32 (decrease by 1 virtual state)
    if ET == 1:
        pathway_omegas_wET = []
        pathway_ges_evenStart = []
        nterms = int(nVib/2)
        # nterms =10
        for i in range(nterms):
            for j in range(nterms):
                for k in range(nterms):
                    # omegas123.append([omegas_ges[i,0], omegas_gfs[k,0], omegas_efs[k,j]])
        #                                omega1              omega2          omega3
                    pathway_omegas_wET.append([omegas_efs_evens[k,j], omegas_ges[j,0], omegas_ges[i,0], omegas_efs_evens[k,i]])
                    # pathway_ges_evenStart.append([omegas_efs_evens[k,j],omegas_ges_evens[j], omegas_ges_evens[i], omegas_efs_evens[k,i]])
                    pathway_ges_evenStart.append([omegas_ges_evens[j], omegas_ges_evens[i]])
        pathway_ges_evenStart = np.array(pathway_ges_evenStart) # where to start the pathway for the energy transfer ef transition
        # pathway_omegas = pathway_omegas_wET # testing energy transfer
        pathway_omegas_wET = np.array(pathway_omegas_wET)
        pathway_omegas = np.vstack([pathway_omegas,pathway_omegas_wET])
        ETarr = np.hstack([ETarr, np.ones(len(pathway_omegas_wET))])
    
         # allow for energy transfer
        j=0
        for m in range(len(pathway_omegas)):
        # for m in range(nterms):
            j+=1                        # t1, idx2
            # plt.vlines(spacing[j], 0, pathway_omegas[m,2],'r',linewidth=linewidths)
            # plt.annotate(text='', xy=(spacing[j],pathway_omegas[m,2]), xytext=(spacing[j],0), arrowprops=dict(arrowstyle='->',linewidth=arrow_linewidth, shrinkA=0, shrinkB=0,color='r'))
            plt.annotate(text='', xy=(spacing[j],pathway_omegas[m,2]), xytext=(spacing[j],0), arrowprops=dict(arrowstyle='->', linewidth=arrow_linewidth, shrinkA=0, shrinkB=0,color='r'))
            j+=1                                             # t2, idx3
            # plt.vlines(spacing[j], pathway_omegas[m,2], pathway_omegas[m,3] + pathway_omegas[m,2],'b',linewidth=linewidths)
            if ETarr[m] == 0: # no ET
                plt.annotate(text='', xy=(spacing[j],pathway_omegas[m,3] + pathway_omegas[m,2]), xytext=(spacing[j],pathway_omegas[m,2]), arrowprops=dict(arrowstyle='->',linewidth=arrow_linewidth, shrinkA=0, shrinkB=0,color='b')) 
            elif ETarr[m] == 1: # w ET
                start_idx = int(m-len(pathway_omegas/2))
                plt.annotate(text='', xy=(spacing[j],pathway_omegas[m,3] + pathway_ges_evenStart[start_idx,1]  ), xytext=(spacing[j],pathway_ges_evenStart[start_idx,1]), arrowprops=dict(arrowstyle='->',linewidth=arrow_linewidth, shrinkA=0, shrinkB=0,color='b'))
            j += 1                      # t3, idx1
            # plt.vlines(spacing[j], 0, pathway_omegas[m,1],'m',linewidth=linewidths)
            plt.annotate(text='', xy=(spacing[j],pathway_omegas[m,1]), xytext=(spacing[j],0), arrowprops=dict(arrowstyle='->',linewidth=arrow_linewidth, shrinkA=0, shrinkB=0,color='m'))
            j += 1                                          # t4, idx0
            # plt.vlines(spacing[j], pathway_omegas[m,1], pathway_omegas[m,0] + pathway_omegas[m,1],'c',linewidth=linewidths)
            if ETarr[m] == 1: # w ET
                start_idx = int(m-len(pathway_omegas/2))
                plt.annotate(text='', xy=(spacing[j],pathway_omegas[m,0] + pathway_ges_evenStart[start_idx,0]), xytext=(spacing[j],pathway_ges_evenStart[start_idx,0]), arrowprops=dict(arrowstyle='->',linewidth=arrow_linewidth, shrinkA=0, shrinkB=0,color='c'))
            elif ETarr[m] == 0: # no ET
                plt.annotate(text='', xy=(spacing[j],pathway_omegas[m,0] + pathway_omegas[m,1]), xytext=(spacing[j],pathway_omegas[m,1]), arrowprops=dict(arrowstyle='->',linewidth=arrow_linewidth, shrinkA=0, shrinkB=0,color='c'))
    
            j += 3
            # plt.vlines(spacing[j], 0, omegas123_dqc[m,0],color='r',linestyle='-',linewidth=linewidths*0.2)
            plt.annotate(text='', xy=(spacing[j],omegas123_dqc[m,0]), xytext=(spacing[j],0), arrowprops=dict(arrowstyle='->',linewidth=arrow_linewidth*0.5, shrinkA=0, shrinkB=0,color='orange'))    
            j+=1
            # plt.vlines(spacing[j], 0, omegas123_dqc[m,1],color='b',linestyle='-',linewidth=linewidths*0.2)
            plt.annotate(text='', xy=(spacing[j],omegas123_dqc[m,1]), xytext=(spacing[j],0), arrowprops=dict(arrowstyle='->',linewidth=arrow_linewidth*0.5, shrinkA=0, shrinkB=0,color='purple'))    
            j+=1
            # plt.vlines(spacing[j], omegas123_dqc[m,0], omegas123_dqc[m,2]+omegas123_dqc[m,0],color='c',linestyle=':')
            # plt.vlines(spacing[j],  omegas123_dqc[m,1] - omegas123_dqc[m,2],omegas123_dqc[m,1],color='c',linestyle='-',linewidth=linewidths*0.2)
            plt.annotate(text='', xy=(spacing[j],omegas123_dqc[m,1]), xytext=(spacing[j],omegas123_dqc[m,1] - omegas123_dqc[m,2]), arrowprops=dict(arrowstyle='->',linewidth=arrow_linewidth*0.5, shrinkA=0, shrinkB=0,color='teal'))    
            j+=2
            plt.vlines(spacing[j],-100,35000,'w',linewidth=10)
            j += 2
            print(j)
    elif ET == 0:
        j=0
        for m in range(len(pathway_omegas)):
        # for m in range(nterms):
            j+=1                        # t1, idx2
            # plt.vlines(spacing[j], 0, pathway_omegas[m,2],'r',linewidth=linewidths)
            # plt.annotate(text='', xy=(spacing[j],pathway_omegas[m,2]), xytext=(spacing[j],0), arrowprops=dict(arrowstyle='->',linewidth=arrow_linewidth, shrinkA=0, shrinkB=0,color='r'))
            plt.annotate(text='', xy=(spacing[j],pathway_omegas[m,2]), xytext=(spacing[j],0), arrowprops=dict(arrowstyle='->', linewidth=arrow_linewidth, shrinkA=0, shrinkB=0,color='r'))
            j+=1                                             # t2, idx3
            # plt.vlines(spacing[j], pathway_omegas[m,2], pathway_omegas[m,3] + pathway_omegas[m,2],'b',linewidth=linewidths)
            plt.annotate(text='', xy=(spacing[j],pathway_omegas[m,3] + pathway_omegas[m,2]), xytext=(spacing[j],pathway_omegas[m,2]), arrowprops=dict(arrowstyle='->',linewidth=arrow_linewidth, shrinkA=0, shrinkB=0,color='b'))
            j += 1                      # t3, idx1
            # plt.vlines(spacing[j], 0, pathway_omegas[m,1],'m',linewidth=linewidths)
            plt.annotate(text='', xy=(spacing[j],pathway_omegas[m,1]), xytext=(spacing[j],0), arrowprops=dict(arrowstyle='->',linewidth=arrow_linewidth, shrinkA=0, shrinkB=0,color='m'))
            j += 1                                          # t4, idx0
            # plt.vlines(spacing[j], pathway_omegas[m,1], pathway_omegas[m,0] + pathway_omegas[m,1],'c',linewidth=linewidths)
            plt.annotate(text='', xy=(spacing[j],pathway_omegas[m,0] + pathway_omegas[m,1]), xytext=(spacing[j],pathway_omegas[m,1]), arrowprops=dict(arrowstyle='->',linewidth=arrow_linewidth, shrinkA=0, shrinkB=0,color='c'))
            j += 3
            # plt.vlines(spacing[j], 0, omegas123_dqc[m,0],color='r',linestyle='-',linewidth=linewidths*0.2)
            plt.annotate(text='', xy=(spacing[j],omegas123_dqc[m,0]), xytext=(spacing[j],0), arrowprops=dict(arrowstyle='->',linewidth=arrow_linewidth*0.5, shrinkA=0, shrinkB=0,color='orange'))    
            j+=1
            # plt.vlines(spacing[j], 0, omegas123_dqc[m,1],color='b',linestyle='-',linewidth=linewidths*0.2)
            plt.annotate(text='', xy=(spacing[j],omegas123_dqc[m,1]), xytext=(spacing[j],0), arrowprops=dict(arrowstyle='->',linewidth=arrow_linewidth*0.5, shrinkA=0, shrinkB=0,color='purple'))    
            j+=1
            # plt.vlines(spacing[j], omegas123_dqc[m,0], omegas123_dqc[m,2]+omegas123_dqc[m,0],color='c',linestyle=':')
            # plt.vlines(spacing[j],  omegas123_dqc[m,1] - omegas123_dqc[m,2],omegas123_dqc[m,1],color='c',linestyle='-',linewidth=linewidths*0.2)
            plt.annotate(text='', xy=(spacing[j],omegas123_dqc[m,1]), xytext=(spacing[j],omegas123_dqc[m,1] - omegas123_dqc[m,2]), arrowprops=dict(arrowstyle='->',linewidth=arrow_linewidth*0.5, shrinkA=0, shrinkB=0,color='teal'))    
            j+=2
            plt.vlines(spacing[j],-100,35000,'w',linewidth=10)
            j += 2
        # print(j)
    plt.xlim(-0.1,len(pathway_omegas))
    plt.ylim(-1000,33000)
    plt.xlabel('term number',fontsize=14)
    plt.ylabel(r'Energy $(cm^{-1})$',fontsize=14)
    
    # plt.annotate(text='', xy=(0.5,omegas123_dqc[m,1]), xytext=(0.5,0), arrowprops=dict(arrowstyle='->',linewidth=2, shrinkA=0, shrinkB=0))
    # plt.ylim(0,omegas123_dqc[m,1]*1.1)
    
       #%
     #%
    # omegas123_nrprp,
    if ET == 1:
        plt.figure(figsize=[30,8])
        Nsteps = len(pathway_omegas)*13
        spacing = np.linspace(0,len(pathway_omegas),Nsteps)
        xmax = len(pathway_omegas)
        plt.hlines(0,0,len(pathway_omegas),'k')
        plt.title('Pathways for NRP & RP',fontsize=18)
        for i in range(len(omegas_ges)):
            plt.hlines(omegas_ges[i],0,xmax,'k',linewidth=2.5,zorder=2)
            plt.hlines(omegas_ges[i]+omegas_efs[i,i],0,xmax,'k',linewidth=2.5,zorder=2)
            # plt.hlines(omegas_gfs[i],0,10,'gray','--')
        for j in range(len(omegas_ges_full)):
            plt.hlines(omegas_ges_full[j],0,xmax,'gray','--',zorder=-1,linewidth=1)
            plt.hlines(omegas_gfs_full[j],0,xmax,'gray','--',zorder=-1, linewidth=1)
        # idx  0  1  2  3
        # t_i  4  2  1  3
        # NRP pathways
        j=0
        for m in range(len(pathway_omegas)):
            j+=1                        # t1, idx2
            # plt.vlines(spacing[j], 0, pathway_omegas[m,2],'r',linewidth=linewidths)
            plt.annotate(text='', xy=(spacing[j],pathway_omegas[m,2]), xytext=(spacing[j],0), arrowprops=dict(arrowstyle='->', linewidth=arrow_linewidth, shrinkA=0, shrinkB=0,color='r'))
            j += 1                      # t2, idx1
            # plt.vlines(spacing[j], 0, pathway_omegas[m,1],'m',linewidth=linewidths)
            plt.annotate(text='', xy=(spacing[j],pathway_omegas[m,1]), xytext=(spacing[j],0), arrowprops=dict(arrowstyle='->', linewidth=arrow_linewidth, shrinkA=0, shrinkB=0,color='m'))
            j+=1                        # t3, idx3
            # plt.vlines(spacing[j], pathway_omegas[m,2], pathway_omegas[m,3] + pathway_omegas[m,2],'b',linewidth=linewidths)
            # plt.annotate(text='', xy=(spacing[j],pathway_omegas[m,3] + pathway_omegas[m,2]), xytext=(spacing[j],pathway_omegas[m,2]), arrowprops=dict(arrowstyle='->', linewidth=arrow_linewidth, shrinkA=0, shrinkB=0,color='b'))
            if ETarr[m] == 0: # no ET
                plt.annotate(text='', xy=(spacing[j],pathway_omegas[m,3] + pathway_omegas[m,2]), xytext=(spacing[j],pathway_omegas[m,2]), arrowprops=dict(arrowstyle='->',linewidth=arrow_linewidth, shrinkA=0, shrinkB=0,color='b')) 
            elif ETarr[m] == 1: # w ET
                start_idx = int(m-len(pathway_omegas/2))
                plt.annotate(text='', xy=(spacing[j],pathway_omegas[m,3] + pathway_ges_evenStart[start_idx,1]  ), xytext=(spacing[j],pathway_ges_evenStart[start_idx,1]), arrowprops=dict(arrowstyle='->',linewidth=arrow_linewidth, shrinkA=0, shrinkB=0,color='b'))
            
            j += 1                      # t4, idx0
            # plt.vlines(spacing[j], pathway_omegas[m,1], pathway_omegas[m,0] + pathway_omegas[m,1],'c',linewidth=linewidths)
            # plt.annotate(text='', xy=(spacing[j],pathway_omegas[m,0] + pathway_omegas[m,1]), xytext=(spacing[j],pathway_omegas[m,1]), arrowprops=dict(arrowstyle='->', linewidth=arrow_linewidth, shrinkA=0, shrinkB=0,color='c'))
            if ETarr[m] == 1: # w ET
                start_idx = int(m-len(pathway_omegas/2))
                plt.annotate(text='', xy=(spacing[j],pathway_omegas[m,0] + pathway_ges_evenStart[start_idx,0]), xytext=(spacing[j],pathway_ges_evenStart[start_idx,0]), arrowprops=dict(arrowstyle='->',linewidth=arrow_linewidth, shrinkA=0, shrinkB=0,color='c'))
            elif ETarr[m] == 0: # no ET
                plt.annotate(text='', xy=(spacing[j],pathway_omegas[m,0] + pathway_omegas[m,1]), xytext=(spacing[j],pathway_omegas[m,1]), arrowprops=dict(arrowstyle='->',linewidth=arrow_linewidth, shrinkA=0, shrinkB=0,color='c'))
    
            j += 3
            # plt.vlines(spacing[j], 0, omegas123_nrprp[m,0],color='r',linestyle='-',linewidth=linewidths*0.2)
            plt.annotate(text='', xy=(spacing[j],omegas123_nrprp[m,0]), xytext=(spacing[j],0), arrowprops=dict(arrowstyle='->', linewidth=arrow_linewidth*0.5, shrinkA=0, shrinkB=0,color='orange'))
            j+=1
            # plt.vlines(spacing[j], omegas123_nrprp[m,0], omegas123_nrprp[m,1] + omegas123_nrprp[m,0] ,color='b',linestyle='-',linewidth=linewidths*0.2)
            if ETarr[m] == 0:
                plt.annotate(text='', xy=(spacing[j],omegas123_nrprp[m,1] + omegas123_nrprp[m,0]), xytext=(spacing[j],omegas123_nrprp[m,0]), arrowprops=dict(arrowstyle='->', linewidth=arrow_linewidth*.5, shrinkA=0, shrinkB=0,color='purple'))
            elif ETarr[m] == 1:
                plt.annotate(text='', xy=(spacing[j],omegas123_nrprp[m,1] +pathway_ges_evenStart[start_idx,1]), xytext=(spacing[j],pathway_ges_evenStart[start_idx,1]), arrowprops=dict(arrowstyle='->', linewidth=arrow_linewidth*.5, shrinkA=0, shrinkB=0,color='purple'))
            j+=1
            # plt.vlines(spacing[j], omegas123_dqc[m,0], omegas123_dqc[m,2]+omegas123_dqc[m,0],color='c',linestyle=':')
            # plt.vlines(spacing[j],  omegas123_nrprp[m,1] + omegas123_nrprp[m,0],omegas123_nrprp[m,2]+omegas123_nrprp[m,1] + omegas123_nrprp[m,0],color='c',linestyle='-',linewidth=linewidths*0.2)
            if ETarr[m] == 0: # no ET
                plt.annotate(text='', xy=(spacing[j],omegas123_nrprp[m,2]+omegas123_nrprp[m,1] + omegas123_nrprp[m,0]), xytext=(spacing[j],omegas123_nrprp[m,1] +omegas123_nrprp[m,0]), arrowprops=dict(arrowstyle='->', linewidth=arrow_linewidth*0.5, shrinkA=0, shrinkB=0,color='teal'))
            elif ETarr[m] == 1: # w ET
                plt.annotate(text='', xy=(spacing[j],omegas123_nrprp[m,2]+omegas123_nrprp[m,1] + pathway_ges_evenStart[start_idx,1]), xytext=(spacing[j],omegas123_nrprp[m,1] +pathway_ges_evenStart[start_idx,1]), arrowprops=dict(arrowstyle='->', linewidth=arrow_linewidth*0.5, shrinkA=0, shrinkB=0,color='teal'))
            j += 2
            plt.vlines(spacing[j],-300,35000,'w',linewidth=10)
            j += 2
        plt.xlim(-0.1,len(pathway_omegas))
        plt.ylim(-1000,np.max(omegas_gfs_full)*1.1)#33000)
        plt.xlabel('term number',fontsize=14)
        plt.ylabel(r'Energy $(cm^{-1})$',fontsize=14)
        
    elif ET == 0:
        plt.figure(figsize=[30,10])
        xmax = len(pathway_omegas)*13
        spacing = np.linspace(0,len(pathway_omegas),xmax)
        plt.hlines(0,0,len(pathway_omegas),'k')
        plt.title('Pathways for NRP & RP',fontsize=18)
        for i in range(len(omegas_ges)):
            plt.hlines(omegas_ges[i],0,xmax,'k',linewidth=2.5,zorder=2)
            plt.hlines(omegas_ges[i]+omegas_efs[i,i],0,xmax,'k',linewidth=2.5,zorder=2)
            # plt.hlines(omegas_gfs[i],0,10,'gray','--')
        for j in range(len(omegas_ges_full)):
            plt.hlines(omegas_ges_full[j],0,xmax,'gray','--',zorder=-1,linewidth=1)
            plt.hlines(omegas_gfs_full[j],0,xmax,'gray','--',zorder=-1, linewidth=1)
        # idx  0  1  2  3
        # t_i  4  2  1  3
        # NRP pathways
        j=0
        for m in range(len(pathway_omegas)):
            j+=1                        # t1, idx2
            # plt.vlines(spacing[j], 0, pathway_omegas[m,2],'r',linewidth=linewidths)
            plt.annotate(text='', xy=(spacing[j],pathway_omegas[m,2]), xytext=(spacing[j],0), arrowprops=dict(arrowstyle='->', linewidth=arrow_linewidth, shrinkA=0, shrinkB=0,color='r'))
            j += 1                      # t2, idx1
            # plt.vlines(spacing[j], 0, pathway_omegas[m,1],'m',linewidth=linewidths)
            plt.annotate(text='', xy=(spacing[j],pathway_omegas[m,1]), xytext=(spacing[j],0), arrowprops=dict(arrowstyle='->', linewidth=arrow_linewidth, shrinkA=0, shrinkB=0,color='m'))
            j+=1                        # t3, idx3
            # plt.vlines(spacing[j], pathway_omegas[m,2], pathway_omegas[m,3] + pathway_omegas[m,2],'b',linewidth=linewidths)
            plt.annotate(text='', xy=(spacing[j],pathway_omegas[m,3] + pathway_omegas[m,2]), xytext=(spacing[j],pathway_omegas[m,2]), arrowprops=dict(arrowstyle='->', linewidth=arrow_linewidth, shrinkA=0, shrinkB=0,color='b'))
            j += 1                      # t4, idx0
            # plt.vlines(spacing[j], pathway_omegas[m,1], pathway_omegas[m,0] + pathway_omegas[m,1],'c',linewidth=linewidths)
            plt.annotate(text='', xy=(spacing[j],pathway_omegas[m,0] + pathway_omegas[m,1]), xytext=(spacing[j],pathway_omegas[m,1]), arrowprops=dict(arrowstyle='->', linewidth=arrow_linewidth, shrinkA=0, shrinkB=0,color='c'))
            j += 3
            # plt.vlines(spacing[j], 0, omegas123_nrprp[m,0],color='r',linestyle='-',linewidth=linewidths*0.2)
            plt.annotate(text='', xy=(spacing[j],omegas123_nrprp[m,0]), xytext=(spacing[j],0), arrowprops=dict(arrowstyle='->', linewidth=arrow_linewidth*0.5, shrinkA=0, shrinkB=0,color='orange'))
            j+=1
            # plt.vlines(spacing[j], omegas123_nrprp[m,0], omegas123_nrprp[m,1] + omegas123_nrprp[m,0] ,color='b',linestyle='-',linewidth=linewidths*0.2)
            plt.annotate(text='', xy=(spacing[j],omegas123_nrprp[m,1] + omegas123_nrprp[m,0]), xytext=(spacing[j],omegas123_nrprp[m,0]), arrowprops=dict(arrowstyle='->', linewidth=arrow_linewidth*.5, shrinkA=0, shrinkB=0,color='purple'))
            j+=1
            # plt.vlines(spacing[j], omegas123_dqc[m,0], omegas123_dqc[m,2]+omegas123_dqc[m,0],color='c',linestyle=':')
            # plt.vlines(spacing[j],  omegas123_nrprp[m,1] + omegas123_nrprp[m,0],omegas123_nrprp[m,2]+omegas123_nrprp[m,1] + omegas123_nrprp[m,0],color='c',linestyle='-',linewidth=linewidths*0.2)
            plt.annotate(text='', xy=(spacing[j],omegas123_nrprp[m,2]+omegas123_nrprp[m,1] + omegas123_nrprp[m,0]), xytext=(spacing[j],omegas123_nrprp[m,1] + omegas123_nrprp[m,0]), arrowprops=dict(arrowstyle='->', linewidth=arrow_linewidth*0.5, shrinkA=0, shrinkB=0,color='teal'))
            j += 2
            plt.vlines(spacing[j],-300,35000,'w',linewidth=10)
            j += 2
        plt.xlim(-0.1,len(pathway_omegas))
        plt.ylim(-1000,np.max(omegas_gfs_full)*1.1)#33000)
        plt.xlabel('term number',fontsize=14)
        plt.ylabel(r'Energy $(cm^{-1})$',fontsize=14)
        
        #%%
import matplotlib.path as mpath
import matplotlib.patches as mpatches
import matplotlib as mpl        
def curly_arrow(start, end, arr_size = 1, n = 5, col='gray', linew=1., width = 0.1):
  xmin, ymin = start
  xmax, ymax = end
  dist = np.sqrt((xmin - xmax)**2 + (ymin - ymax)**2)
  n0 = dist / (2 * np.pi)
  
  x = np.linspace(0, dist, 151) + xmin
  y = width * np.sin(n * x / n0) + ymin
  line = plt.Line2D(x,y, color=col, lw=linew)
  
  del_x = xmax - xmin
  del_y = ymax - ymin
  ang = np.arctan2(del_y, del_x)
  
  line.set_transform(mpl.transforms.Affine2D().rotate_around(xmin, ymin, ang) + ax.transData)
  ax.add_line(line)

  verts = np.array([[0,1],[0,-1],[2,0],[0,1]]).astype(float) * arr_size
  verts[:,1] += ymax
  verts[:,0] += xmax
  path = mpath.Path(verts)
  patch = mpatches.PathPatch(path, fc=col, ec=col)

  patch.set_transform(mpl.transforms.Affine2D().rotate_around(xmax, ymax, ang) + ax.transData)
  return patch      
        
fig, ax = plt.subplots()    
ax.add_patch(curly_arrow((0, 10), (5,8), n=10, arr_size=0.5,linew=1.5,width=0.5))
ax.set_xlim(0,30)
ax.set_ylim(0,30)        
        
#%
# fig,ax = plt.subplots(figsize=(30,10))
# ax.add_patch(curly_arrow((spacing[j-1], pathway_omegas[m,2]), (spacing[j],pathway_ges_evenStart[start_idx,1]), n=5, arr_size=0.01,linew=1,width=0.05,col='r'))
# plt.ylim(-1000,33000)
# plt.xlim(-0.1,len(pathway_omegas))
#%%

def pathway_plotter_v2(epsilon0, omega0, lam):
    #%%
    omega0 = 400 #150 # for visual purposes make this bigger than the actual value
    print('epsilon0 = '+str(epsilon0)+' omega0 = '+str(omega0)+' lam = '+str(lam))
    ET = globals()['ET']

    eigs_2PE, vecs_2PE = Ham_2PE(epsilon0, omega0, lam) 
    # omegas123_nrprp, omegas123_dqc, omegas_ges, omegas_efs, omegas_eeps, omegas_gfs, alphas123_dqc, alphas123_nrprp = eigs2omegas(eigs_2PE, sigI,laser_omega, laser_sig_omega, selection_rules = 1, plot_mode = 0)
    omegas123_nrprp, omegas123_dqc, omegas_ges, omegas_efs, omegas_eeps, omegas_gfs, omegas, ET_arr = eigs2omegas(eigs_2PE, sigI,laser_omega, laser_sig_omega, selection_rules = 1, plot_mode = 0)
    omegas= np.real(np.subtract.outer(eigs_2PE,eigs_2PE))
    # select the regions of the matrix corresponding to each set of transitions
    omegas_ges_full = omegas[nVib:nVib*(nEle-1),0:nVib]
    omegas_ges_full = omegas_ges_full[:,0]
    omegas_efs_full = omegas[(nEle-1)*nVib:nEle*nVib, nVib:2*nVib]
    omegas_eeps_full = omegas[nVib:2*nVib, nVib:(nEle-1)*nVib]
    omegas_gfs_full = omegas[(nEle-1)*nVib:nVib*nEle,0:nVib]
    omegas_gfs_full = omegas_gfs_full[:,0]

    # omegas for all four interactions
# =============================================================================
#   ETarr = [0,0,0,0] no relaxation
# =============================================================================
    # omegas123 = []
    pathway_omegas = []
    nterms = int(nVib/2)
    # nterms =10
    for i in range(nterms):
        for j in range(nterms):
            for k in range(nterms):
                # omegas123.append([omegas_ges[i,0], omegas_gfs[k,0], omegas_efs[k,j]])
    #                                omega1              omega2          omega3
                pathway_omegas.append([omegas_efs[k,j], omegas_ges[j,0], omegas_ges[i,0], omegas_efs[k,i]])
    
    pathway_omegas = np.array(pathway_omegas)
    ETarr = np.zeros(len(pathway_omegas))
    
    # allow for evens through energy transfer... 
    omegas_ges_evens = omegas_ges_full[::2]
    omegas_eeps_evens = omegas_eeps_full[::2,::2] # (even e -> even e) ... use this for possible energy transfer?
    omegas_efs_even2odd = omegas_efs_full[1::2,::2] # (even e's -> odd f's)
    omegas_efs_odd2odd = omegas_efs_full[1::2,1::2] # (odd e's -> odd f's) ...relaxation in f only
    omegas_efs_even2even = omegas_efs_full[::2,::2] # (even e's -> even f's) ... relaxation in e and f
    # if plot_mode == 1:  
        # matrix_plotter(omegas_ges_evens*1e3, alpha_gs[::2],alpha_es[::2],title=r'Energies for evens $\Sigma_{i,j} \omega_{g_ie_j}$',frac=0.99,label_fontsize=18)
        # matrix_plotter(omegas_eeps_evens*1e3, alpha_es[::2],alpha_es[::2],title=r'Energies for evens $\Sigma_{i,j} \omega_{e_ie_j}$',frac=0.99,label_fontsize=18)
        # matrix_plotter(omegas_efs_evens*1e3, alpha_es[::2],alpha_fs[1::2],title=r'Energies for evens->odds $\Sigma_{i,j} \omega_{e_if_j}$',frac=0.99,label_fontsize=18)
        # matrix_plotter(omegas_efs_odd2odd*1e3, alpha_es[1::2],alpha_fs[1::2],title=r'Energies for odds->odds $\Sigma_{i,j} \omega_{e_if_j}$',frac=0.99,label_fontsize=18)
        # matrix_plotter(omegas_efs_even2even*1e3, alpha_es[::2],alpha_fs[::2],title=r'Energies for evens->evens $\Sigma_{i,j} \omega_{e_if_j}$',frac=0.99,label_fontsize=18)
        # matrix_plotter(omegas_efs*1e3, alpha_es[1::2],alpha_fs[::2],title=r'Energies for evens->evens $\Sigma_{i,j} \omega_{e_if_j}$',frac=0.99,label_fontsize=18)
    
    # need to allow relaxation of f's also
    omegas_gfs_evens = omegas_gfs_full[::2]
    omegas_gfs_odds = omegas_gfs_full[1::2]
    # matrix_plotter(omegas_gfs_full*1e3,alpha_gs,alpha_fs,title='omega_gfs',frac=0.9)
    # matrix_plotter(omegas_gfs_evens*1e3,alpha_gs[::2],alpha_fs[::2],title='omega_gfs evens',frac=0.9)
    # matrix_plotter(omegas_gfs_odds*1e3,alpha_gs[1::2],alpha_fs[1::2],title='omega_gfs odds',frac=0.9)

    if ET == 1:
# =============================================================================
#       ETarr_eeRel = [1,0,1,0] both e-states relax (used to be ETarr == 1)
# =============================================================================
        # allow for relaxation of the e-states: 
        pathway_omegas_wET = []
        pathway_ges_evenStart = []
        nterms = int(nVib/2)
        for i in range(nterms):
            for j in range(nterms):
                for k in range(nterms):
                    # omegas123.append([omegas_ges[i,0], omegas_gfs[k,0], omegas_efs[k,j]])
        #                                omega1              omega2          omega3
                    pathway_omegas_wET.append([omegas_efs_even2odd[k,j], omegas_ges[j,0], omegas_ges[i,0], omegas_efs_even2odd[k,i]])
                    # pathway_ges_evenStart.append([omegas_efs_evens[k,j],omegas_ges_evens[j], omegas_ges_evens[i], omegas_efs_evens[k,i]])
                    pathway_ges_evenStart.append([omegas_ges_evens[j], omegas_ges_evens[i]])
        pathway_ges_evenStart = np.array(pathway_ges_evenStart) # where to start the pathway for the energy transfer ef transition
        # pathway_omegas = pathway_omegas_wET # testing energy transfer
        pathway_omegas_wET = np.array(pathway_omegas_wET)
        nET_eeRel_terms = len(pathway_omegas_wET)
        
       
# =============================================================================
#        ETarr_feRel = [0,2,1,0]  first f-state and second e-state (used to be ETarr == 2)
# =============================================================================
        # allow for relaxation of f-states: 
        pathway_omegas_wET_feRel = []
        pathway_ef_noRel_feRel = []
        pathway_ges_evenStart_feRel = []
        nterms = int(nVib/2)
        for i in range(nterms):
            for j in range(nterms):
                for k in range(nterms-1):
                    pathway_omegas_wET_feRel.append([omegas_efs_even2odd[k,j], omegas_ges[j,0], omegas_ges[i,0], omegas_efs[k+1,i]])
                    pathway_ef_noRel_feRel.append([omegas_efs_odd2odd[k,j], omegas_efs_odd2odd[k,i]])
                    pathway_ges_evenStart_feRel.append([omegas_ges_evens[j], omegas_ges_evens[i]])
        pathway_omegas_wET_feRel = np.array(pathway_omegas_wET_feRel)
        pathway_ef_noRel_feRel = np.array(pathway_ef_noRel_feRel) 
        pathway_ges_evenStart_feRel = np.array(pathway_ges_evenStart_feRel)
        nET_feRel_terms = len(pathway_omegas_wET_feRel)
        
 # =============================================================================
 #      ETarr_efRel = [1,0,0,2], first e-state and second f-state
 # =============================================================================
        # allow for relaxation of f-states: 
        pathway_omegas_wET_efRel = []
        pathway_ef_noRel_efRel = []
        pathway_ges_evenStart_efRel = []
        nterms = int(nVib/2)
        for i in range(nterms):
            for j in range(nterms):
                for k in range(nterms-1):
                # for k in range(1,nterms):
                    # omegas123.append([omegas_ges[i,0], omegas_gfs[k,0], omegas_efs[k,j]])
        #                                omega1              omega2          omega3
                    # pathway_omegas_wET_f.append([omegas_efs_odd2odd[k,j], omegas_ges[j,0], omegas_ges[i,0], omegas_efs_odd2odd[k,i]])
                    # pathway_ef_noRel_f.append([omegas_efs[k+1,j], omegas_efs[k+1,i]])
                    pathway_omegas_wET_efRel.append([omegas_efs[k+1,j], omegas_ges[j,0], omegas_ges[i,0], omegas_efs_even2odd[k,i]])
                    pathway_ef_noRel_efRel.append([omegas_efs_odd2odd[k,j], omegas_efs_odd2odd[k,i]])
                    pathway_ges_evenStart_efRel.append([omegas_ges_evens[j], omegas_ges_evens[i]])
                    # pathway_omegas_wET_f.append([omegas_efs[k,j], omegas_ges[j,0], omegas_ges[i,0], omegas_efs_odd2odd[k,i]])
                    # pathway_ef_noRel_f.append([omegas_efs[k,j], omegas_efs[k,i]])
        # pathway_ges_evenStart = np.array(pathway_ges_evenStart) # where to start the pathway for the energy transfer ef transition
        # pathway_omegas = pathway_omegas_wET # testing energy transfer
        pathway_omegas_wET_efRel = np.array(pathway_omegas_wET_efRel)
        pathway_ef_noRel_efRel = np.array(pathway_ef_noRel_efRel) 
        pathway_ges_evenStart_efRel = np.array(pathway_ges_evenStart_efRel)
        nET_efRel_terms = len(pathway_omegas_wET_efRel)
        
        
        
        # ETarr = np.hstack([np.zeros(len(pathway_omegas_wET)), np.ones(len(pathway_omegas_wET)),2*np.ones(len(pathway_omegas_wET_f))])
        # changing ETarr to be a Nx4 where the four columns each represent whether relaxation occurred
                            # no relaxation
        ETarr_noRel = np.array([[0,0,0,0]*len(np.array(pathway_omegas))]).reshape(len(np.array(pathway_omegas)),4)
                            # both e-states relax
        ETarr_eeRel = np.array([[1,0,1,0]*len(pathway_omegas_wET)]).reshape(len(pathway_omegas_wET),4)
                    # the first f state and second e state relax
        ETarr_feRel = np.array([[0,2,1,0]*len(pathway_omegas_wET_feRel)]).reshape(len(pathway_omegas_wET_feRel),4)
                    # the first e-state and second f-state relax
        ETarr_efRel = np.array([[1,0,0,2]*len(pathway_omegas_wET_efRel)]).reshape(len(pathway_omegas_wET_efRel),4)
        # I don't think that [0,2,0,2] will contribute since we aren't allowing the energy that the system relaxes to to be coherent
        ETarr = np.vstack([ETarr_noRel,ETarr_eeRel,ETarr_feRel,ETarr_efRel])

        pathway_omegas = np.vstack([pathway_omegas, pathway_omegas_wET, pathway_omegas_wET_feRel,pathway_omegas_wET_efRel])




    # =============================================================================
    # # Add in wiggly lines for relaxation using curly_arrow function
    # =============================================================================
    #%%
    arrow_linewidth = 2.5
    # DQC
    # allow for energy transfer
    fig,ax = plt.subplots(figsize=[50,10])
    Nsteps = len(pathway_omegas)*13
    spacing = np.linspace(0,len(pathway_omegas),Nsteps)
    xmax = len(pathway_omegas)
    plt.hlines(0,0,len(pathway_omegas),'k')
    plt.title('Pathways for DQC',fontsize=18)
    for i in range(len(omegas_ges)):
        plt.hlines(omegas_ges[i],0,xmax,'k',linewidth=2.5,zorder=2)
        plt.hlines(omegas_ges[i]+omegas_efs[i,i],0,xmax,'k',linewidth=2.5,zorder=2)
        # plt.hlines(omegas_gfs[i],0,10,'gray','--')
    for j in range(len(omegas_ges_full)):
        plt.hlines(omegas_ges_full[j],0,xmax,'gray','--',zorder=-1,linewidth=1)
        plt.hlines(omegas_gfs_full[j],0,xmax,'gray','--',zorder=-1, linewidth=1)
    j=0
    for m in range(len(pathway_omegas)):
    # for m in range(20,40):
    # for m in range(nterms):
        # print(m)
        j+=1                        # t1, idx2
        plt.annotate(text='', xy=(spacing[j],pathway_omegas[m,2]), xytext=(spacing[j],0), arrowprops=dict(arrowstyle='->', linewidth=arrow_linewidth, shrinkA=0, shrinkB=0,color='r'))
        j+=1                
                                   # t2, idx3
        if np.product(ETarr[m] == [0,0,0,0]):#ETarr[m] == 0: # no ET
            plt.annotate(text='', xy=(spacing[j],pathway_omegas[m,3] + pathway_omegas[m,2]), xytext=(spacing[j],pathway_omegas[m,2]), arrowprops=dict(arrowstyle='->',linewidth=arrow_linewidth, shrinkA=0, shrinkB=0,color='b')) 
        elif np.product(ETarr[m] == [1,0,1,0]): #ETarr[m] == 1: # w ET
            start_idx = m%nET_eeRel_terms #8 #int(m-len(pathway_omegas/3))
            plt.annotate(text='', xy=(spacing[j],pathway_omegas[m,3] + pathway_ges_evenStart[start_idx,1]  ), xytext=(spacing[j],pathway_ges_evenStart[start_idx,1]), arrowprops=dict(arrowstyle='->',linewidth=arrow_linewidth, shrinkA=0, shrinkB=0,color='b'))
            ax.add_patch(curly_arrow((spacing[j-1], pathway_omegas[m,2]), (spacing[j],pathway_ges_evenStart[start_idx,1]), n=5, arr_size=0.01,linew=1,width=0.05,col='r'))
        elif np.product(ETarr[m] == [0,2,1,0]): #ETarr[m] == 2:
            start_idx = m%nET_feRel_terms
            plt.annotate(text='', xy=(spacing[j],pathway_omegas[m,3] + pathway_omegas[m,2] ), xytext=(spacing[j],pathway_omegas[m,2]), arrowprops=dict(arrowstyle='->',linewidth=arrow_linewidth, shrinkA=0, shrinkB=0,color='b'))
            # ax.add_patch(curly_arrow((spacing[j], pathway_ef_noRel_f[m%8,1]+pathway_omegas[m,2]), (spacing[j], pathway_omegas[m,3] + pathway_omegas[m,2]), n=5, arr_size=0.01,linew=1,width=0.05,col='m')) 
            ax.add_patch(curly_arrow((spacing[j], pathway_omegas[m,3] + pathway_omegas[m,2]), (spacing[j+1], pathway_ef_noRel_feRel[start_idx,1]+pathway_omegas[m,2]), n=5, arr_size=0.01,linew=1,width=0.05,col='b')) 
        elif np.product(ETarr[m] == [1,0,0,2]): #ETarr[m] == 2:
            start_idx = m%nET_efRel_terms
            # plt.annotate(text='', xy=(spacing[j],pathway_omegas[m,0] + pathway_omegas[m,1]), xytext=(spacing[j],pathway_omegas[m,1]), arrowprops=dict(arrowstyle='->',linewidth=arrow_linewidth, shrinkA=0, shrinkB=0,color='c'))
            plt.annotate(text='', xy=(spacing[j],pathway_omegas[m,3] + pathway_ges_evenStart_efRel[start_idx,1]), xytext=(spacing[j],pathway_ges_evenStart_efRel[start_idx,1]), arrowprops=dict(arrowstyle='->',linewidth=arrow_linewidth, shrinkA=0, shrinkB=0,color='b'))
            # ax.add_patch(curly_arrow((spacing[j], pathway_ef_noRel_f[start_idx,0]+pathway_omegas[m,1]), (spacing[j], pathway_omegas[m,0] + pathway_omegas[m,1]), n=5, arr_size=0.01,linew=1,width=0.05,col='c')) 
            ax.add_patch(curly_arrow((spacing[j-1], pathway_omegas[m,2]), (spacing[j], pathway_ges_evenStart_efRel[start_idx,1]), n=5, arr_size=0.01,linew=1,width=0.05,col='r')) 
            
        j += 1                      # t3, idx1
        plt.annotate(text='', xy=(spacing[j],pathway_omegas[m,1]), xytext=(spacing[j],0), arrowprops=dict(arrowstyle='->',linewidth=arrow_linewidth, shrinkA=0, shrinkB=0,color='m'))
        j += 1  
                                    # t4, idx0
        # plt.vlines(spacing[j], pathway_omegas[m,1], pathway_omegas[m,0] + pathway_omegas[m,1],'c',linewidth=linewidths)
        if np.product(ETarr[m] == [0,0,0,0]): #ETarr[m] == 0: # no ET
            plt.annotate(text='', xy=(spacing[j],pathway_omegas[m,0] + pathway_omegas[m,1]), xytext=(spacing[j],pathway_omegas[m,1]), arrowprops=dict(arrowstyle='->',linewidth=arrow_linewidth, shrinkA=0, shrinkB=0,color='c'))
        elif np.product(ETarr[m] == [1,0,1,0]): #ETarr[m] == 1: # w ET
            start_idx = m%nET_eeRel_terms #8 #int(m-len(pathway_omegas/3))
            # print(start_idx)
            plt.annotate(text='', xy=(spacing[j],pathway_omegas[m,0] + pathway_ges_evenStart[start_idx,0]), xytext=(spacing[j],pathway_ges_evenStart[start_idx,0]), arrowprops=dict(arrowstyle='->',linewidth=arrow_linewidth, shrinkA=0, shrinkB=0,color='c'))
            ax.add_patch(curly_arrow((spacing[j-1], pathway_omegas[m,1]), (spacing[j], pathway_ges_evenStart[start_idx,0]), n=5, arr_size=0.01,linew=1,width=0.05,col='m')) 
        elif np.product(ETarr[m] == [0,2,1,0]): #ETarr[m] == 2:
            start_idx = m%nET_feRel_terms
            # plt.annotate(text='', xy=(spacing[j],pathway_omegas[m,0] + pathway_omegas[m,1]), xytext=(spacing[j],pathway_omegas[m,1]), arrowprops=dict(arrowstyle='->',linewidth=arrow_linewidth, shrinkA=0, shrinkB=0,color='c'))
            plt.annotate(text='', xy=(spacing[j],pathway_omegas[m,0] + pathway_ges_evenStart_feRel[start_idx,0]), xytext=(spacing[j],pathway_ges_evenStart_feRel[start_idx,0]), arrowprops=dict(arrowstyle='->',linewidth=arrow_linewidth, shrinkA=0, shrinkB=0,color='c'))
            # ax.add_patch(curly_arrow((spacing[j], pathway_ef_noRel_f[start_idx,0]+pathway_omegas[m,1]), (spacing[j], pathway_omegas[m,0] + pathway_omegas[m,1]), n=5, arr_size=0.01,linew=1,width=0.05,col='c')) 
            ax.add_patch(curly_arrow((spacing[j-1], pathway_omegas[m,1]), (spacing[j], pathway_ges_evenStart_feRel[start_idx,0]), n=5, arr_size=0.01,linew=1,width=0.05,col='m')) 
        elif np.product(ETarr[m] == [1,0,0,2]): #ETarr[m] == 2:
            start_idx = m%nET_efRel_terms
            # plt.annotate(text='', xy=(spacing[j],pathway_omegas[m,0] + pathway_omegas[m,1]), xytext=(spacing[j],pathway_omegas[m,1]), arrowprops=dict(arrowstyle='->',linewidth=arrow_linewidth, shrinkA=0, shrinkB=0,color='c'))
            plt.annotate(text='', xy=(spacing[j],pathway_omegas[m,0] + pathway_omegas[m,1]), xytext=(spacing[j], pathway_omegas[m,1]), arrowprops=dict(arrowstyle='->',linewidth=arrow_linewidth, shrinkA=0, shrinkB=0,color='c'))
            # ax.add_patch(curly_arrow((spacing[j], pathway_ef_noRel_f[start_idx,0]+pathway_omegas[m,1]), (spacing[j], pathway_omegas[m,0] + pathway_omegas[m,1]), n=5, arr_size=0.01,linew=1,width=0.05,col='c')) 
            # ax.add_patch(curly_arrow((spacing[j+2], pathway_ges_evenStart_feRel[start_idx,0]+pathway_ef_noRel_efRel[start_idx,0]), (spacing[j], pathway_omegas[m,0] + pathway_omegas[m,1]), n=5, arr_size=0.01,linew=1,width=0.05,col='c')) 
            ax.add_patch(curly_arrow((spacing[j], pathway_omegas[m,0] + pathway_omegas[m,1]), (spacing[j+2], pathway_ef_noRel_feRel[start_idx,0]+pathway_omegas[m,1]), n=5, arr_size=0.01,linew=1,width=0.05,col='c')) 

        j += 3
        # plt.annotate(text='', xy=(spacing[j],omegas123_dqc[m,0]), xytext=(spacing[j],0), arrowprops=dict(arrowstyle='->',linewidth=arrow_linewidth*0.5, shrinkA=0, shrinkB=0,color='orange'))    
        j+=1
        # plt.annotate(text='', xy=(spacing[j],omegas123_dqc[m,1]), xytext=(spacing[j],0), arrowprops=dict(arrowstyle='->',linewidth=arrow_linewidth*0.5, shrinkA=0, shrinkB=0,color='purple'))    
        j+=1
        # plt.annotate(text='', xy=(spacing[j],omegas123_dqc[m,1]), xytext=(spacing[j],omegas123_dqc[m,1] - omegas123_dqc[m,2]), arrowprops=dict(arrowstyle='->',linewidth=arrow_linewidth*0.5, shrinkA=0, shrinkB=0,color='teal'))    
        j+=2
        plt.vlines(spacing[j],-100,np.max(omegas_gfs_full)*1.1,'w',linewidth=10)
        j += 2
        # print(j)
    plt.xlim(-0.1,len(pathway_omegas))
    plt.ylim(-1000,np.max(omegas_gfs_full)*1.05)#33000)
    plt.xlabel('term number',fontsize=14)
    plt.ylabel(r'Energy $(cm^{-1})$',fontsize=14)
     #%%
    # NRP & RP
    # plt.figure(figsize=[30,8])
    fig,ax = plt.subplots(figsize=[50,10])
    Nsteps = len(pathway_omegas)*13
    spacing = np.linspace(0,len(pathway_omegas),Nsteps)
    xmax = len(pathway_omegas)
    plt.hlines(0,0,len(pathway_omegas),'k')
    plt.title('Pathways for NRP & RP',fontsize=18)
    for i in range(len(omegas_ges)):
        plt.hlines(omegas_ges[i],0,xmax,'k',linewidth=2.5,zorder=2)
        plt.hlines(omegas_ges[i]+omegas_efs[i,i],0,xmax,'k',linewidth=2.5,zorder=2)
        # plt.hlines(omegas_gfs[i],0,10,'gray','--')
    for j in range(len(omegas_ges_full)):
        plt.hlines(omegas_ges_full[j],0,xmax,'gray','--',zorder=-1,linewidth=1)
        plt.hlines(omegas_gfs_full[j],0,xmax,'gray','--',zorder=-1, linewidth=1)
    # idx  0  1  2  3
    # t_i  4  2  1  3
    # NRP pathways
    j=0
    for m in range(len(pathway_omegas)):
        j+=1                        # t1, idx2
        ax.annotate(text='', xy=(spacing[j],pathway_omegas[m,2]), xytext=(spacing[j],0), arrowprops=dict(arrowstyle='->', linewidth=arrow_linewidth, shrinkA=0, shrinkB=0,color='r'))
        j+= 1                      # t2, idx1
        ax.annotate(text='', xy=(spacing[j],pathway_omegas[m,1]), xytext=(spacing[j],0), arrowprops=dict(arrowstyle='->', linewidth=arrow_linewidth, shrinkA=0, shrinkB=0,color='m'))
        j+=1                        # t3, idx3
        if np.product(ETarr[m] == [0,0,0,0]): #ETarr[m] == 0: # no ET
            ax.annotate(text='', xy=(spacing[j],pathway_omegas[m,3] + pathway_omegas[m,2]), xytext=(spacing[j],pathway_omegas[m,2]), arrowprops=dict(arrowstyle='->',linewidth=arrow_linewidth, shrinkA=0, shrinkB=0,color='b')) 
        elif np.product(ETarr[m] == [1,0,1,0]): #ETarr[m] == 1: # w ET
            start_idx = m%nET_eeRel_terms #int(m-len(pathway_omegas/2))
            ax.annotate(text='', xy=(spacing[j],pathway_omegas[m,3] + pathway_ges_evenStart[start_idx,1]  ), xytext=(spacing[j],pathway_ges_evenStart[start_idx,1]), arrowprops=dict(arrowstyle='->',linewidth=arrow_linewidth, shrinkA=0, shrinkB=0,color='b'))
            ax.add_patch(curly_arrow((spacing[j-2], pathway_omegas[m,2]), (spacing[j],pathway_ges_evenStart[start_idx,1]), n=5, arr_size=0.01,linew=1,width=0.05,col='r'))
        elif np.product(ETarr[m] == [0,2,1,0]): #ETarr[m] == 2:
            start_idx = m%nET_feRel_terms
            plt.annotate(text='', xy=(spacing[j],pathway_omegas[m,2] + pathway_omegas[m,3]), xytext=(spacing[j],pathway_omegas[m,2]), arrowprops=dict(arrowstyle='->',linewidth=arrow_linewidth, shrinkA=0, shrinkB=0,color='b'))
            ax.add_patch(curly_arrow((spacing[j+1], pathway_ef_noRel_feRel[start_idx,1]+pathway_omegas[m,2]), (spacing[j], pathway_omegas[m,2] + pathway_omegas[m,3]), n=5, arr_size=0.01,linew=1,width=0.05,col='b')) 
        elif np.product(ETarr[m] == [1,0,0,2]): #ETarr[m] == 2:
            start_idx = m%nET_efRel_terms
            # plt.annotate(text='', xy=(spacing[j],pathway_omegas[m,0] + pathway_omegas[m,1]), xytext=(spacing[j],pathway_omegas[m,1]), arrowprops=dict(arrowstyle='->',linewidth=arrow_linewidth, shrinkA=0, shrinkB=0,color='c'))
            plt.annotate(text='', xy=(spacing[j],pathway_omegas[m,3] + pathway_ges_evenStart_efRel[start_idx,1]), xytext=(spacing[j],pathway_ges_evenStart_efRel[start_idx,1]), arrowprops=dict(arrowstyle='->',linewidth=arrow_linewidth, shrinkA=0, shrinkB=0,color='b'))
            # ax.add_patch(curly_arrow((spacing[j], pathway_ef_noRel_f[start_idx,0]+pathway_omegas[m,1]), (spacing[j], pathway_omegas[m,0] + pathway_omegas[m,1]), n=5, arr_size=0.01,linew=1,width=0.05,col='c')) 
            ax.add_patch(curly_arrow((spacing[j-2], pathway_omegas[m,2]), (spacing[j], pathway_ges_evenStart_efRel[start_idx,1]), n=5, arr_size=0.01,linew=1,width=0.05,col='r')) 
            
        j += 1                      # t4, idx0
        if np.product(ETarr[m] == [0,0,0,0]):#ETarr[m] == 0: # no ET
            ax.annotate(text='', xy=(spacing[j],pathway_omegas[m,0] + pathway_omegas[m,1]), xytext=(spacing[j],pathway_omegas[m,1]), arrowprops=dict(arrowstyle='->',linewidth=arrow_linewidth, shrinkA=0, shrinkB=0,color='c'))
        elif np.product(ETarr[m] == [1,0,1,0]):#ETarr[m] == 1: # w ET
            start_idx = m%nET_eeRel_terms#int(m-len(pathway_omegas/2))
            ax.annotate(text='', xy=(spacing[j],pathway_omegas[m,0] + pathway_ges_evenStart[start_idx,0]), xytext=(spacing[j],pathway_ges_evenStart[start_idx,0]), arrowprops=dict(arrowstyle='->',linewidth=arrow_linewidth, shrinkA=0, shrinkB=0,color='c'))
            ax.add_patch(curly_arrow((spacing[j-2], pathway_omegas[m,1]), (spacing[j],pathway_ges_evenStart[start_idx,0]), n=5, arr_size=0.01,linew=1,width=0.05,col='m'))
        elif np.product(ETarr[m] == [0,2,1,0]): #ETarr[m] == 2:
            start_idx = m%nET_feRel_terms
            # plt.annotate(text='', xy=(spacing[j],pathway_omegas[m,0] + pathway_omegas[m,1]), xytext=(spacing[j],pathway_omegas[m,1]), arrowprops=dict(arrowstyle='->',linewidth=arrow_linewidth, shrinkA=0, shrinkB=0,color='c'))
            plt.annotate(text='', xy=(spacing[j],pathway_omegas[m,0] + pathway_ges_evenStart_feRel[start_idx,0]), xytext=(spacing[j],pathway_ges_evenStart_feRel[start_idx,0]), arrowprops=dict(arrowstyle='->',linewidth=arrow_linewidth, shrinkA=0, shrinkB=0,color='c'))
            # ax.add_patch(curly_arrow((spacing[j], pathway_ef_noRel_f[start_idx,1]+pathway_omegas[m,2]), (spacing[j], pathway_omegas[m,2] + pathway_omegas[m,3]), n=5, arr_size=0.01,linew=1,width=0.05,col='c')) 
            ax.add_patch(curly_arrow((spacing[j-2], pathway_omegas[m,1]), (spacing[j], pathway_ges_evenStart_feRel[start_idx,0]), n=5, arr_size=0.01,linew=1,width=0.05,col='m')) 
        elif np.product(ETarr[m] == [1,0,0,2]): #ETarr[m] == 2:
            start_idx = m%nET_efRel_terms
            # plt.annotate(text='', xy=(spacing[j],pathway_omegas[m,0] + pathway_omegas[m,1]), xytext=(spacing[j],pathway_omegas[m,1]), arrowprops=dict(arrowstyle='->',linewidth=arrow_linewidth, shrinkA=0, shrinkB=0,color='c'))
            plt.annotate(text='', xy=(spacing[j],pathway_omegas[m,0] + pathway_omegas[m,1]), xytext=(spacing[j], pathway_omegas[m,1]), arrowprops=dict(arrowstyle='->',linewidth=arrow_linewidth, shrinkA=0, shrinkB=0,color='c'))
            # ax.add_patch(curly_arrow((spacing[j], pathway_ef_noRel_f[start_idx,0]+pathway_omegas[m,1]), (spacing[j], pathway_omegas[m,0] + pathway_omegas[m,1]), n=5, arr_size=0.01,linew=1,width=0.05,col='c')) 
            # ax.add_patch(curly_arrow((spacing[j+2], pathway_ges_evenStart_feRel[start_idx,0]+pathway_ef_noRel_efRel[start_idx,0]), (spacing[j], pathway_omegas[m,0] + pathway_omegas[m,1]), n=5, arr_size=0.01,linew=1,width=0.05,col='c')) 
            ax.add_patch(curly_arrow((spacing[j], pathway_omegas[m,0] + pathway_omegas[m,1]), (spacing[j+2], pathway_ef_noRel_feRel[start_idx,0]+pathway_omegas[m,1]), n=5, arr_size=0.01,linew=1,width=0.05,col='c')) 

    #%
        j += 3
        # plt.annotate(text='', xy=(spacing[j],omegas123_nrprp[m,0]), xytext=(spacing[j],0), arrowprops=dict(arrowstyle='->', linewidth=arrow_linewidth*0.5, shrinkA=0, shrinkB=0,color='orange'))
        j+=1
        # if ETarr[m] == 0:
        #     plt.annotate(text='', xy=(spacing[j],omegas123_nrprp[m,1] + omegas123_nrprp[m,0]), xytext=(spacing[j],omegas123_nrprp[m,0]), arrowprops=dict(arrowstyle='->', linewidth=arrow_linewidth*.5, shrinkA=0, shrinkB=0,color='purple'))
        # elif ETarr[m] == 1:
        #     plt.annotate(text='', xy=(spacing[j],omegas123_nrprp[m,1] +pathway_ges_evenStart[start_idx,1]), xytext=(spacing[j],pathway_ges_evenStart[start_idx,1]), arrowprops=dict(arrowstyle='->', linewidth=arrow_linewidth*.5, shrinkA=0, shrinkB=0,color='purple'))
        j+=1
        # if ETarr[m] == 0: # no ET
        #     plt.annotate(text='', xy=(spacing[j],omegas123_nrprp[m,2]+omegas123_nrprp[m,1] + omegas123_nrprp[m,0]), xytext=(spacing[j],omegas123_nrprp[m,1] +omegas123_nrprp[m,0]), arrowprops=dict(arrowstyle='->', linewidth=arrow_linewidth*0.5, shrinkA=0, shrinkB=0,color='teal'))
        # elif ETarr[m] == 1: # w ET
        #     plt.annotate(text='', xy=(spacing[j],omegas123_nrprp[m,2]+omegas123_nrprp[m,1] + pathway_ges_evenStart[start_idx,1]), xytext=(spacing[j],omegas123_nrprp[m,1] +pathway_ges_evenStart[start_idx,1]), arrowprops=dict(arrowstyle='->', linewidth=arrow_linewidth*0.5, shrinkA=0, shrinkB=0,color='teal'))
        #     ax.add_patch(curly_arrow((spacing[j-2],omegas123_nrprp[m,0]), (spacing[j-1],pathway_ges_evenStart[start_idx,1]), n=5, arr_size=0.01,linew=1,width=0.05,col='orange'))
    
        j += 2
        plt.vlines(spacing[j],-300,np.max(omegas_gfs_full)*1.05,'w',linewidth=10)
        j += 2
    plt.xlim(-0.1,len(pathway_omegas))
    plt.ylim(-1000,np.max(omegas_gfs_full)*1.1)#33000)
    plt.xlabel('term number',fontsize=14)
    plt.ylabel(r'Energy $(cm^{-1})$',fontsize=14)


        #%%

pathway_plotter(epsilon0, omega0, lam)

