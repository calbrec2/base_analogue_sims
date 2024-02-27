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
fig,ax = plt.subplots(2,1,figsize=[10,8],sharex=True)
ax[0].plot(cAbsSpectrum[:,0],cAbsSpectrum[:,1])
ax[0].axhline(0,color='k',linestyle='--')
ax[0].set_xlim(25500,36000)
ax[1].plot(cCDSpectrum[:,0],cCDSpectrum[:,1])
ax[1].axhline(0,color='k',linestyle='--')
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
nVib = 4
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

#### For 1PE ####
# electronic raising and lowering operators: c, cD
c_1PE = sp.zeros((nEle,nEle))       # nxn where n is the number of electronic states
c_1PE[0,nEle-1] = 1                 # 1PE takes you all the way to the f-state
cD_1PE = c_1PE.T
cDc_1PE = sp.dot(cD_1PE,c_1PE)      # electronic number operator
# number ops (need to be same size as corresponding identity ops)
muOp_1PE = cD_1PE + c_1PE           # proportional to position operator (x = sqrt(hbar/ 2m omega) (c + cD))
#################

# ----------- Vibrational operators ----------- # Vibrational Modes   #***#   6 (for Cy3)
# vibrational raising and lowering operators: b, bD
b = sp.zeros((nVib,nVib)) # need to be mxm where m is the number of vibrational states
for i in range(nVib-1):
    b[i,i+1] = sp.sqrt(i+1)  # vibrational raising and lowering ops
bD = b.T

# number ops (need to be same size and corresponding identity ops)
bDb = sp.dot(bD,b)  # vibrational number operator

# identity ops
Iel = sp.eye(nEle)  # g, e, fg,        ( from selection rules model: e1, f0 )
Ivib = sp.eye(nVib) # 0, 1             ( from selection rules model: e3, f2 )

#################################################

#%% Hamiltonians
# =============================================================================
# Generate Hamiltonian for monomer A (see eq 17 from Kringle et al)
# =============================================================================
def Ham_2PE(epsilon0, omega0, lam):
    # ----------- 2PE Hamiltonian ----------- #
    h1 = (epsilon0/2) * kr(cDc_2PE, Ivib)   # electronic levels
    h4 = omega0 * kr(Iel, bDb)              # vibrational levels
    h6 = omega0 * kr(cDc_2PE, lam * (bD + b) + (lam**2)*Ivib) # coupling between electronic and vibrational
    ham_2PE = h1 + h4 + h6
    # matrix_plotter(hamA_2PE, alpha, alpha, title=r'Hamiltonian of monomer $(cm^{-1} x10^{3})$',size=nEle*nVib,title_fontsize=20,label_fontsize=16,fontsize=22)#14)
    
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
    # matrix_plotter(diag_ham, alpha, alpha, title=r'Diagonalized Hamiltonian of monomer $(cm^{-1} x10^{3})$' ,frac=0.8,size=nEle*nVib,title_fontsize=20,label_fontsize=16,fontsize=22)
    return eigs_2PE, vecs_2PE#, ham_2PE #, diag_ham_2PE
def Ham_1PE(epsilon0, omega0, lam):
    # ----------- 1PE Hamiltonian ----------- #
    h1 = epsilon0 * kr(cDc_1PE, Ivib)       # electronic levels
    h4 = omega0 * kr(Iel, bDb)              # vibrational levels
    h6 = omega0 * kr(cDc_1PE, lam * (bD + b) + (lam**2)*Ivib) # coupling between electronic and vibrational
    ham_1PE = h1 + h4 + h6
    # matrix_plotter(hamA_1PE, alpha, alpha, title=r'Hamiltonian of monomer $(cm^{-1} x10^{3})$',size=nEle*nVib,title_fontsize=20,label_fontsize=16,fontsize=22)#14)
    
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
    # matrix_plotter(diag_ham, alpha, alpha, title=r'Diagonalized Hamiltonian of monomer $(cm^{-1} x10^{3})$' ,frac=0.8,size=nEle*nVib,title_fontsize=20,label_fontsize=16,fontsize=22)
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

#%%
# =============================================================================
# Optimize calculated Abs to experimental CD
# =============================================================================
def monomerCD_chisq_opt(params):
    epsilon0, omega0, lam = params
    eigs_1PE, vecs_1PE = Ham_1PE(epsilon0, omega0, lam)
    AbsData, simAbs = calc_Abs(vecs_1PE, eigs_1PE, sigma=20, gamma=200, plot_mode=0)
    
    mask_lower_transition = (simAbs[:,0]>27500) * (simAbs[:,0] < 32000) # put zeros outsize of this range
    chisq = np.sum( (simAbs[:,1] - cCDSpectrum[:,1])**2 * mask_lower_transition)
    return chisq

lam = 2.4
epsilon0 = 29500 
omega0=120
x0 = np.array([epsilon0, omega0, lam])
lam_bounds = [1,2.8] #[0,4]
epsilon0_bounds = [29000, 30000]
omega0_bounds = [80, 150]
bounds = np.vstack([epsilon0_bounds, omega0_bounds,lam_bounds])
if __name__ == '__main__':
    res = opt.differential_evolution(func=monomerCD_chisq_opt, 
                                      bounds=bounds,
                                      x0=x0,
                                      disp=True,
                                      workers=1,
                                      maxiter=1000,
                                      polish=True)#,
                                      # atol=1e-8, #1e-6, 1e-10,
                                      # tol = 1e-8, #1e-6, 10,
                                      # mutation=(0,1.9),
                                      # popsize=30,
                                      # updating='immediate',
                                      # strategy = 'best1exp') 
    epsilon0, omega0, lam = res.x
print('epsilon0 = '+str(epsilon0))
print('omega0 = '+str(omega0))
print('lam = '+str(lam))
eigs_1PE, vecs_1PE = Ham_1PE(epsilon0, omega0, lam)
AbsData, simAbs = calc_Abs(vecs_1PE, eigs_1PE, sigma=20, gamma=200, plot_mode=1)

#%% plot2Dspectra function from simple2Dcalc_fromRbcode_CSA_v14a

# plotting function that creates aspect ratio 1 plots with the correct axes and labels, etc.
def plot2Dspectra(ax1, ax2, data, n_cont,ax_lim, title = '', domain = 'time',save_mode = 0,file_name=' ',scan_folder = ' '):
    #%
    axes_fontsize = 14
    cmap = 'jet' 
    # print('scan_folder = '+scan_folder)
    scan_params = scan_folder[len('20230101-120000-'):len(scan_folder)-len('_2DFPGA_FFT')]
    # stages = scan_params[len(scan_params)-2:]
    scan_type = scan_params[:len(scan_params)-3]
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
    if timing_mode == 't32 = 0':
        if ax1.shape == (len(ax1),len(ax1)):
            plt.plot(ax1[0,:],ax2[:,0],linestyle='--',color='w')
        else:
            plt.plot(ax1,ax2,linestyle='--',color='w')    
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
    if timing_mode == 't32 = 0':
        if ax1.shape == (len(ax1),len(ax1)):
            plt.plot(ax1[0,:],ax2[:,0],linestyle='--',color='w')
        else:
            plt.plot(ax1,ax2,linestyle='--',color='w')
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
    if timing_mode == 't32 = 0':
        if ax1.shape == (len(ax1),len(ax1)):
            plt.plot(ax1[0,:],ax2[:,0],linestyle='--',color='w')
        else:
            plt.plot(ax1,ax2,linestyle='--',color='w')    
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
def plot_comparer(ax1,ax2, data, sim, phase_cond, compare_mode = 'real', domain='freq',figsize=(16,4), ax_lim=[28,30],n_cont=15, save_mode = 0, file_name = '',scan_folder=scan_folder,weight_func_mode=1):
    axes_fontsize = 14
    title=phase_cond + ' Experiment vs Simulation'
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
        
    scan_params = scan_folder[len('20230101-120000-'):len(scan_folder)-len('_2DFPGA_FFT')]
    # stages = scan_params[len(scan_params)-2:]
    scan_type = scan_params[:len(scan_params)-3]
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
        plt.subplot(131)
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
            plt.plot(ax1[0,:],ax2[:,0],linestyle='--',color='w')
        else:
            plt.plot(ax1,ax2,linestyle='--',color='w')    
        fig.colorbar(cf)
        # plt.contour(self.axes[0], self.axes[1], np.real(self.dat), levels = cf.levels, cmap = 'binary')#colors='black')
        # plt.contour(ax1, ax2, np.real(data), levels = cf.levels[cf.levels >= 0], colors='black',linestyle='-')
        # plt.contour(ax1, ax2, np.real(data), levels = cf.levels[cf.levels < 0], colors='white',linestyle='.')
        plt.contour(ax1, ax2, np.real(data)/np.max(np.max(np.real(data))), levels = cf.levels[cf.levels >= 0], colors='black')
        plt.contour(ax1, ax2, np.real(data)/np.max(np.max(np.real(data))), levels = cf.levels[cf.levels < 0], colors='white')
        plt.tick_params(axis='both', which='major', labelsize=15)
        plt.xlim(x_lim)
        plt.ylim(y_lim)
        
        plt.subplot(132)
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
            plt.plot(ax1[0,:],ax2[:,0],linestyle='--',color='w')
        else:
            plt.plot(ax1,ax2,linestyle='--',color='w')    
        fig.colorbar(cf)
        # plt.contour(self.axes[0], self.axes[1], np.real(self.dat), levels = cf.levels, cmap = 'binary')#colors='black')
        # plt.contour(ax1, ax2, np.real(data), levels = cf.levels[cf.levels >= 0], colors='black',linestyle='-')
        # plt.contour(ax1, ax2, np.real(data), levels = cf.levels[cf.levels < 0], colors='white',linestyle='.')
        plt.contour(ax1, ax2, np.real(sim)/np.max(np.max(np.real(sim))), levels = cf.levels[cf.levels >= 0], colors='black')
        plt.contour(ax1, ax2, np.real(sim)/np.max(np.max(np.real(sim))), levels = cf.levels[cf.levels < 0], colors='white')
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
            plt.plot(ax1[0,:],ax2[:,0],linestyle='--',color='w')
        else:
            plt.plot(ax1,ax2,linestyle='--',color='w')    
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
            plt.plot(ax1[0,:],ax2[:,0],linestyle='--',color='w')
        else:
            plt.plot(ax1,ax2,linestyle='--',color='w')    
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
            plt.plot(ax1[0,:],ax2[:,0],linestyle='--',color='w')
        else:
            plt.plot(ax1,ax2,linestyle='--',color='w')    
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
            plt.plot(ax1[0,:],ax2[:,0],linestyle='--',color='w')
        else:
            plt.plot(ax1,ax2,linestyle='--',color='w')    
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
    tau_ax_lim = [0, 104]
    if dqc_mode == 0:
        smatIntPx2 = smatIntPx2 / np.max(np.max(smatIntPx2))
        plot2Dspectra(t21ax_rt, t43ax_rt, smatIntPx2, n_cont, ax_lim=tau_ax_lim, title=r'Experimental NRP($\tau$) with ($\tau_{32}$ = 0) w/re-timing & re-phasing')
        dmatIntPx2 = dmatIntPx2 / np.max(np.max(dmatIntPx2))
        plot2Dspectra(t21ax_rt, t43ax_rt, dmatIntPx2, n_cont, ax_lim=tau_ax_lim, title=r'Experimental RP($\tau$) with ($\tau_{32}$ = 0) w/re-timing & re-phasing')
    else:
        smatIntPx2 = smatIntPx2 / np.max(np.max(smatIntPx2))
        plot2Dspectra(t21ax_rt, t43ax_rt, smatIntPx2, n_cont, ax_lim=tau_ax_lim, title=r'Experimental DQC($\tau$) with ($\tau_{32}$ = 0) w/re-timing & re-phasing')
    
    
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
        plot2Dspectra(t21ax, t43ax, smatW, n_cont, ax_lim=tau_ax_lim, title=r'Experimental NRP($\tau$) with ($\tau_{32}$ = 0)', scan_folder = scan_folder)
        dmatW = dmatW / np.max(np.max(dmatW))
        plot2Dspectra(t21ax, t43ax, dmatW, n_cont, ax_lim=tau_ax_lim, title=r'Experimental RP($\tau$) with ($\tau_{32}$ = 0)', scan_folder = scan_folder)
    else:
        smatW = smatW / np.max(np.max(smatW))
        plot2Dspectra(t21ax, t43ax, smatW, n_cont, ax_lim=tau_ax_lim, title=r'Experimental DQC($\tau$) with ($\tau_{32}$ = 0)', scan_folder = scan_folder)
    
    
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
        plot2Dspectra(xaxis, yaxis, sumFunc_RT, n_cont, ax_lim=xbounds, title=r'Experimental NRP($\omega$) with ($\tau_{32}$ = 0)', domain='freq',scan_folder = scan_folder_nrprp)
        plot2Dspectra(xaxis, yaxis, difFunc_RT, n_cont, ax_lim=xbounds, title=r'Experimental RP($\omega$) with ($\tau_{32}$ = 0)',domain='freq',  scan_folder = scan_folder_nrprp)
    else:
        if date_folder == '20221122':
            sumFunc_RT = np.rot90(sumFunc_RT,1)
        plot2Dspectra(xaxis, yaxis, sumFunc_RT, n_cont, ax_lim=xbounds, title=r'Experimental DQC($\omega$) with ($\tau_{32}$ = 0)',domain='freq', scan_folder = scan_folder_dqc)

    return xaxis, yaxis, sumFunc_RT, difFunc_RT, xbounds, dqc_mode, t21ax_rt, t43ax_rt, smatIntPx2, dmatIntPx2, Tsn, t43ax, t21ax, dmatW, smatW


global xaxis, yaxis, DQC_exp, NRP_exp, RP_exp , NRP_tau_exp, RP_tau_exp, DQC_tau_exp, t43ax, t21ax, dmatW, smatW
global timing_mode, FPGA_mode, sample_name, FT2D_mode
FT2D_mode = 1

#%% LOAD 2PE-2DFS data for particular sample (unfold section to see all options)
# =============================================================================
# # sample_name = 'MNS_4um' # all samples before ~20231030
# =============================================================================

FPGA_mode = 0
sample_name = 'MNS_4uM'

date_folder = '20221202'
scan_folder_nrprp = '20221202-135005_NRP_RP_xz' # first set of data optimized 
scan_folder_dqc = '20221202-142926_DQC_xz'
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



# FPGA_mode = 1
# sample_name = 'MNS_4uM'
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
# # scan_folder_nrprp = '20230801-144625-NRP_RP_yz_2DFPGA'
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

#%%
# =============================================================================
# 
# =============================================================================
scan_folder = scan_folder_nrprp
scan_params = scan_folder[len('20230101-120000-'):len(scan_folder)-len('_2DFPGA_FFT')]
stages = scan_params[len(scan_params)-2:]
scan_type = scan_params[:len(scan_params)-3]
# =============================================================================
timing_mode = 't32 = 0'
# timing_mode ='t21 = 0'
# timing_mode ='t43 = 0'
# =============================================================================

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

# =============================================================================
#  Calculate all possible energy differences given the eigenenergies 
#  => these give the omega21, omega_32, omega_43 values
# =============================================================================
def eigs2omegas(eigs_2PE, selection_rules = 1, plot_mode = 0):
    omegas= np.real(np.subtract.outer(eigs_2PE,eigs_2PE))
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
    # matrix_plotter(omegas_eeps, alpha_es, alpha_es,title=r"Energies for $\Sigma_{i,j} \omega_{e_ie'_j}$",frac=0.99,label_fontsize=18)
    # matrix_plotter(omegas_gfs, alpha_gs, alpha_fs,title=r'Energies for $\Sigma_{i} \omega_{gf_i}$',frac=0.99,label_fontsize=18)
    
    # =============================================================================
    # Impose selection rules
    # =============================================================================
    if selection_rules == 1:
        omegas_ges = omegas_ges[1::2,::2] # only select the omegas_ges (g -> odd e's)
        omegas_ges = omegas_ges[:,0].reshape(omegas_ges[:,0].shape[0],1) # we actually dont want any g other than g0
        if plot_mode == 1:
            matrix_plotter(omegas_ges, [alpha_gs[0]], np.array(alpha_es[1::2]),title=r'Energies for $\Sigma_i \omega_{ge_i}$',frac=0.99,label_fontsize=18)
        omegas_eeps = omegas_eeps[1::2,1::2] # (odd -> even) ... is this right?
        if plot_mode == 1:
            matrix_plotter(omegas_eeps, alpha_es[1::2],alpha_es[1::2],title=r'Energies for $\Sigma_{i,j} \omega_{e_ie_j}$',frac=0.99,label_fontsize=18)
        # how to take care of selection rules for eep? should it only be the odd e's?
        omegas_gfs = omegas_gfs[1::2,::2] # (g -> even f's) ... is this right?
        omegas_gfs = omegas_gfs[:,0].reshape(omegas_ges[:,0].shape[0],1) # we actually dont want any g other than g0
        if plot_mode == 1:
            matrix_plotter(omegas_gfs, [alpha_gs[0]], alpha_fs[::2],title=r'Energies for $\Sigma_{i,j} \omega_{gf_j}$',frac=0.99,label_fontsize=18)
        omegas_efs = omegas_efs[::2,1::2] # select omegas_efs that we want (odd e's -> even f's)
        if plot_mode == 1:
            matrix_plotter(omegas_efs, alpha_es[1::2], alpha_fs[::2],title=r'Energies for $\Sigma_{i,j} \omega_{e_if_j}$',frac=0.99,label_fontsize=18)
    
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
    
    # omegas for DQC
    omegas123 = []
    for i in range(len(omegas_ges)):
        for j in range(len(omegas_eeps)):
            for k in range(len(omegas_efs)):
                omegas123.append([omegas_ges[i,0], omegas_gfs[k,0], omegas_efs[k,j]])
    #                                omega1              omega2          omega3
    omegas123_dqc = np.array(omegas123)
    
    # omegas for NRP & RP
    omegas123 = []
    for i in range(len(omegas_ges)):
        for j in range(len(omegas_eeps)):
            for k in range(len(omegas_efs)):
                omegas123.append([omegas_ges[i,0], omegas_eeps[j,i], omegas_efs[k,j]])
    #                                omega1              omega2          omega3
    omegas123_nrprp = np.array(omegas123)
    
    return omegas123_nrprp, omegas123_dqc, omegas_ges, omegas_efs, omegas_eeps, omegas_gfs

omegas123_nrprp, omegas123_dqc,omegas_ges, omegas_efs, omegas_eeps, omegas_gfs = eigs2omegas(eigs_2PE)

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
def gen_mu_prods(mu1, mu2, omegas_ges, omegas_eeps, omegas_efs):
    # use test values for dipoles for now
    mu_ge = mu_ef = mu1/2
    mu_gep = mu_epfp = mu2/2
    mu_epf = (mu_ge + mu_ef) - mu_gep
    mu_efp = (mu_gep + mu_epfp) - mu_ge
    
    # need to generalize this part?
    mus_ges = np.array([mu_ge, mu_gep]) # i=0, i=1
    #           j,k =     0,0    0,1
    mus_efs = np.array([[mu_ef, mu_epf],        
                        [mu_efp,mu_epfp]])
    #           j,k =     1,0    1,1
    
    mu_prods = []
    for i in range(len(omegas_ges)):
        for j in range(len(omegas_eeps)):
            for k in range(len(omegas_efs)):
                mu_prods.append(mus_efs[k,j] * mus_ges[j] * mus_ges[i] * mus_efs[k,i])
    mu_prods = np.array(mu_prods) 
    return mu_prods


#%%

def gauss(x, lam1, sig1, amp1):
    return amp1 * np.exp(-(x-lam1)**2 / (2 *sig1**2))

# =============================================================================
# Calculate 2D FFT
# =============================================================================
def FFT_2d(data, t21ax_rt, time_ax, monoC_lam, scan_type): # do the zeropadding more explicitly than above...
    data_0 = np.zeros([len(t21ax_rt), len(t21ax_rt)], dtype='complex')
    data_0[:len(data),:len(data)] = data
    
    NFFT = len(data_0)
    dataFT = np.fft.fftshift(np.fft.fft2(data_0,[NFFT,NFFT]))
    
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
                print('t43=0 & NRP_RP => Fmono = '+str(Fmono))
            else: # => scan_type = 'DQC'
                Fmono = 2 * (10**7/monoC_lam)
                print('t43=0 & DQC => Fmono = '+str(Fmono))
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

def sim2Dspec3(t21, laser_lam, laser_fwhm, mu1, mu2, Gam, sigI, monoC_lam, epsilon0, omega0, lam): # optimize these if using CD results doesn't work
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
    eigs_2PE, vecs_2PE = Ham_2PE(epsilon0, omega0, lam)  
    # do we want to optimize these values or use the ones from the abs/cd opt?
    # for now just use CD output
    
    # use eigenvalues to generate omegas for dqc, nrprp and each of the transitions
    omegas123_nrprp, omegas123_dqc,omegas_ges, omegas_efs, omegas_eeps, omegas_gfs = eigs2omegas(eigs_2PE)
    
    # generate mu products for each pathway (can this be further generalized?)
    mu_prods = gen_mu_prods(mu1, mu2, omegas_ges, omegas_eeps, omegas_efs)
    
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
    orient_avg_arr = np.ones(8) * (1/5) 
    # 1/5 comes from the orientational average of the angle between the molecule EDTM   and the laser polarization (horizontal)

    cm_DQC = np.zeros([time_size,time_size],dtype='complex')
    cm_NRP = np.zeros([time_size,time_size],dtype='complex')
    cm_RP =  np.zeros([time_size,time_size],dtype='complex')
    nterms = len(mu_prods)
    for i in range(nterms):
        # start with DQC energies
        omega1, omega2, omega3 = omegas123_dqc[i,:]

        # shift omegas based on overlap with laser spectrum
        # use inhomogenous linewidth as width of molecular absorption peak to calculate shifts (variable parameter)
        omega1 = (laser_omega * sigI**2 + np.array(omega1) * laser_sig_omega**2) / (sigI**2 + laser_sig_omega**2)
        omega2 = ((2*laser_omega) * sigI**2 + np.array(omega2) * (laser_sig_omega/2)**2) / (sigI**2 + (laser_sig_omega/2)**2)
        # need 2*laser omega for omega2 in the DQC calculation because omega2 is during t32 which has an |g><f| coherence = 2x energy of laser
        omega3 = (laser_omega * sigI**2 + np.array(omega3) * laser_sig_omega**2) / (sigI**2 + laser_sig_omega**2)
   
        # alphas are: what is the amplitude of the overlap between molecule abs and laser at the newly shifted energies (directly above) 
        alpha1 = gauss(np.array(omega1), laser_omega, laser_sig_omega,1)
        # alpha2 = gauss(np.array(omega2), laser_omega, laser_sig_omega,1) # things are funky about omega2... sort this out
        alpha3 = gauss(np.array(omega3), laser_omega, laser_sig_omega,1)
        # alpha1 = alpha1/np.max(alpha1)
        # alpha2 = alpha2/np.max(alpha2) # things are funky about omega2... sort this out
        # alpha3 = alpha3/np.max(alpha3)
        overlap_alpha = alpha1 * alpha3 
        # product of the alphas will scale this peak intensity
 
        # subtract off monochromator reference frequency (because we are downsampling as explained in Tekavec 2006 & 2007)
        omega1 = (np.array(omega1) - monoC_omega)
        omega2 = (np.array(omega2) - (2 * monoC_omega) ) # factor of 2 came out of calculations
        omega3 = (np.array(omega3) - monoC_omega)

        # calculate response function for DQC
        cm_DQC += overlap_alpha * orient_avg_arr[i] * mu_prods[i] * np.exp(1j*nubar2omega*(omega3*t43 + omega2*t32 + omega1*t21)) \
                                                            * np.exp(-Gam*nubar2omega*(t43 + t32 + t21)) * np.exp(-(1/2)*(sigI*nubar2omega)**2*(t21 + t32 + t43)**2)
        
        # now for NRP & RP energies
        test, omega2, test = omegas123_nrprp[i,:] # test for 1 and 3 because same as dqc so we don't have to re-do the above calculations

        # how to deal with alpha2?
        ##### omega2 doesn't get monoC_omega subtracted off for NRP & RP... comes out of calculations

        # calculate response functions for NRP & RP
        cm_NRP += overlap_alpha * orient_avg_arr[i] * mu_prods[i] * np.exp(1j*nubar2omega*(omega3*t43 + omega2*t32 + omega1*t21)) \
                                                            * np.exp(-Gam*nubar2omega*(t43 + t32 + t21)) * np.exp(-(1/2)*(sigI*nubar2omega)**2*(t21 + t32 + t43)**2)

        
        cm_RP += overlap_alpha * orient_avg_arr[i] * mu_prods[i] * np.exp(1j*nubar2omega*(omega3*(-1*t43) + omega2*t32 + omega1*t21)) \
                                                            * np.exp(-Gam*nubar2omega*(t43 + t32 + t21)) * np.exp(-(1/2)*(sigI*nubar2omega)**2*(t21 + t32 + (-1*t43))**2)

    ### this commented block is for testing frequency domain calculations
    # if timing_mode == 't32 = 0':
    #     time_ax = t21
    #     FT_dqc_temp, ax1, ax2 = FFT_2d(cm_DQC, t21ax_rt, time_ax, monoC_lam)
    #     # FT_nrp_temp, ax1, ax2 = FFT_2d(cm_NRP, t21ax_rt, time_ax, monoC_lam)
    #     # FT_rp_temp,  ax1, ax2 = FFT_2d(cm_RP,  t21ax_rt, time_ax, monoC_lam)
    #     # FT_rp_temp = np.flipud(FT_rp)
    #     ax_lim = [13.6, 15.75]
    #     plot2Dspectra(ax1, ax2, FT_dqc_temp, n_cont=15, ax_lim=ax_lim, title=r'DQC($\omega$) with '+timing_mode_str,domain='freq')#'($\tau_{32}$ = 0)',domain = 'freq')
    #     # plot2Dspectra(ax1, ax2, FT_nrp_temp, n_cont, ax_lim=ax_lim, title=r'NRP($\omega$) with '+timing_mode_str,domain='freq')#'($\tau_{32}$ = 0)',domain = 'freq')
    #     # plot2Dspectra(ax1, ax2, FT_rp_temp, n_cont, ax_lim=ax_lim, title=r'RP($\omega$) with '+timing_mode_str,domain='freq')#'($\tau_{32}$ = 0)',domain = 'freq')
    #     # plt.show()

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
    # plot2Dspectra(t1_out, t2_out, cm_DQC, n_cont,ax_lim=ax_lim, title=r'DQC($\tau$) with '+timing_mode_str,domain='time')#'($\tau_{32}$ = 0)',domain = 'time')
    # plot2Dspectra(t1_out, t2_out, cm_NRP, n_cont,ax_lim=ax_lim, title=r'NRP($\tau$) with '+timing_mode_str,domain='time')#'($\tau_{32}$ = 0)',domain = 'time')
    # plot2Dspectra(t1_out, t2_out, cm_RP, n_cont,ax_lim=ax_lim, title=r'RP($\tau$) with '+timing_mode_str,domain='time')#'($\tau_{32}$ = 0)',domain = 'time')
    # colormap = 'jet'

    # create a window to apply to time domain before taking FFT (adjust window time and steepness depending on delay space in experiment)
    xx = t21
    yy = t43
    # w = delayedGaussian(np.sqrt(xx**2 + yy**2),80, 10); #70e-15,10e-15); 
    w = delayedGaussian(np.sqrt(xx**2 + yy**2),100, 10); #70e-15,10e-15); 
    #                                         onset, steepness
    # plt.contourf(xx, yy, w, cmap=colormap) # test window
    # plt.axis('scaled')
    # plt.colorbar()
    # plt.show
    
    cm_DQC = cm_DQC * w
    cm_NRP = cm_NRP * w
    cm_RP = cm_RP * w
    
    ### this commented block is for testing windowed time domain calculations
    # ax_lim = [np.min(t21),np.max(t21)]
    # ax_lim = [0, 116]
    # plot2Dspectra(t21, t43, cm_DQC, n_cont, ax_lim=ax_lim,title=r'DQC($\tau$) with ($\tau_{32}$ = 0)',domain = 'time')
    # plot2Dspectra(t21, t43, cm_NRP, n_cont, ax_lim=ax_lim,title=r'NRP($\tau$) with ($\tau_{32}$ = 0)',domain = 'time')
    # plot2Dspectra(t21, t43, cm_RP, n_cont, ax_lim=ax_lim,title=r'RP($\tau$) with ($\tau_{32}$ = 0)',domain = 'time')
    
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


#%% Define chisquared function
# do we need to add a window function? (see example in simple2Dcalc_fromRbcode_CSA_v14a)
def chisq_calc(params):
    t21 = np.linspace(0,tmax,num=Ntimesteps) # generalize this so that when data coming in is over a different range this doesn't cause problems...
    laser_lam, laser_fwhm, mu1, mu2, Gam, sigI, monoC_lam = params
    
    t1_out, t2_out, cm_DQC, cm_NRP, cm_RP, ax1_dqc, ax2_dqc, ax1_nrprp, ax2_nrprp, FT_dqc, FT_nrp, FT_rp = sim2Dspec3(t21, laser_lam, laser_fwhm, mu1, mu2, Gam, sigI, monoC_lam)
    
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

    chisq_tot = chisq_nrp + chisq_dqc + chisq_rp
    
    return chisq_tot



#%%
global timing_mode, Ntimesteps, FT2D_mode, tmax
if stages == 'xz':
    timing_mode ='t32 = 0'
elif stages == 'yz':
    timing_mode ='t21 = 0'
elif stages == 'xy':
    timing_mode ='t43 = 0'
# manually set timing mode instead of from data
timing_mode ='t32 = 0'
# timing_mode ='t21 = 0'
# timing_mode ='t43 = 0'
print(timing_mode)

tmax = np.max(t21ax)

FT_2D_mode = 1 # If =1, always take 2D FT, if = 0, don't FT along t32, so 1DFT for xy and yz experiments and 2DFT for xz experiment
Ntimesteps = int(np.max(np.abs(t21ax))/(t21ax_rt[1] - t21ax_rt[0]))
# t21 = np.linspace(0,116.0805,num=Ntimesteps) # simulate at the retimed timesteps
t21 = np.linspace(0,tmax,num=Ntimesteps) # simulate at the retimed timesteps


#%%

laser_lam = 680 #675
laser_fwhm = 30
mu1 =3.5
mu2 =3 
Gam =85 #85
sigI =95 #65 
monoC_lam = 700#701.9994
epsilon0 = 27900 #29023
omega0 = 70 # 149.8
lam =2.6 #2.677
# pack parameters into param array
params = laser_lam, laser_fwhm, mu1, mu2, Gam, sigI, monoC_lam 

t1_out, t2_out, cm_DQC, cm_NRP, cm_RP, ax1_dqc, ax2_dqc, ax1_nrprp, ax2_nrprp, FT_dqc, FT_nrp, FT_rp = sim2Dspec3(t21, laser_lam, laser_fwhm, mu1, mu2, Gam, sigI, monoC_lam, epsilon0, omega0, lam)
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
plot2Dspectra(t1_out, t2_out, cm_DQC, n_cont,ax_lim=[min(t1_out[0,:]), max(t1_out[0,:])], title=r'DQC($\tau$) with '+timing_mode_str, domain='time',save_mode = save_mode, file_name = save_name,scan_folder=scan_folder_dqc)
save_name = 'sim_' + scan_folder_nrprp + '_tauNRP'
plot2Dspectra(t1_out, t2_out, cm_NRP, n_cont,ax_lim=[min(t1_out[0,:]), max(t1_out[0,:])], title=r'NRP($\tau$) with '+timing_mode_str, domain='time',save_mode = save_mode, file_name = save_name,scan_folder = scan_folder_nrprp)
save_name = 'sim_' + scan_folder_nrprp + '_tauRP'
plot2Dspectra(t1_out, t2_out, cm_RP, n_cont,ax_lim=[min(t1_out[0,:]), max(t1_out[0,:])], title=r'RP($\tau$) with '+timing_mode_str, domain='time',save_mode = save_mode, file_name = save_name,scan_folder=scan_folder_nrprp)

FT_dqc = FT_dqc/ np.max(np.max(FT_dqc))
FT_nrp = FT_nrp/ np.max(np.max(FT_nrp))
FT_rp = FT_rp /  np.max(np.max(FT_rp))

save_mode = save_mode
save_name = 'sim_' + scan_folder_dqc+'_FTdqc'
plot2Dspectra(ax1_dqc, ax2_dqc, FT_dqc, n_cont,ax_lim, title=r'DQC($\omega$) with '+timing_mode_str, domain='freq',save_mode = save_mode, file_name = save_name,scan_folder=scan_folder_dqc)
save_name = 'sim_' + scan_folder_nrprp+'_FTnrp'
plot2Dspectra(ax1_nrprp, ax2_nrprp, FT_nrp, n_cont,ax_lim, title=r'NRP($\omega$) with '+timing_mode_str, domain='freq',save_mode = save_mode, file_name = save_name,scan_folder=scan_folder_nrprp)
save_name = 'sim_' + scan_folder_nrprp+'_FTrp'
plot2Dspectra(ax1_nrprp, ax2_nrprp, FT_rp, n_cont,ax_lim, title=r'RP($\omega$) with '+timing_mode_str, domain='freq',save_mode = save_mode, file_name = save_name,scan_folder=scan_folder_nrprp)

save_mode = 0#save_mode
# save_name = 'sim_' + scan_folder_dqc +'_FTdqcReComp'
plot_comparer(ax1_dqc, ax2_dqc, DQC_exp, FT_dqc, 'DQC',figsize=(16,4),ax_lim = ax_lim ,save_mode = save_mode, file_name = save_name, scan_folder = scan_folder_dqc) #,weight_func_mode=weight_func_mode)
save_name = 'sim_' + scan_folder_nrprp +'_FTnrpReComp'
plot_comparer(ax1_nrprp,ax2_nrprp, NRP_exp, FT_nrp, 'NRP',figsize=(16,4),ax_lim = ax_lim,save_mode = save_mode, file_name = save_name, scan_folder = scan_folder_nrprp)#,weight_func_mode=weight_func_mode)
save_name = 'sim_' + scan_folder_nrprp +'_FTrpReComp'
plot_comparer(ax1_nrprp,ax2_nrprp, RP_exp, FT_rp, 'RP',figsize=(16,4),ax_lim = ax_lim,save_mode = save_mode, file_name = save_name,scan_folder = scan_folder_nrprp)#,weight_func_mode=weight_func_mode)

#%% Optimize 2D specs

t21 = np.linspace(0,116.0805,num=Ntimesteps)
# MNS bounds params
laser_lam_bounds = [674.5, 675.5] 
laser_fwhm_bounds = [29,30]
mu1_bounds = [3,4]
mu2_bounds = [3,4]
Gam_bounds = [30, 1e2] 
sigI_bounds = [30, 1e2] 
monoC_lam_bounds = [699.5, 700.5]
epsilon0_bounds = [27000, 30000]



