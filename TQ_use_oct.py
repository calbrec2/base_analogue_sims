# !/bin/bash
# SBATCH --partition=long        ### Partition (like a queue in PBS)
# SBATCH --job-name=Abs_Spec_Fit      ### Job Name
# SBATCH --output=Abs.out         ### File in which to store job output
# SBATCH --error=Abs.err          ### File in which to store job error messages
# SBATCH --time=0-00:05:00       ### Wall clock time limit in Days-HH:MM:SS
# SBATCH --nodes=1               ### Number of nodes needed for the job
# SBATCH --ntasks-per-node=1     ### Number of tasks to be launched per Node

# You can ignore what's above this, relevant for talapas only. 
################################################################
"""
<><> SPECTRA FITTING <><>

Created on Thu Jul  5 09:37:29 2018

Fits 1D spectra for the Cy-3 Dimer.

All relevant data files for these experiments are stored in a
subdirectory under this one called "Data/" and are labeled 
in the form 

"Data/" + construct + "_Abs_visible_" + str(temp) + "C.txt"

@author: Dylan J Heussman


Notes:
    Requires loading of experimental spectra
    Requires atomistic_tq to calculate J and dipole moments
"""
############# IMPORT STATEMENTS ##############

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

import atomistic_tq_jan23 as tc 
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

###############################################

########### VARIABLE DEFINITIONS ##############
Pi = np.pi

D = 3.33e-30  #C*m per Debye
#D = 2.99e-29 #Debeye per C*m
mumonomer = 12.8 * D  #EDTM for Cy-3
gamma = 93 # In this version of the code 2gamma=FWHM*)
low = 16700
high = 22222 # Data ranges
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

################## OPERATORS ###################
c = sp.zeros((2,2))
c[0,1] = 1
cD = c.T    # electronic raising and lowering operators

muOp = cD + c

# Vibrational Modes                                #***#   6
nVib = 6

b = sp.zeros((nVib,nVib))
for i in range(nVib-1):
    b[i,i+1] = sp.sqrt(i+1)  # vibrational raising and lowering ops
bD = b.T

Iel = sp.eye(2)
Ivib = sp.eye(nVib)  # identity ops
cDc = sp.dot(cD,c)
bDb = sp.dot(bD,b)

#################################################

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
        simval = 0
        for stick in stickdata:
            simval += stick[1]/norm * \
                PseudoVoigtDistribution(point[0], gamma, sigma, stick[0])
                #CauchyDist(point[0],stick[0],gamma)
                #NormalDist(point[0],stick[0], sigma)
        output.append([point[0], simval])
    return np.array(output)

###################################################

################## CALCULATION ####################

def Coupling(mu1, mu2, R12):
    '''calculate coupling strength for dimer system
    
    vec, vec, sep -> J (joules)'''
    
    return (1/(4*Pi*permFree*(la.norm(R12))**3))*(np.dot(mu1, mu2)- \
                                            3*(np.dot(R12,mu1)*(np.dot(mu2,R12)/(la.norm(R12))**2))) 

def TargetAbsCDSpectra(phiN_thetaN_rollN_shiftN_shearN_R12ang_sigma_chiAbs_chiCD):
    '''determines difference between a calculated spectra 
    and a measured one.
    
    6 element array-like -> num (chi^2 for spectra)'''
    
    # first we unpack the input parameter array
    
    phiN = phiN_thetaN_rollN_shiftN_shearN_R12ang_sigma_chiAbs_chiCD[0]
            # Twist
    thetaN = phiN_thetaN_rollN_shiftN_shearN_R12ang_sigma_chiAbs_chiCD[1]
            # Tilt
    rollN = phiN_thetaN_rollN_shiftN_shearN_R12ang_sigma_chiAbs_chiCD[2]
            #Shift
    shiftN = phiN_thetaN_rollN_shiftN_shearN_R12ang_sigma_chiAbs_chiCD[3]
            # Roll 
    shearN = phiN_thetaN_rollN_shiftN_shearN_R12ang_sigma_chiAbs_chiCD[4]
            #Shear
    R12ang = phiN_thetaN_rollN_shiftN_shearN_R12ang_sigma_chiAbs_chiCD[5]
            # Separation
    sigma = phiN_thetaN_rollN_shiftN_shearN_R12ang_sigma_chiAbs_chiCD[6]
            # broadening
    chiAbs = phiN_thetaN_rollN_shiftN_shearN_R12ang_sigma_chiAbs_chiCD[7]
    chiCD = phiN_thetaN_rollN_shiftN_shearN_R12ang_sigma_chiAbs_chiCD[8]
            # scale factors
    
#def TargetAbsCDSpectra(phiN_thetaN_R12ang_sigma_chiAbs_chiCD):
#    '''determines difference between a calculated spectra 
#    and a measured one.
#    
#    6 element array-like -> num (chi^2 for spectra)'''
#    
#    # first we unpack the input parameter array
#    
#    phiN = phiN_thetaN_R12ang_sigma_chiAbs_chiCD[0]
#            # Twist
#    thetaN = phiN_thetaN_R12ang_sigma_chiAbs_chiCD[1]
#            # Tilt
#    R12ang = phiN_thetaN_R12ang_sigma_chiAbs_chiCD[2]
#            # Separation
#    sigma = phiN_thetaN_R12ang_sigma_chiAbs_chiCD[3]
#            # broadening
#    chiAbs = phiN_thetaN_R12ang_sigma_chiAbs_chiCD[4]
#    chiCD = phiN_thetaN_R12ang_sigma_chiAbs_chiCD[5]
#            # scale factors


    
    phi = phiN * (Pi/180)       #
    theta = thetaN * (Pi/180)   # some quick conversions
    roll = rollN * (Pi/180)
    shift = shiftN *10**-(10)
    shear = shearN *10**-(10)
    R12 = R12ang  *10**-(10)
    lam = ma.sqrt(lambdaSq)
    # vector definitions in ext_dip.py
    
#    muA, muB = ed.dip_vecs(phi, theta)
    muA, muB = tc.dip_vecs(phi,theta,roll,shift,shear,R12)
    # here we generate the vector dependent operators:
    
    muTot = np.array([muA[i]*kr4(muOp, Iel, Ivib, Ivib) + \
             muB[i]*kr4(Iel, muOp, Ivib, Ivib) for i in range(3)])
    
    unitR = np.array([0, 0, 1])
    Rvec = R12*unitR
    
    magVecA = np.cross(unitR, muA)
    magVecB = np.cross(-unitR, muB) 
    
    magA = [magVecA[i]*muOp for i in (0,1,2)]
    magB = [magVecB[i]*muOp for i in (0,1,2)]
    
    op1 = kr4(magA[0], Iel, Ivib, Ivib) + kr4(Iel, magB[0], Ivib, Ivib)
    op2 = kr4(magA[1], Iel, Ivib, Ivib) + kr4(Iel, magB[1], Ivib, Ivib)
    op3 = kr4(magA[2], Iel, Ivib, Ivib) + kr4(Iel, magB[2], Ivib, Ivib)
    
      
    magOps = [op1, op2, op3]
    
    mVecA = muA*mumonomer 
    mVecB = muB*mumonomer
    
    # model, a global parameter, 0 for point dipole
    #                            1 for exten dipole

    if model == 0:
        # Point Coupling
        J = tc.J_point(phi, theta, roll, shift, shear, R12)
       # J = J2nubar*Coupling(mVecA, mVecB, Rvec)
        if make_plots:
            print("Point: ", J)           # debugging
    
    # if model == 1:
    #     # Extended
    #     j = ed.J_exten(phi, theta, R12, ed.l,ed.c)
    #     J = J2nubar * j
    #     if make_plots:
    #         print("Extended: ", J)           # debugging
            
    if model == 1:
        # Extended
        J = tc.J_exten(phi, theta, roll, shift, shear, R12)
        if make_plots:
            print("Extended: ", J)         # debugging
            
    if model == 2:
        J = tc.J_trans(phi, theta, roll, shift, shear, R12)
        if make_plots:
            print("Transition: ", J)
        
    RS = (H*nubar2nu*epsilon0/(4*Hbar)) * dot(np.cross(mVecA, mVecB), Rvec)
    
    Area = RS*epsilon0*nubar2nu/(7.659e-54)
    Height = Area/(sigma * ma.sqrt(2 * Pi) * nubar2nu) / 2
    
    
    # generate the full system hamiltonian
    
    h1 = epsilon0*kr4(cDc, Iel, Ivib, Ivib)
    h2 = epsilon0*kr4(Iel, cDc, Ivib, Ivib)
    h3 = J*kr(kr(kr(cD, c) + kr(c, cD), Ivib), Ivib)
    h4 = omega0*kr4(Iel, Iel, bDb, Ivib)
    h5 = omega0*kr4(Iel, Iel, Ivib, bDb)
    h6 = omega0*kr4(cDc, Iel, lam * (bD + b) + (lam**2) * Ivib, Ivib)
    h7 = omega0*kr4(Iel, cDc, Ivib, lam * (bD + b) + (lam**2 * Ivib))
    ham = h1 + h2 + h3 + h4 + h5 + h6 + h7

    # Diagonalize Hamiltonian
    eps, vecs = la.eig(ham)
        
    idx = eps.argsort()[::-1]   
    eps = eps[idx]
    vecs = vecs[:,idx]
    #print(vecs)           # debugging
    eps = np.flip(eps, 0)
    vecs = np.fliplr(vecs)

    # absorbtion intensities
    Ix = dot(muTot[0], vecs)[0]
    Iy = dot(muTot[1], vecs)[0]
    Iz = dot(muTot[2], vecs)[0]
    SimI = (Ix**2 + Iy**2 + Iz**2)*(2/3)
    
    AbsData = np.transpose([eps, SimI])
    
    # CD intensities
    cdk1 = - dot(muTot[0], vecs)[0] * dot(magOps[0], vecs)[0]
    cdk2 = - dot(muTot[1], vecs)[0] * dot(magOps[1], vecs)[0]
    cdk3 = - dot(muTot[2], vecs)[0] * dot(magOps[2], vecs)[0]
    
    cdTot = Height*(cdk1 + cdk2 + cdk3)
    np.set_printoptions(threshold=1000)
    
    CDdata = np.transpose([eps, cdTot])

    closeabsdata = Closeup(AbsData, low, high)
    closecddata = Closeup(CDdata, low, high)
    
    normAbs = PseudoVoigtDistribution(18282, gamma, sigma, 18282)
    
    # The variables cAbsSpectrum and cCDSpectrum (see below)
    # are external to the function definition (global) and are 
    # created from data files via the function "get_c_spectra"
    # which is defined below.
    
    simAbsSpectra = SimData(closeabsdata, cAbsSpectrum, gamma, sigma, normAbs)
    simCDSpectra = SimData(closecddata, cCDSpectrum, gamma, sigma, normAbs)
    
    # Implement internal chi fitting, to allow coarse grid searching for
    # start values:

    if chi_int:
        # Abs #
        fit_abs = lambda chi: sum([(cAbsSpectrum[i][1]-(chi*simAbsSpectra[i][1]))**2 
                              for i in range(len(simAbsSpectra))])
        res = minimize_scalar(fit_abs, bounds = [0, 1])
        chiAbs = res.x
        # CD #
        fit_cd = lambda chi: sum([(cCDSpectrum[i][1] - (chi * simCDSpectra[i][1]) )**2 
                              for i in range(len(simCDSpectra))])
        res = minimize_scalar(fit_cd, bounds = [0, 1])
        chiCD = res.x 
 #       print(chiAbs, chiCD)
        
    simAbsSpectra[:,1] *= chiAbs        
    simCDSpectra[:,1] *= chiCD
    
    
    # determine the chi^2 for both Abs and CD
    AbsRes = sum([(cAbsSpectrum[i][1] - simAbsSpectra[i][1])**2 for i in range(len(simAbsSpectra))])
    CDRes = sum([(cCDSpectrum[i][1] - simCDSpectra[i][1])**2 for i in range(len(simCDSpectra))])

    # For plotting: make_plots is a global Boolean flag which 
    # dictates whether a target function calculation generates
    # a plot.
    
    if make_plots:
        
        fig1 = plt.figure(1, figsize = (15,5))
        
        # sort data for plots #
        CDplus = []
        Absplus = []
        CDminus = []
        Absminus = []
        for i in range(len(CDdata)):
            if CDdata[i][1]>0:
                CDplus.append(CDdata[i])
                Absplus.append(AbsData[i])
            else:
                CDminus.append(CDdata[i])
                Absminus.append(AbsData[i])
        
        CDplus = np.array(CDplus)
        Absplus = np.array(Absplus)
        CDminus = np.array(CDminus)
        Absminus = np.array(Absminus)
        
        ### ABS ###
        ax1 = fig1.add_subplot(121)
        
        # bars #
        simAbsplus = SimData(Absplus, cAbsSpectrum, gamma, sigma, normAbs)
        simAbsminus = SimData(Absminus, cAbsSpectrum, gamma, sigma, normAbs)

        ax1.stem(Absplus[:,0], Absplus[:,1]*chiAbs, linefmt='r-', markerfmt='ro',basefmt=' ')
        ax1.plot(simAbsplus[:,0], simAbsplus[:,1]*chiAbs, color = "r",linestyle="dashed")
        
        ax1.stem(Absminus[:,0], Absminus[:,1]*chiAbs, linefmt='b-', markerfmt='bo',basefmt=' ')
        ax1.plot(simAbsminus[:,0], simAbsminus[:,1]*chiAbs, color = "b",linestyle = "dashed")
        
        # spectra #
        ax1.plot(cAbsSpectrum[:,0], cAbsSpectrum[:,1], color = "g",linewidth=3)
        ax1.plot(simAbsSpectra[:,0], simAbsSpectra[:,1], color = "k",linewidth=3)

        ax1.set_xlim((low, high))
        ax1.set_title("Absorption")
        ax1.set_ylabel("Intensity (arb. units)")
        ax1.set_xlabel(r'$\overline{\nu}\ (cm^{-1})$')
        ax1.axhline(0, color = "k")
        
        ### CD ###
        ax2 = fig1.add_subplot(122)
        
        # bars #
        simCDplus = SimData(CDplus, cCDSpectrum, gamma, sigma, normAbs)
        simCDminus = SimData(CDminus, cCDSpectrum, gamma, sigma, normAbs)
        
        ax2.stem(CDplus[:,0], CDplus[:,1]*chiCD, linefmt='r-', markerfmt='ro',basefmt=' ')
        ax2.plot(simCDplus[:,0], simCDplus[:,1]*chiCD, color = "r",linestyle="dashed")
        
        ax2.stem(CDminus[:,0], CDminus[:,1]*chiCD, linefmt='b-', markerfmt='bo',basefmt=' ')
        ax2.plot(simCDminus[:,0], simCDminus[:,1]*chiCD, color = "b",linestyle = "dashed")
        
        # spectra #
        ax2.plot(cCDSpectrum[:,0], cCDSpectrum[:,1], color = "g",linewidth=3)
        ax2.plot(simCDSpectra[:,0], simCDSpectra[:,1], color = "k",linewidth=3)
        
        ax2.set_xlim((low, high))
        ax2.set_title("CD")
        ax2.set_ylabel("Intensity (arb. units)")
        ax2.set_xlabel(r'$\overline{\nu}\ (cm^{-1})$')
        ax2.axhline(0, color = "k")
        fig1.savefig("m1m2sym.pdf", bbox_inches = "tight")
        plt.show()
    
 #   For minimization:
    #print("*", end = '')  # this debugging print statement just 
                           #  makes it so you know each function
                            # evaluation happens.
    
    
  #   return the residues with appropriate weights:
 #   print(AbsRes, CDRes)
    return np.real(AbsRes * 1000 + CDRes * .005) # Target value!!




def Closeup(data, low, high):
    '''trims a data set down to fit in x range between low and high
    
    array, num, num -> array'''
    
    temp = []
    for dat in data:
        if (dat[0] > low) and (dat[0] < high):
            temp.append(dat)
    return np.array(temp)

def get_c_spectra(temp):
    '''generates Abs and CD spectra from a text file
    
    str (e.g. "J89"), str (e.g. "15") -> list of arrays'''
    
#    fileAbs = open("Fork_Temp_Data/J89_65C_Abs_visible.txt", "r")
#    fileCD = open("Fork_Temp_Data/J89_65C_CD_visible.txt", "r")
    
    #duplex
    # fileAbs = open("Duplex_Temp_Data/D12_"+str(temp)+"C_Abs_visible.txt", "r")
    # fileCD = open("Duplex_Temp_Data/D12_"+str(temp)+"C_CD_visible.txt", "r")
    
    #fork
   # fileAbs = open("Fork_Temp_Data/J89_"+str(temp)+"C_Abs_visible.txt", "r")
   # fileCD = open("Fork_Temp_Data/J89_"+str(temp)+"C_CD_visible.txt", "r")
    
    #
#    fileAbs = open("D12_25C_Abs_visible.txt", "r")
#    fileCD = open("D12_25C_CD_visible.txt", "r")



# =============================================================================
# 20230807: Claire commented temporarily
# =============================================================================
    # fileAbs = open("C:/MarcusLab/CD data analysis/2023_0718/MolE/1_ABS_0.txt", "r")
    # fileCD = open("C:/MarcusLab/CD data analysis/2023_0718/MolE/1_CD_0.txt", "r")
    fileAbs = open("/Users/clairealbrecht/Dropbox/MATLAB_programs/claire_programs/from_Lulu/D12_25C_Abs_visible.txt", "r")
    fileCD = open("/Users/clairealbrecht/Dropbox/MATLAB_programs/claire_programs/from_Lulu/D12_25C_CD_visible.txt", "r")


    AbsSpec = []
    CDSpec = []
    
    for line in fileAbs:
        AbsSpec.append([float(line.split()[i]) for i in range(len(line.split()))])
    fileAbs.close()
    for line in fileCD:
        CDSpec.append([float(line.split()[i]) for i in range(len(line.split()))])
    fileCD.close()
    
    return [Closeup(AbsSpec, low, high), Closeup(CDSpec, low, high)]
    
    CDSpec = CDSpec
    
    
    
    
    
    
    
################################################################
##################### SUBROUTINES #########################
#### Single Run ##
'''produces a target function value between a simulated spectra and taken data
   for a single construct and temperature'''
''
model = 0 #2
chi_int = True
# Temp Depen Params #

sampleNumber = 1

temperatures = np.array([15, 25, 35, 45, 55, 
                        65, 75, 85])
epsilon0s = np.array([18285, 18277, 18266, 18262, 18280, 
                      18301,18323, 18309])
omega0s = np.array([1116, 1109, 1119, 1113, 1124, 
                    1103, 1072, 1091])
lambdaSqs = np.array([0.54, 0.56, 0.56, 0.56, 0.55, 
                      0.54, 0.54, 0.56])
    
# get params #
temp = temperatures[sampleNumber]
epsilon0 = epsilon0s[sampleNumber]
omega0 = omega0s[sampleNumber]
lambdaSq = lambdaSqs[sampleNumber]

cAbsSpectrum, cCDSpectrum = get_c_spectra(temp)

make_plots = True
graphic_out = "duplex_e.svg"

start = [ 77.63, 21.42, 104.27, 0, 0, 7.7*1e10, 316.3164, 0.077028, 0.37574886]

print(start)
print(TargetAbsCDSpectra(start))

d = np.array(TargetAbsCDSpectra(start))
# file = np.savetxt('output.txt', d, delimiter =',' )
#%%
'''

# ###############################################################

# ### Single Optimization ##


chi_int = True
model = 2

sampleNumber = 1

temperatures = np.array([15, 25, 35, 45, 55, 
                        65, 75, 85])
epsilon0s = np.array([18285, 18277, 18266, 18262, 18280, 
              18289, 18301, 18308, 18323, 18309])

omega0s = np.array([1116, 1109, 1119, 1113, 1124, 
            1107, 1103, 1100, 1072,1091])

lambdaSqs = np.array([0.54, 0.56, 0.56, 0.56, 0.55, 
              0.54, 0.54, 0.56, 0.54, 0.56])

temp = temperatures[sampleNumber]
epsilon0 = epsilon0s[sampleNumber]
omega0 = omega0s[sampleNumber]
lambdaSq = lambdaSqs[sampleNumber]
print(temp)
cAbsSpectrum, cCDSpectrum = get_c_spectra(temp)
make_plots = True #False

#ranges = [(62.61, 62.63), (57.58, 57.60), (91.07, 91.09), (-1.26, -1.24), (-1.46, -1.44), (9.1,9.3), (200,450), (.05, 1), (.05, .9)]
#ranges = [(107.95, 107.97), (6.31, 6.33), (115.14, 115.16), (-1.77, -1.74), (-3.04, -3.02), (6.70, 6.72), (344.61, 344.63), (0.01, 1), (0.01, 1)]
#ranges = [(56.77, 56.79), (65.28, 65.30), (92.59, 92.61), (0.96, 0.98), (3.04, 3.06), (10.53, 10.55), (297.7, 297.9), (0.01, 1), (0.01, 1)]
#ranges = [(68.77, 68.79), (60.34, 60.36), (77.25, 77.27), (-1.41, -1.39), (-2.81, -2.79), (9.67, 9.69), (300.0, 300.2), (0.01, 1), (0.01, 1)]
#ranges = [(108.28, 108.30), (7.72, 7.74), (114.66, 114.68), (-1.72, -1.70), (1.59, 1.61), (6.69, 6.71), (337.45, 337.47), (0.01, 1), (0.01, 1)]
ranges = [(90, 120), (0, 70), (60, 150), (-4, 4), (-4, 4), (5, 10), (210, 450), (0, 1), (0, 1)]
print("Ranges= "+str(ranges))
evo_output = differential_evolution(TargetAbsCDSpectra, ranges, polish = True,
                                    #mutation=(0.5, 1), recombination=(.5,1),
                                    updating='deferred',   workers=1)

print("Differential Evolution X = "+str(evo_output.x))
print("Differential Evolution Value= "+str(evo_output.fun))
make_plots = True
graphic_out = "minus1_m2.svg"
TargetAbsCDSpectra(evo_output.x)
file = np.savetxt('minus1_m2sym.txt', evo_output.x, delimiter =',' )

'''

##################################################################

####### Error Bar Calculation #######


'''
Calculates the +/- bars for each of the optiized values calculated from the
above loop, and stores them as lists of duples for each temp and construct.
'''


'''
# get optimum values: date format "2018-09-19"

#step sizes for ∆ param:
#make_plots = False
##model = 1
#chi_int = False
#steps = [1, 1, .1, 2]
steps = [.2, .2, .2, .1, .1, .1, 2]
###


# e_cont = np.loadtxt("Negative_1_parameters.txt", delimiter = ",")
# p_cont = np.loadtxt("Negative_1_parameters.txt", delimiter = ",")

#p_cont = np.loadtxt("plus1_m0_p.txt", delimiter = ",")

t_cont = np.loadtxt("minus1_m2sym_smooth.txt", delimiter = ",")

#t_cont = np.loadtxt("transdipoptimizedvalues.txt")
#PhiError= []
#ThetaError = []
#RError = []
#RollError = []
#SigmaError = []
#

#print(t_cont)
#print(len(t_cont))

opt_xs_e = []
opt_xs_p = []
opt_xs_t = []

# for line in e_cont:
#     opt_xs_e.append(line[2:])

# for line in p_cont:
#     opt_xs_p.append(line[2:])
    
#for line in t_cont:
opt_xs_t = t_cont[0:]
#opt_xs_p = p_cont[2:]

opt_xs_t = np.array(opt_xs_t)
#opt_xs_p = np.array(opt_xs_p)
#print(len(opt_xs_t))
##
###
###
####Temp dependent params:
###
temperatures = np.array([15,25,35,45,55,65,70,75,85])

epsilon0s = np.array([18285, 18277, 18266, 18262, 18280, 
                      18289, 18301, 18308, 18323, 18309])

omega0s = np.array([1116, 1109, 1119, 1113, 1124, 
                    1107, 1103, 1100, 1072,1091])

lambdaSqs = np.array([0.54, 0.56, 0.56, 0.56, 0.55, 
                      0.54, 0.54, 0.56, 0.54, 0.56])

data = []
    
new_data=[]

sampleNumber=1
# get params #
temp = temperatures[sampleNumber]
epsilon0 = epsilon0s[sampleNumber]
omega0 = omega0s[sampleNumber]
lambdaSq = lambdaSqs[sampleNumber]
# 
print("TEMP: " + str(temp))

cAbsSpectrum, cCDSpectrum = get_c_spectra(temp)

#opt_x_e = opt_xs_e
#print(opt_x_e)    
#opt_x_p = opt_xs_p
#print(opt_x_p)
opt_x_t = opt_xs_t
# #print(opt_x_t)
    
    
#model = 0
#chi_p = TargetAbsCDSpectra(opt_x_p)
    
    # model = 1
    # chi_e = TargetAbsCDSpectra(opt_x_e)
    
model = 2
#print(opt_x_t)
chi_t = TargetAbsCDSpectra(opt_x_t)
print(chi_t)
    

#    fig = plt.figure(figsize = (10,10))
    
#    phis = fig.add_subplot(221)
#    thetas = fig.add_subplot(222)
#    rs = fig.add_subplot(223)
#    sigs = fig.add_subplot(224)
#   rolls = fig.add_subplot(225)

#    plots = [phis, thetas, rs, sigs]
    
params = ["Phi", "Theta", "Roll", "Shift", "Shear", "R", "Sigma"]

  #  both = [opt_x_p, opt_x_e, opt_x_t]
    
  #  chis = [chi_p, chi_e, chi_t]
    
both = [opt_x_t]
chis = [chi_t]
make_plots = False
print(chis)
sampleNumber = 1
for param in range(7):
    i = 0
    print(params[param])
    sub_data = []
    for opt_x in both:
        model = 2 #i
        title1 = params[param]
        opt_x_copy = opt_x.copy()
        targets = []
        step = steps[param]
        fob = 1
        
        # Forward
        while step > .0001:
            opt_x_copy[param] += (fob)*step
            new_chi = TargetAbsCDSpectra(opt_x_copy)
         
            if (fob * ((new_chi-chis[i])/(chis[i]))) >= fob*.01:
                step = step/5
                fob = -fob
        plus = opt_x_copy[param] - opt_x[param]
        
        opt_x_copy = opt_x.copy()
        step = steps[param]
        fob = 1
#        file1 = open(str(params[param])+"PlusError.txt", 'a')
#        file1.write(str(plus))
#        file1.write('/n')
#        file1.close()
      #  thing1 = str(params[param])+'Error'
      #  thing1.append(plus)        
        
        print('fwd')
        
        # Backward
        while step > .0001:
            opt_x_copy[param] -= (fob)*step
            new_chi = TargetAbsCDSpectra(opt_x_copy)
            if (fob * ((new_chi-chis[i])/(chis[i]))) >= fob*.01:
                step = step/5
                fob = -fob
        minus = opt_x_copy[param] - opt_x[param]
        
        sub_data.append([plus, minus])
#        file2 = open(str(params[param])+"MinusError.txt",'a')
#        file2.write(str(minus))
#        file2.close()
      #  thing2 = str(params[param])+'Error'
      #  thing2.append(minus)
        
        print('bkwd')
        i += 1
    print(sub_data)
    data.append(sub_data)
    for elem in sub_data:
        new_data.append(elem)

  #      output error data

new_data=pd.DataFrame(new_data)
new_data.to_csv('Error_minus1_m2sym.csv') #used to be city3 generator
'''