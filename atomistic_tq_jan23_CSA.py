# -*- coding: utf-8 -*-
"""
Created on Tue Oct 15 15:04:44 2019

@author: Dylan
"""


"""
Hold the functions to execute the atomistic transition charge calculation for the transition dipole
moment of constiuent moments and J coupling based on a coulombic interaction scheme

Used by TransitionCharge_Roll_Shear.py or equivalent

This script requires:
1. atoms.txt file which holds the atom number and type, which
is used to calculate the van der Waals radii for each atom.

2. xyzchargeforcy3.txt file or equivalent, which holds the cartesian position and 
partial transition charge information for each atom

Notes:
    
    Optimal monomer plotting at azim=90, elev=90 
    

"""

import math as ma
Pi = ma.pi
import numpy as np
from numpy import linalg as la
from matplotlib import pyplot as plt
import numpy as np
from numpy import loadtxt
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib as matplotlib
from scipy import interpolate



############################################################
######Run Modes#############################################

charge_style = False
inter_atom = True
atom_clash = False

saveimag = False

#Sets the roll action
symmetric = True
asymmetric = False
roll_mode = "symmetric"


#Sets the cutoff interatom distance for overlap flag
vdw_cutoff = 3.0e-10

l = 7e-10

def mag(x): 
    return ma.sqrt(sum(i**2 for i in x))

def normalize(v):
    norm = np.linalg.norm(v)
    if norm == 0: 
       return v
    return v / norm

#################
###Empty Arrays##

vdw_flags_count = []

############################################################

# Debye:
D = 3.33e-30
mumonomer = 12.8 * D

# charge
c = .38*1.6e-19
k = 8.99e9
E = 1.602176e-19   
r = 5e-10

############# Constants ##################
c0 = 2.9979e8
H = 6.626e-34
Hbar = 1.054e-34
nubar2nu = 100*c0
permFree = 8.8542e-12
J2nubar = 1/(100*c0*H)
k = 8.99e9
E = 1.602176e-19            #converts fundamental charge to coulombs 
#  Ke = Coulomb Constant, 8.9875517873681764 * 109 N.m2.C-
deb = 2.99e-29
        
#######################################################################
#############Load in the atom number and identification################

df = pd.read_csv('atoms.txt', delimiter= '\s+', names = ['Atom Number', 'Atom Type'])


atom_type = np.array(df['Atom Type'])

#Designate a list for the corresponding van der Waals radii
vdw_radii = []
for i in atom_type:
   # print(i) 
    if i == 'C':
        vdw_radii.append(1.70)
    if i == 'H':
        vdw_radii.append(1.1)
    if i == 'N':
        vdw_radii.append(1.55)
vdw_radii = np.array(vdw_radii)
#print(vdw_radii)
vdw_circle = np.power(vdw_radii,8)*Pi * 3 #vdw radii scaled for plotter sq and *100
#print(vdw_circle)

##Designate list for atom colors 
atom_color = []
for i in atom_type:
   # print(i) 
    if i == 'C':
        atom_color.append('b')
    if i == 'H':
        atom_color.append('w')
    if i == 'N':
        atom_color.append('k')


########################################################################
#############Load in the Position and Charge Data#######################

coordscharge = loadtxt('20200316_xyzchargecy3.txt', delimiter='\t')
coords = coordscharge[:,0:3]
charges = coordscharge[:,3]
coordsA = np.array(coords)*10**-10
coordsB = np.array(coordsA)# + [0,0,r])
chargesA = coordscharge[:,3] * E * (12.8/12.27)
chargesB = coordscharge[:,3] * E * (12.8/12.27) 

########################################################################
########### Calculation the transition dipole moments ##################

initial_roll = (270*(Pi/180)) 

def dip_vecs(phi, theta, roll, shift, shear, R12):    
  

    RotateZA= [[ma.cos(phi/2), -ma.sin(phi/2), 0],
               [ma.sin(phi/2), ma.cos(phi/2), 0],
               [0,0,1]]
    RotateZB= [[ma.cos(-phi/2), -ma.sin(-phi/2), 0],
               [ma.sin(-phi/2), ma.cos(-phi/2), 0],
               [0,0,1]]
    
    RotateYA = [[ma.cos(theta/2), 0 , ma.sin(theta/2)],
                 [0, 1, 0],
                 [-ma.sin(theta/2), 0, (ma.cos(theta/2))]]
    
    RotateYB = [[ma.cos(-theta/2), 0 , ma.sin(-theta/2)],
                 [0, 1, 0],
                 [-ma.sin(-theta/2), 0, (ma.cos(-theta/2))]]
    
   
    # Initial Roll
     
    RotateXAi = [[1, 0 , 0],
                [0, ma.cos(initial_roll), -ma.sin(initial_roll)],
                [0, ma.sin(initial_roll), (ma.cos(initial_roll))]]
    
    RotateXBi = [[1, 0 , 0],
                [0, ma.cos(-initial_roll), -ma.sin(-initial_roll)],
                [0, ma.sin(-initial_roll), (ma.cos(-initial_roll))]]
   
    #Actual Roll
    
    RotateXA = [[1, 0 , 0],
                [0, ma.cos(roll), -ma.sin(roll)],
                [0, ma.sin(roll), (ma.cos(roll))]]
  
    
    if asymmetric:
        RotateXB = [[1, 0 , 0],
                    [0, ma.cos(-roll), -ma.sin(-roll)],
                    [0, ma.sin(-roll), (ma.cos(-roll))]]
    if symmetric:
        RotateXB = [[1, 0 , 0],
                    [0, ma.cos(roll), -ma.sin(roll)],
                    [0, ma.sin(roll), (ma.cos(roll))]]
                
    newA = []
    for i, val in enumerate(coordsA):
        rotatedcoords = np.dot(np.dot(np.dot(np.dot(coordsA[i,:],RotateXAi),RotateXA),RotateYA),RotateZA)
        newA.append(rotatedcoords)
    newA = np.insert(newA,3,coordscharge[:,3],axis=1)
    newA = np.array(newA)
    
########Applies a Displ to Chromophore A    
    
#     R12 = R12 
#     R12_dipl = (np.ones(len(newA[:,2])) * R12) + newA[:,2]
# #    print(shear_dipl)
# #    print(newA[:,2]-shear_dipl)
#     newA[:,2] = R12_dipl
    

# ####### Adds a shear diplacement between the two molecules 
#     shear = shear 
#     shear_dipl = (np.ones(len(newA[:,0])) * shear) + newA[:,0]
#     newA[:,0] = shear_dipl
# #    print(newA)
    
# ########Applies a Shift to Chromophore A    
    
#     shift = shift 
#     shift_dipl = (np.ones(len(newA[:,1])) * shift) + newA[:,1]
# #    print(shear_dipl)
# #    print(newA[:,2]-shear_dipl)
#     newA[:,1] = shift_dipl
    
    newB = []
    for i, val in enumerate(coordsB):
        rotatedcoords = np.dot(np.dot(np.dot(np.dot(coordsB[i,:],RotateXBi),RotateXB),RotateYB),RotateZB) #+ R12 #never any R12 here
        newB.append(rotatedcoords)
    newB = np.insert(newB,3,coordscharge[:,3],axis=1)
    newB = np.array(newB)


################################################################
#######Calculates the transition dipole moment in cartesian space    
    
    DipoleA = []
    for i, val in enumerate(newA):
        rCOM = newA[i,3]*E*((newA[i,0:3]))
        DipoleA.append(rCOM)
    DipoleA = np.array(DipoleA)
    DipoleA = [sum(DipoleA[:,0]) , sum(DipoleA[:,1]), sum(DipoleA[:,2])]
    DipoleA = DipoleA/np.sqrt(DipoleA[0]**2+DipoleA[1]**2+DipoleA[2]**2)
    
    DipoleB = []
    for i, val in enumerate(newB):
        rCOM = newB[i,3]*E*((newB[i,0:3]))
        DipoleB.append(rCOM)
    DipoleB = np.array(DipoleB)
    DipoleB = [sum(DipoleB[:,0]) , sum(DipoleB[:,1]), sum(DipoleB[:,2])]
    DipoleB = DipoleB/np.sqrt(DipoleB[0]**2+DipoleB[1]**2+DipoleB[2]**2)
    
    return [np.array(DipoleA), np.array(DipoleB)]

    newdistances = []
    for i, val in enumerate(newA):
        rAB = ((newA[i,0]-newB[:,0])**2+(newA[i,1]-newB[:,1])**2+(newA[i,2]-newB[:,2])**2)
        newdistances.append(rAB)
    newdistances = np.sqrt(np.array(newdistances))
    
    #may want to turn this into linalg.norm found in the J plotter
    
   # flags = np.count_nonzero(newdistances < vdw_cutoff)
  # #   flagsfinal = []
    # if flags > 0:
    #     return [np.array([0,0,1]), np.array([0,0,1])]  #if clash happens,it returns monomer value
    # else: 
    
    return [np.array(DipoleA), np.array(DipoleB)]

     
def J_trans(phi, theta, roll, shift, shear, R12):
    '''Coupling for atomistc transition charges:
    rads, rads, meters, meters, coulombs -> Joules'''
    
    RotateZA= [[ma.cos(phi/2), -ma.sin(phi/2), 0],
               [ma.sin(phi/2), ma.cos(phi/2), 0],
               [0,0,1]]
    RotateZB= [[ma.cos(-phi/2), -ma.sin(-phi/2), 0],
               [ma.sin(-phi/2), ma.cos(-phi/2), 0],
               [0,0,1]]
    
    RotateYA = [[ma.cos(theta/2), 0 , ma.sin(theta/2)],  ##Check theta
                 [0, 1, 0],
                 [-ma.sin(theta/2), 0, (ma.cos(theta/2))]]
    
    RotateYB = [[ma.cos(-theta/2), 0 , ma.sin(-theta/2)],
                 [0, 1, 0],
                 [-ma.sin(-theta/2), 0, (ma.cos(-theta/2))]]
    
    # Initial Roll
     
    RotateXAi = [[1, 0 , 0],
                [0, ma.cos(initial_roll), -ma.sin(initial_roll)],
                [0, ma.sin(initial_roll), (ma.cos(initial_roll))]]
    
    RotateXBi = [[1, 0 , 0],
                [0, ma.cos(-initial_roll), -ma.sin(-initial_roll)],
                [0, ma.sin(-initial_roll), (ma.cos(-initial_roll))]]
    
   
    #Actual Roll
    
    RotateXA = [[1, 0 , 0],
                [0, ma.cos(roll), -ma.sin(roll)],
                [0, ma.sin(roll), (ma.cos(roll))]]
    
    if asymmetric:
        RotateXB = [[1, 0 , 0],
                    [0, ma.cos(-roll), -ma.sin(-roll)],
                    [0, ma.sin(-roll), (ma.cos(-roll))]]
    if symmetric:
        RotateXB = [[1, 0 , 0],
                    [0, ma.cos(roll), -ma.sin(roll)],
                    [0, ma.sin(roll), (ma.cos(roll))]]
     
    newA = []
    for i, val in enumerate(coordsA):
        rotatedcoords = np.dot(np.dot(np.dot(np.dot(coordsA[i,:],RotateXAi),RotateXA),RotateYA),RotateZA)
        newA.append(rotatedcoords)
    newA = np.insert(newA,3,coordscharge[:,3],axis=1)
    newA = np.array(newA)
    
########Applies a Displ to Chromophore A    
    
    R12 = R12 
    R12_dipl = (np.ones(len(newA[:,2])) * R12) + newA[:,2]

    newA[:,2] = R12_dipl


###### Adds a shear diplacement between the two molecules 
    
    shear = shear 
    shear_dipl = (np.ones(len(newA[:,0])) * shear) + newA[:,0]
    newA[:,0] = shear_dipl

    
########Applies a Shift to Chromophore A    
    
    shift = shift 
    shift_dipl = (np.ones(len(newA[:,1])) * shift) + newA[:,1]
    newA[:,1] = shift_dipl
   
    
##########
    
    newB = []
    for i, val in enumerate(coordsB):
        rotatedcoords = np.dot(np.dot(np.dot(np.dot(coordsB[i,:],RotateXBi),RotateXB),RotateYB),RotateZB) #+ R12 #never any R12 here
        newB.append(rotatedcoords)
    newB = np.insert(newB,3,coordscharge[:,3],axis=1)
    newB = np.array(newB)


    newdistances = []
    for i, val in enumerate(newA):
        rAB = ((newA[i,0]-newB[:,0])**2+(newA[i,1]-newB[:,1])**2+(newA[i,2]-newB[:,2])**2)
        newdistances.append(rAB)
    newdistances = np.sqrt(np.array(newdistances))
    
    Jvalues2 = []
    coulomblist2 = []
    for i, val in enumerate(chargesA):
        for j, val in enumerate(chargesB):
            coulomb2 = (k * chargesA[i] * chargesB[j]) / newdistances[i,j]
            coulomblist2.append(coulomb2)
    Jvalues2.append((np.array(sum(coulomblist2)))*J2nubar)  
 
    
    
    flags = np.count_nonzero(newdistances < vdw_cutoff)
#    flagsfinal = []
    if flags > 0:
        return 10  #if clash happens,it returns monomer value
    else: 
        return Jvalues2



##############HERES WHERE A ADD MY GIFS BACK IN!!!!!!
    

##################################################
    
def van_der_Waals_allowed(phi, theta, roll, shift, shear, R12):
    '''Flags if a conformation is disallowed due to encrouchment upon
    van der Waals distances at consituent atoms for a given
    molecular arrangement
    '''
   # roll = roll - (90*(Pi/180)) #shifts the conformation at the start 

    RotateZA= [[ma.cos(phi/2), -ma.sin(phi/2), 0],
               [ma.sin(phi/2), ma.cos(phi/2), 0],
               [0,0,1]]
    RotateZB= [[ma.cos(-phi/2), -ma.sin(-phi/2), 0],
               [ma.sin(-phi/2), ma.cos(-phi/2), 0],
               [0,0,1]]
    
    RotateYA = [[ma.cos(theta/2), 0 , ma.sin(theta/2)],
                 [0, 1, 0],
                 [-ma.sin(theta/2), 0, (ma.cos(theta/2))]]
    
    RotateYB = [[ma.cos(-theta/2), 0 , ma.sin(-theta/2)],
                 [0, 1, 0],
                 [-ma.sin(-theta/2), 0, (ma.cos(-theta/2))]]
    
    # Initial Roll
     
    RotateXAi = [[1, 0 , 0],
                [0, ma.cos(initial_roll), -ma.sin(initial_roll)],
                [0, ma.sin(initial_roll), (ma.cos(initial_roll))]]
    
    RotateXBi = [[1, 0 , 0],
                [0, ma.cos(-initial_roll), -ma.sin(-initial_roll)],
                [0, ma.sin(-initial_roll), (ma.cos(-initial_roll))]]
    
    
    #Actual Roll
    RotateXA = [[1, 0 , 0],
                [0, ma.cos(roll), -ma.sin(roll)],
                [0, ma.sin(roll), (ma.cos(roll))]]
    if asymmetric:
        RotateXB = [[1, 0 , 0],
                    [0, ma.cos(-roll), -ma.sin(-roll)],
                    [0, ma.sin(-roll), (ma.cos(-roll))]]
    if symmetric:
        RotateXB = [[1, 0 , 0],
                    [0, ma.cos(roll), -ma.sin(roll)],
                    [0, ma.sin(roll), (ma.cos(roll))]]
        
            
    newA = []
    for i, val in enumerate(coordsA):
        rotatedcoords = np.dot(np.dot(np.dot(np.dot(coordsA[i,:],RotateXAi),RotateXA),RotateYA),RotateZA)
        newA.append(rotatedcoords)
    newA = np.insert(newA,3,coordscharge[:,3],axis=1)
    newA = np.array(newA)
    
########Applies a Displ to Chromophore A  (R12)  
    
    R12 = R12 
    R12_dipl = (np.ones(len(newA[:,2])) * R12) + newA[:,2]
    newA[:,2] = R12_dipl
    

####### Adds a shear diplacement between the two molecules 
    
    shear = shear 
    shear_dipl = (np.ones(len(newA[:,0])) * shear) + newA[:,0]
    newA[:,0] = shear_dipl

########Applies a Shift to Chromophore A    
    
    shift = shift 
    shift_dipl = (np.ones(len(newA[:,1])) * shift) + newA[:,1]
    newA[:,1] = shift_dipl
    
##########
    
    newB = []
    for i, val in enumerate(coordsB):
        rotatedcoords = np.dot(np.dot(np.dot(np.dot(coordsB[i,:],RotateXBi),RotateXB),RotateYB),RotateZB) #+ R12 #never any R12 here
        newB.append(rotatedcoords)
    newB = np.insert(newB,3,coordscharge[:,3],axis=1)
    newB = np.array(newB)

    
#########One way to do this is through the coulomb interaction itself
    newdistances = []
    for i, val in enumerate(newA):
        rAB = ((newA[i,0]-newB[:,0])**2+(newA[i,1]-newB[:,1])**2+(newA[i,2]-newB[:,2])**2)
        newdistances.append(rAB)
    newdistances = np.sqrt(np.array(newdistances))
    newdistances = np.array(newdistances)
    # print(newdistances[0])
    # print(len(newdistances))
    
    if inter_atom:
        plt.figure(figsize=(18,14))
        plt.contourf(newdistances)
        plt.colorbar()
        plt.xlabel('Index of atom on Molecule A')
        plt.ylabel('Index of atom on Molecule B')
        plt.title('Intermolecular Atomic COM-COM Distances')
        if atom_clash:
            plt.xticks(range(56), atom_type)
            plt.yticks(range(56), atom_type)
            
            #for i, j in newdistances[i,j] < vdw_cutoff:
            #    plt.append(newdistances[i,j])
        plt.show()
        atom_color = []
   
    
    # validmovechecker = []

    # for i in newdistances:
    #      if any(i <= 3.4e-10): #should be twice the van der waals radii
    #          validmovechecker.append(0) #change this so it checks the entire array
    #      else:
    #          validmovechecker.append(1)
    # print(validmovechecker)
        
    # if any distance in newdistances < 3e-10:
    # if any(i in validmovechecker[i] <= .5):
    #     print(0)
  #  if np.argwhere(newdistances < 1) =

    
    # for ix, row in enumerate(newdistances) for iy, i in enumerate(row) if i < 3.4e-10:
    #     validmovechecker.append(0)
        
    # for ix, row in enumerate(newdistances) for iy, i in enumerate(row) if i >= 3.4e-10:
    #     validmovechecker.append(1)
    
    
    # flagsfinal = []    
    # for i in newdistances[:,1]:
    #     if i < 3.4e-10:
    #         flagsfinal.append(1)
    #     else:
    #         flagsfinal.append(0)
    
    flags = np.count_nonzero(newdistances < vdw_cutoff)
    flagsfinal = []
    if flags > 0:
        flagsfinal.append(np.array(1))
    else: 
        flagsfinal.append(np.array(0))
    
    return flagsfinal
    print(flagsfinal)


    
def J_exten(phi, theta, roll, shift, shear, R12):
    
    '''Coupling for extended dipoles:
    rads, rads, meters, meters, coulombs -> Joules'''
 
    k = 8.99e9
    l = 7e-10
    q = .38*1.602176e-19
    
    #determine vector locations of 4 charges:
    vecs = dip_vecs(phi, theta, roll, shift, shear, R12)
    
  #  roll = roll - (90*(Pi/180)) #shifts the conformation 

    RotateZA= [[ma.cos(phi/2), -ma.sin(phi/2), 0],
               [ma.sin(phi/2), ma.cos(phi/2), 0],
               [0,0,1]]
    RotateZB= [[ma.cos(-phi/2), -ma.sin(-phi/2), 0],
               [ma.sin(-phi/2), ma.cos(-phi/2), 0],
               [0,0,1]]
    
    RotateYA = [[ma.cos(theta/2), 0 , ma.sin(theta/2)],
                 [0, 1, 0],
                 [-ma.sin(theta/2), 0, (ma.cos(theta/2))]]
    
    RotateYB = [[ma.cos(-theta/2), 0 , ma.sin(-theta/2)],
                 [0, 1, 0],
                 [-ma.sin(-theta/2), 0, (ma.cos(-theta/2))]]
    
    # Initial Roll
     
    RotateXAi = [[1, 0 , 0],
                [0, ma.cos(initial_roll), -ma.sin(initial_roll)],
                [0, ma.sin(initial_roll), (ma.cos(initial_roll))]]
    
    RotateXBi = [[1, 0 , 0],
                [0, ma.cos(-initial_roll), -ma.sin(-initial_roll)],
                [0, ma.sin(-initial_roll), (ma.cos(-initial_roll))]]
    
    
    
    #Actual Roll
    RotateXA = [[1, 0 , 0],
                [0, ma.cos(roll), -ma.sin(roll)],
                [0, ma.sin(roll), (ma.cos(roll))]]
    if asymmetric:
        RotateXB = [[1, 0 , 0],
                    [0, ma.cos(-roll), -ma.sin(-roll)],
                    [0, ma.sin(-roll), (ma.cos(-roll))]]
    if symmetric:
        RotateXB = [[1, 0 , 0],
                    [0, ma.cos(roll), -ma.sin(roll)],
                    [0, ma.sin(roll), (ma.cos(roll))]]
                
          
    newA = []
    for i, val in enumerate(coordsA):
        rotatedcoords = np.dot(np.dot(np.dot(np.dot(coordsA[i,:],RotateXAi),RotateXA),RotateYA),RotateZA)
        newA.append(rotatedcoords)
    newA = np.insert(newA,3,coordscharge[:,3],axis=1)
    newA = np.array(newA)
   
    newB = []
    for i, val in enumerate(coordsB):
        rotatedcoords = np.dot(np.dot(np.dot(np.dot(coordsB[i,:],RotateXBi),RotateXB),RotateYB),RotateZB) #+ R12 #never any R12 here
        newB.append(rotatedcoords)
    newB = np.insert(newB,3,coordscharge[:,3],axis=1)
    newB = np.array(newB)

################################################################
#######Calculates the transition dipole moment in cartesian space    
    
    DipoleA = []
    for i, val in enumerate(newA):
        rCOM = newA[i,3]*E*((newA[i,0:3]))
        DipoleA.append(rCOM)
    DipoleA = np.array(DipoleA)
    DipoleA = [sum(DipoleA[:,0]) , sum(DipoleA[:,1]), sum(DipoleA[:,2])]
    DipoleA = DipoleA/np.sqrt(DipoleA[0]**2+DipoleA[1]**2+DipoleA[2]**2)
    
    DipoleB = []
    for i, val in enumerate(newB):
        rCOM = newB[i,3]*E*((newB[i,0:3]))
        DipoleB.append(rCOM)
    DipoleB = np.array(DipoleB)
    DipoleB = [sum(DipoleB[:,0]) , sum(DipoleB[:,1]), sum(DipoleB[:,2])]
    DipoleB = DipoleB/np.sqrt(DipoleB[0]**2+DipoleB[1]**2+DipoleB[2]**2)
    
    vecs =  [np.array(DipoleA), np.array(DipoleB)]

    
    #J Interaction Calculation
    
    
    head1 = np.array([shear,shift,R12/2]) + vecs[0]*(l/2)
    head2 = np.array([0, 0,-R12/2]) + vecs[1]*(l/2)
    tail1 = np.array([shear,shift,R12/2]) - vecs[0]*(l/2)
    tail2 = np.array([0, 0,-R12/2]) - vecs[1]*(l/2)
    
    # head1 = np.array([0,0,0]) + vecs[0]*(l/2)
    # head2 = np.array([0,0,0]) + vecs[1]*(l/2)
    # tail1 = np.array([0,0,0]) - vecs[0]*(l/2)
    # tail2 = np.array([0,0,0]) - vecs[1]*(l/2)
    
    #calculate energy:
    head_head = k * q**2 / mag(head1-head2)
    tail_tail = k * q**2 / mag(tail1-tail2)
    head_tail =-k * q**2 / mag(head1-tail2)
    
    
    JvaluesExtend = (head_head + tail_tail + 2 * head_tail) * J2nubar
    
    ###########################################################################
    #######Flag (needs to add shift and shear into the Calculation)
    ##########################################################################

    
########Applies a Displ to Chromophore A    
    
    R12 = R12 
    R12_dipl = (np.ones(len(newA[:,2])) * R12) + newA[:,2]

    newA[:,2] = R12_dipl


###### Adds a shear diplacement between the two molecules 
    
    shear = shear 
    shear_dipl = (np.ones(len(newA[:,0])) * shear) + newA[:,0]
    newA[:,0] = shear_dipl

    
########Applies a Shift to Chromophore A    
    
    shift = shift 
    shift_dipl = (np.ones(len(newA[:,1])) * shift) + newA[:,1]
    newA[:,1] = shift_dipl
   
    
##########
    

    newdistances = []
    for i, val in enumerate(newA):
        rAB = ((newA[i,0]-newB[:,0])**2+(newA[i,1]-newB[:,1])**2+(newA[i,2]-newB[:,2])**2)
        newdistances.append(rAB)
    newdistances = np.sqrt(np.array(newdistances))
    
    Jvalues2 = []
    coulomblist2 = []
    for i, val in enumerate(chargesA):
        for j, val in enumerate(chargesB):
            coulomb2 = (k * chargesA[i] * chargesB[j]) / newdistances[i,j]
            coulomblist2.append(coulomb2)
    Jvalues2.append((np.array(sum(coulomblist2)))*J2nubar)  

    
    flags = np.count_nonzero(newdistances < vdw_cutoff)
#    flagsfinal = []
    if flags > 0:
        return 10  #if clash happens,it returns monomer value
    else: 
        return JvaluesExtend

def J_point(phi, theta, roll, shift, shear, R12):
    D = 3.33e-30
    mumonomer = 12.8 * D
    
    vecs = dip_vecs(phi, theta, roll, shift, shear, R12)
    mu1 = vecs[0] * mumonomer
    mu2 = vecs[1] * mumonomer
    
    R12point = np.array([shear,shift,R12])
    
    JvaluesPoint = (1/(4*Pi*permFree*(la.norm(R12point))**3))*(np.dot(mu1, mu2)- \
                                            3*(np.dot(R12point,mu1)*(np.dot(mu2,R12point)/(la.norm(R12point))**2))) 
        
    JvaluesPoint = JvaluesPoint * J2nubar
    
    ###########################################################################
    #######Flag for Atomic Overlap for a Particular Configuration##############
    ###########################################################################

    
   # roll = roll - (90*(Pi/180)) #shifts the conformation 
    
    RotateZA= [[ma.cos(phi/2), -ma.sin(phi/2), 0],
               [ma.sin(phi/2), ma.cos(phi/2), 0],
               [0,0,1]]
    RotateZB= [[ma.cos(-phi/2), -ma.sin(-phi/2), 0],
               [ma.sin(-phi/2), ma.cos(-phi/2), 0],
               [0,0,1]]
    
    RotateYA = [[ma.cos(theta/2), 0 , ma.sin(theta/2)],  ##Check theta
                 [0, 1, 0],
                 [-ma.sin(theta/2), 0, (ma.cos(theta/2))]]
    
    RotateYB = [[ma.cos(-theta/2), 0 , ma.sin(-theta/2)],
                 [0, 1, 0],
                 [-ma.sin(-theta/2), 0, (ma.cos(-theta/2))]]
    
    # Initial Roll
     
    RotateXAi = [[1, 0 , 0],
                [0, ma.cos(initial_roll), -ma.sin(initial_roll)],
                [0, ma.sin(initial_roll), (ma.cos(initial_roll))]]
    
    RotateXBi = [[1, 0 , 0],
                [0, ma.cos(-initial_roll), -ma.sin(-initial_roll)],
                [0, ma.sin(-initial_roll), (ma.cos(-initial_roll))]]
    
    
    #Actual Roll 
    RotateXA = [[1, 0 , 0],
                [0, ma.cos(roll), -ma.sin(roll)],
                [0, ma.sin(roll), (ma.cos(roll))]]
    if asymmetric:
        RotateXB = [[1, 0 , 0],
                    [0, ma.cos(-roll), -ma.sin(-roll)],
                    [0, ma.sin(-roll), (ma.cos(-roll))]]
    if symmetric:
        RotateXB = [[1, 0 , 0],
                    [0, ma.cos(roll), -ma.sin(roll)],
                    [0, ma.sin(roll), (ma.cos(roll))]]
        
    newA = []
    for i, val in enumerate(coordsA):
        rotatedcoords = np.dot(np.dot(np.dot(np.dot(coordsA[i,:],RotateXAi),RotateXA),RotateYA),RotateZA)
        newA.append(rotatedcoords)
    newA = np.insert(newA,3,coordscharge[:,3],axis=1)
    newA = np.array(newA)
    
    
########Applies a Displ to Chromophore A    
    
    R12 = R12 
    R12_dipl = (np.ones(len(newA[:,2])) * R12) + newA[:,2]

    newA[:,2] = R12_dipl


###### Adds a shear diplacement between the two molecules 
    
    shear = shear 
    shear_dipl = (np.ones(len(newA[:,0])) * shear) + newA[:,0]
    newA[:,0] = shear_dipl

    
########Applies a Shift to Chromophore A    
    
    shift = shift 
    shift_dipl = (np.ones(len(newA[:,1])) * shift) + newA[:,1]
    newA[:,1] = shift_dipl
   
    
##########
    
    newB = []
    for i, val in enumerate(coordsB):
        rotatedcoords = np.dot(np.dot(np.dot(np.dot(coordsB[i,:],RotateXBi),RotateXB),RotateYB),RotateZB) #+ R12 #never any R12 here
        newB.append(rotatedcoords)
    newB = np.insert(newB,3,coordscharge[:,3],axis=1)
    newB = np.array(newB)

    newdistances = []
    for i, val in enumerate(newA):
        rAB = ((newA[i,0]-newB[:,0])**2+(newA[i,1]-newB[:,1])**2+(newA[i,2]-newB[:,2])**2)
        newdistances.append(rAB)
    newdistances = np.sqrt(np.array(newdistances))
    
    Jvalues2 = []
    coulomblist2 = []
    for i, val in enumerate(chargesA):
        for j, val in enumerate(chargesB):
            coulomb2 = (k * chargesA[i] * chargesB[j]) / newdistances[i,j]
            coulomblist2.append(coulomb2)
    Jvalues2.append((np.array(sum(coulomblist2)))*J2nubar)  

    
    
    flags = np.count_nonzero(newdistances < vdw_cutoff)
#    flagsfinal = []
    if flags > 0:
        return 10  #if clash happens,it returns monomer value
    else: 
        return JvaluesPoint

     
    
    
    
    
    