#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 15 16:12:05 2023

@author: calbrecht

This code is the current version as of 20230724 to optimize t32=0 
and plot t32=/=0 with the optimized parameters. It also saves the 
params in the data folder.


GOAL: ADJUST SIM FOR SELECTION RULE APPROACH


"""

import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

from scipy.interpolate import interp2d
import scipy
import scipy.optimize as opt

# import data .mat file
import os
import glob
    
#%%
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


# vals = np.abs(NRP_tau_raw_exp) #np.array([[-5., 0], [5, 10]]) 
# vmin = vals.min()
# vmax = vals.max()

# norm = MidpointNormalize(vmin=vmin, vmax=vmax, midpoint=0)
# cmap = 'jet' 

# plt.contourf(vals, cmap=cmap, norm=norm, shading='interp')
# plt.colorbar()
# plt.show()

#%%
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


#%%


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
    scan_params = scan_folder[len('20230101-120000-'):len(scan_folder)-len('_2DFPGA')]
    stages = scan_params[len(scan_params)-2:]
    scan_type = scan_params[:len(scan_params)-3]
    
    # Update 20231116
    if stages == 'xz':
        FT2D_mode = 0
    # files_mat = glob.glob('*.mat')
    if FT2D_mode == 1:
        file_FFT = glob.glob('*'+scan_type+'*'+stages+'*FFT2.mat')[0]
    else: 
        file_FFT = glob.glob('*'+scan_type+'*'+stages+'*FFT.mat')[0]
    file_RF_raw = glob.glob('*'+scan_type+'*'+stages+'*RF_raw*.mat')[0]
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

#%
global xaxis, yaxis, DQC_exp, NRP_exp, RP_exp , NRP_tau_exp, RP_tau_exp, DQC_tau_exp, t43ax, t21ax, dmatW, smatW
global timing_mode, FPGA_mode, sample_name




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

date_folder = '20230801' # two good data sets
scan_folder_nrprp = '20230801-115033-NRP_RP_xz_2DFPGA'
scan_folder_dqc = '20230801-130235-DQC_xz_2DFPGA'
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


# =============================================================================
# 
# =============================================================================
scan_folder = scan_folder_nrprp
scan_params = scan_folder[len('20230101-120000-'):len(scan_folder)-len('_2DFPGA_FFT')]
stages = scan_params[len(scan_params)-2:]
scan_type = scan_params[:len(scan_params)-3]
# =============================================================================
timing_mode ='t32 = 0'
# timing_mode ='t21 = 0'
# timing_mode ='t43 = 0'
FT2D_mode = 1
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
import pandas as pd
def gauss2(x, lam1,lam2, sig1,sig2, amp1, amp2):
    return amp1 * np.exp(-(x-lam1)**2 / (2 *sig1**2)) + amp2*np.exp(-(x-lam2)**2 / (2 *sig2**2))
def gauss1(x, lam1, sig1, amp1):
    return amp1 * np.exp(-(x-lam1)**2 / (2 *sig1)) #**2))

def overlap_params(xs, mu1, mu2, sig1, sig2, amp1, amp2):
    mu_adj = (sig2**2 * mu1 + sig1**2 * mu2)/(sig1**2 + sig2**2)
    sig_adj = np.sqrt(1/((1/sig1**2) + (1/sig2**2)))
    return mu_adj, sig_adj

#%%    #%

def gauss(x, lam1, sig1, amp1):
    return amp1 * np.exp(-(x-lam1)**2 / (2 *sig1**2))
# =============================================================================
# Calculate FFT
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
        else: # => timing_mode = 't43 = 0'  # only take 2D transform when t32=0
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
    


def delayedGaussian(x,c,s): # used for windowing the 2D sim plots to make sure they go to zero at edges (avoid ringing in freq domain)
    w = np.ones(np.shape(x));
    shifted = x-c;
    index = shifted > 0;
    w[index] = np.exp(-4.5*(shifted[index]**2)/(s**2));
    return w    

# def gaussian2d(x, x0, sig):
#     w = np.ones(np.shape(x))
#     w = np.exp(-1*(x - x0)**2/(2*sig**2))
#     # mask = w > 1
#     # w[mask] = 1
#     return w   
 
#%%

def overlap_params(xs, mu1, mu2, sig1, sig2): # caluclate new mu and sig after overlapping molecule absorption spectrum with laser spectrum
    mu_adj = (sig2**2 * mu1 + sig1**2 * mu2)/(sig1**2 + sig2**2)
    sig_adj = np.sqrt(1/((1/sig1**2) + (1/sig2**2)))
    return mu_adj, sig_adj

def sim2Dspec2(t21, laser_lam, laser_fwhm, mu1_6MI, mu2_6MI, Gam, sigI,monoC_lam, omega_ge, omega_gep):
    #%
    c0 = 0.000299792458 # mm / fs
    # t21 = 2 * np.pi * 10 * c0 * t21
    nubar2omega = 1/ ((10) / (2 * np.pi * c0)) # where c0 is in mm/fs
    #  multiplying by nubar2omega converts cm^-1 to fs^-1 
    # ==> use this conversion in the exponential of the response functions
    
    laser_sig =  laser_fwhm / (2 * np.sqrt(2 * np.log(2)))
    
    laser_fwhm_omega = 10**7/(laser_lam - (laser_fwhm/2)) - 10**7/(laser_lam + (laser_fwhm/2))
    laser_sig_omega = laser_fwhm_omega / (2 * np.sqrt(2 * np.log(2))) # convert from FWHM to stdev of gaussian
    

    laser_omega = (10**7 / laser_lam) #/ 10**3 # convert wavelength to wavenumber

    # set monochromator wavelength and convert to wavenumber
    monoC_omega = (10**7 / monoC_lam) #/ 10**3

    # =============================================================================
    # Define the transition dipole moments
    # =============================================================================
    mu_ge = mu1_6MI/2 # let the g->e transition be half the g->f dipole moment
    mu_gep = mu2_6MI/2 # let the g->e' transition be half the g->f' dipole moment
    
    # these are the on-diagonal dipole moments
    mu_ef = mu_ge # let e->f = g->e = 1/2 mu_gf
    mu_epfp = mu_gep # let e'->f' = g->e' = 1/2 mu_gf'
    
    mu_gf = mu_ge + mu_ef
    mu_gfp = mu_gep + mu_epfp
    
    # mu_efp = mu_gfp - mu_ge
    # mu_epf = mu_gf - mu_gep
    
    # these are the cross peak dipole moments
    # mu_efp = mu_gep # let e->f' = g->e' (larger energy)
    # mu_epf = mu_ge # let e'->f = g->e (smaller energy)
    mu_efp = mu_ge 
    mu_epf = mu_gep 
    
    # =============================================================================
    # Write down the energies that correspond to the dipole moments
    # 20231215: this is where we will plug in the eigenenergies from the hamiltonian approach
    # =============================================================================
    
    delta = (omega_gep - omega_ge)/2 # 20231018 update
    
    omega_ef = omega_ge - delta
    omega_epfp = omega_gep - delta
    
    omega_gf = omega_ge + omega_ef
    omega_gfp = omega_gep + omega_epfp
    
    # omega_efp = omega_epfp + (2 * delta)
    # omega_epf = omega_ef - (2 * delta)
    omega_efp = omega_gfp - omega_ge
    omega_epf = omega_gf - omega_gep
        
    omega_ee = omega_epep = 0
    omega_eep = omega_gep - omega_ge # does this equal 2 * delta?
    
   
    nterms = 8 # the number of pathways through the two virtual states to population in each of the two real states
    # =============================================================================
    # Write down the response functions
    # =============================================================================
    # Ntimesteps = int(np.max(np.abs(t21ax))/(t21ax_rt[1] - t21ax_rt[0]))
    # t21 = np.linspace(0,116.0805,num=Ntimesteps)
    # start by transforming the time array into nxn arrays for the experiment you want to simulate 
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
    

    
    # Dipoles to multiply for all 8 pathways: put orientation factor in the orientation avg arr
    # the order of the mu's sets up the interaction path for: 
    #    |ket><bra|   |ket><bra|   |ket><bra|   |ket><bra|       |ket><bra|      |ket><bra|    |ket><bra|    |ket><bra|
        # gef,gef  gepfp, gepfp     gef, gepf    gepf,gef      gefp,gepfp       gepfp,gefp    gefp, gefp    gefp, gefp
    mu1 = [mu_ef,   mu_epfp,        mu_ef,       mu_epf,       mu_efp,          mu_epfp,      mu_efp,     mu_epf]
    mu2 = [mu_ge,   mu_gep,         mu_ge,       mu_gep,       mu_ge,           mu_gep,       mu_ge,      mu_gep]
    mu3 = [mu_ge,   mu_gep,         mu_gep,      mu_ge,        mu_gep,          mu_ge,        mu_ge,      mu_gep] 
    mu4 = [mu_ef,   mu_epfp,        mu_epf,      mu_ef,        mu_epfp,         mu_efp,       mu_efp,     mu_epf]
    #      l&diag   u&diag        l&OFFdiag             l&diag        u&diag         u&OFFdiag

    mu1 = np.array(mu1) 
    mu2 = np.array(mu2) 
    mu3 = np.array(mu3) 
    mu4 = np.array(mu4) 
   

    # Assuming we are only probing one electronic dipole transition (EDTM) moment in the molecule (and its virtual and vibrational states)
    orient_avg_arr = np.ones(8) * (1/5) 
    # 1/5 comes from the orientational average of the angle between the molecule EDTM   and the laser polarization (horizontal)
   
    #%
    cm_DQC = np.zeros([time_size,time_size],dtype='complex')
    cm_NRP = np.zeros([time_size,time_size],dtype='complex')
    cm_RP =  np.zeros([time_size,time_size],dtype='complex')
    # for i in [0,1,2,3,5,7]:
    # for i in [0,3,2,7,1,5,6,4]:
    # for i in [0,1,2,5,6,4,7,3]:
    for i in range(nterms):
    # for i in [0,1,2,5]:
        # print('i = ', i+1)
        
        # omegaN[i] for N=1,2,3 is an individual term giving one freq peak, currently 8 terms... hamiltonian form will hopefully generalize this
        omega1 = [omega_ge, omega_gep,   omega_gep,  omega_ge,   omega_gep,   omega_ge,     omega_ge,   omega_gep]
        omega2 = [omega_gf, omega_gfp,   omega_gf,   omega_gf,   omega_gfp,   omega_gfp,    omega_gfp,  omega_gf]        
        omega3 = [omega_ef, omega_epfp,  omega_ef,   omega_epf,  omega_efp,   omega_epfp,   omega_efp,  omega_epf]

        # print('omega1: '+str(omega1[i])+' omega3: '+str(omega3[i]))
        # print('mu1: '+str(mu1[i])+' mu2: '+str(mu2[i])+' mu3: '+str(mu3[i])+' mu4: '+str(mu4[i]))
        # print('mu1*mu2*mu3*mu4: '+str(mu1[i] * mu2[i] * mu3[i] * mu4[i]))
        
        # use inhomogenous linewidth as width of molecular absorption peak to calculate shifts (variable parameter)
        omega1 = (laser_omega * sigI**2 + np.array(omega1) * laser_sig_omega**2) / (sigI**2 + laser_sig_omega**2)
        omega2 = ((2*laser_omega) * sigI**2 + np.array(omega2) * (laser_sig_omega/2)**2) / (sigI**2 + (laser_sig_omega/2)**2)
        # need 2*laser omega for omega2 in the DQC calculation because omega2 is during t32 which has an |g><f| coherence = 2x energy of laser
        omega3 = (laser_omega * sigI**2 + np.array(omega3) * laser_sig_omega**2) / (sigI**2 + laser_sig_omega**2)


        # alphas are: what is the amplitude of the overlap between molecule abs and laser at the newly shifted energies (directly above) 
        alpha1 = gauss(np.array(omega1), laser_omega, laser_sig_omega,1)
        # alpha2 = gauss(np.array(omega2), laser_omega, laser_sig_omega,1) # things are funky about omega2... sort this out
        alpha3 = gauss(np.array(omega3), laser_omega, laser_sig_omega,1)
        alpha1 = alpha1/np.max(alpha1)
        # alpha2 = alpha2/np.max(alpha2) # things are funky about omega2... sort this out
        alpha3 = alpha3/np.max(alpha3)
        alpha = alpha1[i] * alpha3[i] #* alpha2[i]
        # product of the alphas will scale this peak intensity
        


        # subtract off monochromator reference frequency (because we are downsampling as explained in Tekavec 2006 & 2007)
        omega1 = (np.array(omega1) - monoC_omega)
        omega2 = (np.array(omega2) - (2 * monoC_omega) ) # factor of 2 came out of calculations
        omega3 = (np.array(omega3) - monoC_omega)

        
        cm_DQC += alpha * orient_avg_arr[i] * mu1[i] * mu2[i] * mu3[i] * mu4[i] * np.exp(1j*nubar2omega*(omega3[i]*t43 + omega2[i]*t32 + omega1[i]*t21)) \
                                                            * np.exp(-Gam*nubar2omega*(t43 + t32 + t21)) * np.exp(-(1/2)*(sigI*nubar2omega)**2*(t21 + t32 + t43)**2)

        # set up omega2's for the RP and NRP (different from DQC)                                                    
        omega2 = [omega_ee, omega_epep,   omega_eep,   omega_eep,   omega_eep,   omega_eep, omega_ee, omega_epep] 
        
        
        #apply freq shift from overlap with laser -- how to do this properly for omega 2?
        # omega2 = (laser_omega * sigI**2 + np.array(omega2) * laser_sig_omega**2) / (sigI**2 + laser_sig_omega**2)

        # alpha2 = gauss(np.array(omega2), laser_omega, laser_sig_omega,1)
        # alpha2 = alpha2/np.max(alpha2)
        alpha = alpha1[i]  * alpha3[i] #* alpha2[i] # how do we take care of omega2 here?

        # subtract of monochromator freq
        # omega2 = np.abs(np.array(omega2) - monoC_omega)
        # omega2 = monoC_omega - np.array(omega2)
        ##### omega2 doesn't get subtracted off for NRP & RP... comes out of calculations
        
        cm_NRP += alpha * orient_avg_arr[i] * mu1[i] * mu2[i] * mu3[i] * mu4[i] * np.exp(1j*nubar2omega*(omega3[i]*t43 + omega2[i]*t32 + omega1[i]*t21)) \
                                                            * np.exp(-Gam*nubar2omega*(t43 + t32 + t21)) * np.exp(-(1/2)*(sigI*nubar2omega)**2*(t21 + t32 + t43)**2)

        
        cm_RP += alpha * orient_avg_arr[i] * mu1[i] * mu2[i] * mu3[i] * mu4[i] * np.exp(1j*nubar2omega*(omega3[i]*(-1*t43) + omega2[i]*t32 + omega1[i]*t21)) \
                                                            * np.exp(-Gam*nubar2omega*(t43 + t32 + t21)) * np.exp(-(1/2)*(sigI*nubar2omega)**2*(t21 + t32 + (-1*t43))**2)

        ### this commented block is for testing
        # if timing_mode == 't32 = 0':
        #     time_ax = t21
        #     FT_dqc_temp, ax1, ax2 = FFT_2d(cm_DQC_temp, t21ax_rt, time_ax, monoC_lam)
        #     # FT_nrp_temp, ax1, ax2 = FFT_2d(cm_NRP_temp, t21ax_rt, time_ax, monoC_lam)
        #     # FT_rp_temp,  ax1, ax2 = FFT_2d(cm_RP_temp,  t21ax_rt, time_ax, monoC_lam)
        #     # FT_rp_temp = np.flipud(FT_rp_temp)
        
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
    
    
    # test response functions by plotting
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
    
    # rephase the time domain - isn't necessary unless we do something weird with extra phases applied
    # angle_dqc = np.angle(cm_DQC[0,0])
    # angle_nrp = np.angle(cm_NRP[0,0])
    # angle_rp = np.angle(cm_RP[0,0])
    
    # cm_DQC = cm_DQC * np.exp(-1j * angle_dqc)
    # cm_NRP = cm_NRP * np.exp(-1j * angle_nrp)
    # cm_RP = cm_RP * np.exp(-1j * angle_rp)

    
    cm_DQC = cm_DQC * w
    cm_NRP = cm_NRP * w
    cm_RP = cm_RP * w
    
    # plots to test the windowed time domains
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
    
    # plots to test the FFT
    # ax_lim = [np.min(ax1),np.max(ax1)]
    # # ax_lim = [27, 29] 
    # ax_lim = [28,30]
    # # ax_lim = [39, 41]
    # ax_lim = [13,16.5]
    # ax_lim = [14, 15.5]
    # # ax_lim=[min(ax1), max( ax1)]
    # plot2Dspectra(ax1_dqc, ax2_dqc, FT_dqc, n_cont, ax_lim=ax_lim, title=r'DQC($\omega$) with '+timing_mode_str, domain='freq',save_mode=0,file_name='20230101-120101-DQC_qq_2DFPGA_FFT',scan_folder='20230101-120101-DQC_qq_2DFPGA_FFT')#'($\tau_{32}$ = 0)',domain = 'freq')
    # plot2Dspectra(ax1, ax2, FT_nrp, n_cont, ax_lim=ax_lim, title=r'NRP($\omega$) with '+timing_mode_str,domain='freq')#'($\tau_{32}$ = 0)',domain = 'freq')
    # plot2Dspectra(ax1, ax2, FT_rp, n_cont, ax_lim=ax_lim, title=r'RP($\omega$) with '+timing_mode_str,domain='freq')#'($\tau_{32}$ = 0)',domain = 'freq')
#%

    return t1_out, t2_out, cm_DQC, cm_NRP, cm_RP, ax1_dqc, ax2_dqc, ax1_nrprp, ax2_nrprp, FT_dqc, FT_nrp, FT_rp

#%%
# =============================================================================
# Can we optimize the difference between the interpolated and simulated?
# =============================================================================
def find_nearest(array, value): # I think this was used when I was trying to window the chi-squared...
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx], idx

def makeGaussian(size, fwhm = 3, center=None): # this too for windowing the chisquared
    """ Make a square gaussian kernel.

    size is the length of a side of the square
    fwhm is full-width-half-maximum, which
    can be thought of as an effective radius.
    """

    x = np.arange(0, size, 1, float)
    y = x[:,np.newaxis]

    if center is None:
        x0 = y0 = size // 2
    else:
        x0 = center[0]
        y0 = center[1]

    # w,h=plt.figaspect(1.)
    # plt.figure(figsize=(w,h))
    # plt.contourf(ax1, ax2, makeGaussian(len(ax1), 20, [loc,loc]));
    # plt.xlim(ax_lim);
    # plt.ylim(ax_lim)
    return np.exp(-4*np.log(2) * ((x-x0)**2 + (y-y0)**2) / fwhm**2)


# def delayedGaussian(x,c,s):
#     w = np.ones(np.shape(x));
#     shifted = x-c;
#     index = shifted > 0;
#     w[index] = np.exp(-4.5*(shifted[index]**2)/(s**2));
#     return w 
#  #%
 
#%
# calculate chisquared difference between data and sim
def chisq_calc_int2(params):
    t21 = np.linspace(0,116.0805,num=Ntimesteps) # generalize this so that when data coming in is over a different range this doesn't cause problems...
    laser_lam, laser_sig, mu1_6MI, mu2_6MI, Gam, sigI, monoC_lam, omega_ge, omega_gep = params
    
    t1_out, t2_out, cm_DQC, cm_NRP, cm_RP, ax1_dqc, ax2_dqc, ax1_nrprp, ax2_nrprp, FT_dqc, FT_nrp, FT_rp = sim2Dspec2(t21, laser_lam, laser_fwhm, mu1_6MI, mu2_6MI, Gam, sigI, monoC_lam, omega_ge, omega_gep)


    # I think this is where I was trying to create a window for the difference map
    # sim_idx = np.array([459, 642])
    # sim_idx_28to30 = sim_idx
    # ax1_28to30 = ax1[sim_idx[0]:sim_idx[1]]

    # ax1_2d = np.tile(ax1_28to30, (len(ax1_28to30),1))
    # ax2_2d = ax1_2d.T
    # wx = gaussian2d(ax1_2d[0,:], 29.1, 1.5)
    # wy = gaussian2d(ax2_2d[:,0], 29.1, 1.5)
    # weight_func = np.reshape(wx,[len(wx),1]) * np.reshape(wy,[1,len(wy)]) #wx*wy
    # w,h = plt.figaspect(0.8)
    # plt.figure(figsize=[w,h])
    # plt.contourf(ax1_2d, ax2_2d,weight_func,cmap='jet')
    # ax_lim = [28,30]
    # plt.xlim(ax_lim)
    # plt.ylim(ax_lim)
    # plt.colorbar()
    
    # weight_func = np.ones(np.shape(ax1_2d))
    
    
    # val,loc = find_nearest(ax1, 10**7/laser_lam)
    # val,loc = find_nearest(ax1, 14.650)
    loc = 521
     
    xx = np.linspace(0,400,400)
    xx = np.tile(xx, (len(xx),1))
    yy = xx.T
    # w = delayedGaussian(np.sqrt(xx**2 + yy**2),80, 10); #70e-15,10e-15); 
    wind = delayedGaussian(np.sqrt(xx**2 + yy**2),4, 6); #70e-15,10e-15); 
    w,h=plt.figaspect(1.)
    # plt.figure(figsize=(w,h))
    # plt.contourf(xx,yy,wind)
    wind2d = np.hstack([np.rot90(wind,2),np.rot90(wind)])
    wind2d = np.vstack([wind2d,np.rot90(wind2d,2)])
    # plt.figure(figsize=(w,h))
    # plt.contourf(wind2d)
    
    wind2d_len = len(wind2d)
    zeros2add_left = int(loc - (wind2d_len/2))
    zeros2add_right = len(ax1) - (zeros2add_left + wind2d_len)
    downward_shift = 3
    # wind2d_full = np.hstack([np.zeros([len(wind2d),int(loc - (wind2d_len/2))]),wind2d])
    # wind2d_full = np.hstack([wind2d_full, np.zeros([len(wind2d),len(ax1) - wind2d_full.shape[1]])])
    wind2d_full = np.hstack([np.zeros([wind2d_len,zeros2add_left]),wind2d, np.zeros([wind2d_len,zeros2add_right])])
    wind2d_full = np.vstack([np.zeros([zeros2add_left-downward_shift,len(ax1)]), wind2d_full, np.zeros([zeros2add_right+downward_shift,len(ax1)])])
    # wind2d_full = np.hstack([wind2d_full, np.zeros([len(wind2d),len(ax1) - wind2d_full.shape[1]])])
    # plt.figure(figsize=(w,h))
    # plt.contourf(ax1, ax2, wind2d_full)#* DQC_exp)
    # plt.xlim(ax_lim)
    # plt.ylim(ax_lim)
    # plt.figure(figsize=(w,h))
    # plt.contourf(ax1, ax2, wind2d_full* DQC_exp)
    # plt.xlim(ax_lim)
    # plt.ylim(ax_lim)
    # plt.figure(figsize=(w,h))
    # plt.contourf(ax1, ax2, DQC_exp)
    # plt.xlim(ax_lim)
    # plt.ylim(ax_lim)

    # 
    weight_func_mode = 0
    if weight_func_mode == 0:
        weight_func = np.ones(np.shape(RP_exp))
    elif weight_func_mode == 1:
        weight_func = wind2d_full
    
    # FT_rp = FT_rp[sim_idx_28to30[0]:sim_idx_28to30[1], sim_idx_28to30[0]:sim_idx_28to30[1]]
    # FT_nrp = FT_nrp[sim_idx_28to30[0]:sim_idx_28to30[1],sim_idx_28to30[0]:sim_idx_28to30[1]]
    # FT_dqc = FT_dqc[sim_idx_28to30[0]:sim_idx_28to30[1],sim_idx_28to30[0]:sim_idx_28to30[1]]

    RP_exp_interp = RP_exp
    NRP_exp_interp = NRP_exp
    DQC_exp_interp = DQC_exp

    sim_denom = 1

    sim_denom = np.max(np.max(np.abs(np.real(FT_rp))))
    exp_denom = np.max(np.max(np.abs(np.real(RP_exp_interp))))
    # print('sim: '+str(sim_denom))
    if sim_denom == 0: # is there a better way to remove the zeros?
        sim_denom = 1
    #     print(params)
    chisq_rp_re = (np.abs(np.real(FT_rp))/sim_denom - np.abs(np.real(RP_exp_interp))/exp_denom)**2 * weight_func
    
    sim_denom = np.max(np.max(np.abs(np.imag(FT_rp))))
    if sim_denom == 0:
        sim_denom = 1
    exp_denom = np.max(np.max(np.abs(np.imag(RP_exp_interp))))
    chisq_rp_im = (np.abs(np.imag(FT_rp))/sim_denom - np.abs(np.imag(RP_exp_interp))/exp_denom)**2 * weight_func
    
    chisq_rp = np.abs(chisq_rp_re + 1j * chisq_rp_im)
    
    sim_denom = np.max(np.max(np.abs(np.real(FT_nrp))))
    if sim_denom == 0:
        sim_denom = 1
    exp_denom = np.max(np.max(np.abs(np.real(NRP_exp_interp))))
    chisq_nrp_re =(np.abs(np.real(FT_nrp))/sim_denom - np.real(NRP_exp_interp)/exp_denom)**2 * weight_func
    
    sim_denom = np.max(np.max(np.abs(np.imag(FT_nrp))))
    if sim_denom == 0:
        sim_denom = 1
    exp_denom = np.max(np.max(np.abs(np.imag(NRP_exp_interp))))
    chisq_nrp_im =(np.abs(np.imag(FT_nrp))/sim_denom- np.abs(np.imag(NRP_exp_interp))/exp_denom)**2 * weight_func
    
    chisq_nrp = np.abs(chisq_nrp_re + 1j * chisq_nrp_im)
    
    sim_denom = np.max(np.max(np.abs(np.real(FT_dqc))))
    if sim_denom == 0:
        sim_denom = 1
    exp_denom = np.max(np.max(np.abs(np.real(DQC_exp_interp))))
    chisq_dqc_re =(np.abs(np.real(FT_dqc))/sim_denom - np.abs(np.real(DQC_exp_interp))/exp_denom)**2 * weight_func
    
    sim_denom = np.max(np.max(np.abs(np.imag(FT_dqc))))
    if sim_denom == 0:
        sim_denom = 1
    exp_denom = np.max(np.max(np.abs(np.imag(DQC_exp_interp))))
    chisq_dqc_im =(np.abs(np.imag(FT_dqc))/sim_denom - np.abs(np.imag(DQC_exp_interp))/exp_denom)**2 * weight_func
    chisq_dqc = np.abs(chisq_dqc_re + 1j * chisq_dqc_im)

    chisq_rp = np.mean(np.mean(chisq_rp))
    chisq_nrp = np.mean(np.mean(chisq_nrp))
    chisq_dqc = np.mean(np.mean(chisq_dqc))
    
    
    chisq_tot = chisq_nrp + chisq_dqc + chisq_rp

    return chisq_tot 

#%
# plot data and sim and difference comparisons
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
            file_path = os.path.join('/Users/clairealbrecht/Dropbox/Claire_Dropbox/Data/6MI MNS/2D scan/',date_folder,scan_folder)
            date_str_len = len('20230101-120000')
            file_name_str = file_name #+ '_real_' + '_' + scan_folder[date_str_len:]
            fig.savefig(file_path+'/'+file_name_str+'.pdf')
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
            file_path = os.path.join('/Users/clairealbrecht/Dropbox/Claire_Dropbox/Data/6MI MNS/2D scan/',date_folder,scan_folder)
            date_str_len = len('20230101-120000')
            file_name_str = file_name #+ scan_folder[date_str_len:]
            fig.savefig(file_path+'/'+file_name_str+'.pdf')
            print('...saving plot as: '+file_name_str)
            print('in location: '+file_path)



#%% Plot one example with  sim2Dspec2

# with the nubar2omega stuff fixed
# laser_lam = 675
# laser_fwhm = 11.48 # 16 #12.45#15 #30 #15 #15 #6.37 #50 #30
# mu1_6MI = 3.98 #2.42
# mu2_6MI =3.98 #3.3
# Gam =99.23#99.97 #58.6 #2.6e-2
# sigI = 72.8 #50 #29.44 #48.3 #5e-4
# # delta = 72.8 #0 #197 #177 #400 #4e-2 #4.55e-2
# monoC_lam = 701 #700
# theta12 = 0 #30 #36.8
# # lam1 = 336 #336
# # lam2 = 346 #346
# omega_ge = 14572 #14600
# omega_gep = 15013#15020
# # delta = (omega_gep - omega_ge)/2

laser_lam = 680#677 #675#10**7/14700 #674 # 675
laser_fwhm = 33#30#8 #30 #15 #15 #6.37 #50 #30
mu1_6MI =3.8#4.49 #2.42
mu2_6MI =3 # 3#3.5 #3.3
Gam =110#85 #99.9971 #2.6e-2
sigI =105#55 #35.9397 # 70#48.3 #5e-4
# delta = 177 #400 #4e-2 #4.55e-2
monoC_lam = 701.9994 #700
theta12 = 0 #30 #36.8
# lam1 = 334 #336
# lam2 = 344 #346
# omega_ge = 10**7/(334*2)
# omega_gep = 10**7/(344*2)
omega_ge = 14550#14606.9616#10**7/(344*2)
omega_gep = 15100#15080#15005.2009#10**7/(332*2)


laser_lam = 677#677 #675#10**7/14700 #674 # 675
laser_fwhm = 33#30#8 #30 #15 #15 #6.37 #50 #30
mu1_6MI =3#4.49 #2.42
mu2_6MI =3 # 3#3.5 #3.3
Gam =100#85 #99.9971 #2.6e-2
sigI =100#55 #35.9397 # 70#48.3 #5e-4
# delta = 177 #400 #4e-2 #4.55e-2
monoC_lam = 701.9994 #700
theta12 = 0 #30 #36.8
# lam1 = 334 #336
# lam2 = 344 #346
# omega_ge = 10**7/(334*2)
# omega_gep = 10**7/(344*2)
omega_ge = 14650#14606.9616#10**7/(344*2)
omega_gep = 15000#15080#15005.2009#10**7/(332*2)

# print('omega_ge = '+str(10**7/omega_ge)+' nm')
# print('omega_gep = '+str(10**7/omega_gep)+' nm')
# print('omega_ge = '+str(omega_ge)+' cm^(-1)')
# print('omega_gep = '+str(omega_gep)+' cm^(-1)')
# print('omega_gf = '+str(2*omega_ge-delta)+' nm')
# print('omega_gfp = '+str(2*omega_gep-delta)+' nm')
# # opt_vals = [laser_lam, laser_fwhm, mu1_6MI, mu2_6MI, Gam, sigI, delta, monoC_lam, theta12, lam1, lam2]
# opt_vals = [laser_lam, laser_fwhm, mu1_6MI, mu2_6MI, Gam, sigI, delta, monoC_lam, theta12, omega_ge, omega_gep]
# opt_vals = [laser_lam, laser_fwhm, mu1_6MI, mu2_6MI, Gam, sigI, monoC_lam, theta12, omega_ge, omega_gep]
# opt_vals = [laser_lam, laser_fwhm, mu1_6MI, mu2_6MI, Gam, sigI, monoC_lam,omega_ge, omega_gep]

# opt_res_path = '/Users/clairealbrecht/Dropbox/Claire_Dropbox/Data/6MI MNS/2D scan/20221202/optimized_simulation'

# opt_res_path = '/Users/calbrecht/Dropbox/Claire_Dropbox/Data/6MI MNS/2D scan/20221202/optimized_simulation'
# opt_res_file = '2023-06-28_optimized_params.npy'
# opt_res_file = '2023-07-21_optimized_params.npy'

# opt_res_path = '/Users/clairealbrecht/Dropbox/Claire_Dropbox/Data_FPGA/MNS_4uM/2D_scans/'+date_folder+'/optimized_simulation'
# opt_res_path = os.path.join('/Users/clairealbrecht/Dropbox/Claire_Dropbox/Data_FPGA/MNS_4uM/2D_scans/',date_folder,'optimized_simulation')
# opt_res_file = '2023-08-31_optimized_params.npy'

# opt_vals = np.load(opt_res_path+'/'+opt_res_file)
# laser_lam, laser_fwhm, mu1_6MI, mu2_6MI, Gam, sigI, delta, monoC_lam, theta12, lam1, lam2 = opt_vals
# laser_lam, laser_fwhm, mu1_6MI, mu2_6MI, Gam, sigI, monoC_lam, theta12, omega_ge, omega_gep, scan_folder_nrprp, scan_folder_dqc = opt_vals

# =============================================================================
# opt vals to load for 20230729 with selection rules
# =============================================================================
# opt_res_path = '/Users/calbrecht/Dropbox/Claire_Dropbox/Data_FPGA/MNS_4uM/2D_scans/20230729/optimized_simulation_wSelecRules'
# opt_res_file = '2023-10-11_optimized_params.npy'
# opt_vals = np.load(opt_res_path+'/'+opt_res_file)
# laser_lam, laser_fwhm, mu1_6MI, mu2_6MI, Gam, sigI, delta, monoC_lam, theta12, omega_ge, omega_gep, scan_folder_nrprp, scan_folder_dqc = opt_vals
# opt_vals = np.double(opt_vals[:len(opt_vals)-2])
# laser_lam, laser_fwhm, mu1_6MI, mu2_6MI, Gam, sigI, delta, monoC_lam, theta12, omega_ge, omega_gep = opt_vals

#%

# laser_lam, laser_fwhm, mu1_6MI, mu2_6MI, Gam, sigI, delta, monoC_lam, theta12, lam1, lam2, scan_folder_nrprp, scan_folder_dqc = opt_vals
#%

scan_params = scan_folder_nrprp[len('20230101-120000-'):len(scan_folder)-len('_2DFPGA_FFT')-1]
stages = scan_params[len(scan_params)-2:]
scan_type = scan_params[:len(scan_params)-3]

global timing_mode
if stages == 'xz':
    timing_mode ='t32 = 0'
elif stages == 'yz':
    timing_mode ='t21 = 0'
elif stages == 'xy':
    timing_mode ='t43 = 0'


# if timing_mode == 't43 = 0':
#     scan_folder_dqc = '20221202-150919_DQC_xy'
#     scan_folder_nrprp = '20221202-154305_NRP_RP_xy'
# elif timing_mode == 't21 = 0':
#     scan_folder_nrprp = '20221202-163247_NRP_RP_yz'
#     scan_folder_dqc = '20221202-170516_DQC_yz'
        
global Ntimesteps
# Ntimesteps = 232 #30*2
Ntimesteps = int(np.max(np.abs(t21ax))/(t21ax_rt[1] - t21ax_rt[0]))
t21 = np.linspace(0,116.0805,num=Ntimesteps) # simulate at the retimed timesteps



# t21, t43, cm_DQC, cm_NRP, cm_RP, ax1, ax2, FT_dqc, FT_nrp, FT_rp  = sim2Dspec2(t21, laser_lam, laser_fwhm, mu1_6MI, mu2_6MI, Gam, sigI, delta, monoC_lam,theta12, lam1, lam2)#, plot_mode=1)
# t21, t43, cm_DQC, cm_NRP, cm_RP, ax1, ax2, FT_dqc, FT_nrp, FT_rp  = sim2Dspec2(t21, laser_lam, laser_fwhm, mu1_6MI, mu2_6MI, Gam, sigI, delta, monoC_lam, theta12, omega_ge, omega_gep)
# t21, t43, cm_DQC, cm_NRP, cm_RP, ax1, ax2, FT_dqc, FT_nrp, FT_rp  = sim2Dspec2(t21, laser_lam, laser_fwhm, mu1_6MI, mu2_6MI, Gam, sigI, monoC_lam, theta12, omega_ge, omega_gep)
# t21, t43, cm_DQC, cm_NRP, cm_RP, ax1, ax2, FT_dqc, FT_nrp, FT_rp  = sim2Dspec2(t21, laser_lam, laser_fwhm, mu1_6MI, mu2_6MI, Gam, sigI, monoC_lam, omega_ge, omega_gep)
t1_out, t2_out, cm_DQC, cm_NRP, cm_RP, ax1_dqc, ax2_dqc, ax1_nrprp, ax2_nrprp, FT_dqc, FT_nrp, FT_rp = sim2Dspec2(t21, laser_lam, laser_fwhm, mu1_6MI, mu2_6MI, Gam, sigI, monoC_lam, omega_ge, omega_gep)

#%
# ax1_28idx = np.where(np.abs(ax1-28) < (ax1[1]-ax1[0])/2)[0]
# ax1_30idx = np.where(np.abs(ax1-30) < (ax1[1]-ax1[0])/2)[0]
# print(ax1_28idx, ax1_30idx)
# xaxis_28idx = np.where(np.abs(xaxis-28) < (xaxis[1]-xaxis[0])/2)[0]
# xaxis_30idx = np.where(np.abs(xaxis-30) < (xaxis[1]-xaxis[0])/2)[0]
# print(xaxis_28idx, xaxis_30idx)

n_cont = 15
# ax_lim = [min(ax1), max(ax1)]
# ax_lim = [15, 19]
# ax_lim = [14,16]
# ax_lim = [13.75, 15.5]
ax_lim = [14, 15.5]
ax_lim = [13.75, 15.5]
# ax_lim = [13, 16]
# ax_lim = [13.6, 15.75]

# FT_dqc = np.fliplr(np.flipud(FT_dqc))
# DQC_exp = np.fliplr(np.flipud(DQC_exp))


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
# plot2Dspectra(t21, t43, cm_DQC, n_cont,ax_lim=[min(t21[0,:]), max(t21[0,:])], title=r'DQC($\tau$) with ($\tau_{32}$ = 0)', domain='time',save_mode = save_mode, file_name = save_name,scan_folder=scan_folder_dqc)
# save_name = 'sim_' + scan_folder_nrprp + '_tauNRP'
# plot2Dspectra(t21, t43, cm_NRP, n_cont,ax_lim=[min(t21[0,:]), max(t21[0,:])], title=r'NRP($\tau$) with ($\tau_{32}$ = 0)', domain='time',save_mode = save_mode, file_name = save_name,scan_folder = scan_folder_nrprp)
# save_name = 'sim_' + scan_folder_nrprp + '_tauRP'
# plot2Dspectra(t21, t43, cm_RP, n_cont,ax_lim=[min(t21[0,:]), max(t21[0,:])], title=r'RP($\tau$) with ($\tau_{32}$ = 0)', domain='time',save_mode = save_mode, file_name = save_name,scan_folder=scan_folder_nrprp)
save_name = 'sim_' + scan_folder_dqc + '_tauDQC'
plot2Dspectra(t1_out, t2_out, cm_DQC, n_cont,ax_lim=[min(t1_out[0,:]), max(t1_out[0,:])], title=r'DQC($\tau$) with '+timing_mode_str, domain='time',save_mode = save_mode, file_name = save_name,scan_folder=scan_folder_dqc)
save_name = 'sim_' + scan_folder_nrprp + '_tauNRP'
plot2Dspectra(t1_out, t2_out, cm_NRP, n_cont,ax_lim=[min(t1_out[0,:]), max(t1_out[0,:])], title=r'NRP($\tau$) with '+timing_mode_str, domain='time',save_mode = save_mode, file_name = save_name,scan_folder = scan_folder_nrprp)
save_name = 'sim_' + scan_folder_nrprp + '_tauRP'
plot2Dspectra(t1_out, t2_out, cm_RP, n_cont,ax_lim=[min(t1_out[0,:]), max(t1_out[0,:])], title=r'RP($\tau$) with '+timing_mode_str, domain='time',save_mode = save_mode, file_name = save_name,scan_folder=scan_folder_nrprp)

FT_dqc = FT_dqc/ np.max(np.max(FT_dqc))
FT_nrp = FT_nrp/np.max(np.max(FT_nrp))
FT_rp = FT_rp / np.max(np.max(FT_rp))

save_mode = save_mode
save_name = 'sim_' + scan_folder_dqc+'_FTdqc'
plot2Dspectra(ax1_dqc, ax2_dqc, FT_dqc, n_cont,ax_lim, title=r'DQC($\omega$) with '+timing_mode_str, domain='freq',save_mode = save_mode, file_name = save_name,scan_folder=scan_folder_dqc)
save_name = 'sim_' + scan_folder_nrprp+'_FTnrp'
plot2Dspectra(ax1_nrprp, ax2_nrprp, FT_nrp, n_cont,ax_lim, title=r'NRP($\omega$) with '+timing_mode_str, domain='freq',save_mode = save_mode, file_name = save_name,scan_folder=scan_folder_nrprp)
save_name = 'sim_' + scan_folder_nrprp+'_FTrp'
plot2Dspectra(ax1_nrprp, ax2_nrprp, FT_rp, n_cont,ax_lim, title=r'RP($\omega$) with '+timing_mode_str, domain='freq',save_mode = save_mode, file_name = save_name,scan_folder=scan_folder_nrprp)
#%
# ax_lim = [28,30]

# if timing_mode == 't32 = 0':
weight_func_mode = 0 #1
save_mode = save_mode
save_name = 'sim_' + scan_folder_dqc +'_FTdqcReComp'
plot_comparer(ax1_dqc, ax2_dqc, DQC_exp, FT_dqc, 'DQC',figsize=(16,4),ax_lim = ax_lim ,save_mode = save_mode, file_name = save_name, scan_folder = scan_folder_dqc,weight_func_mode=weight_func_mode)
save_name = 'sim_' + scan_folder_nrprp +'_FTnrpReComp'
plot_comparer(ax1_nrprp,ax2_nrprp, NRP_exp, FT_nrp, 'NRP',figsize=(16,4),ax_lim = ax_lim,save_mode = save_mode, file_name = save_name, scan_folder = scan_folder_nrprp,weight_func_mode=weight_func_mode)
save_name = 'sim_' + scan_folder_nrprp +'_FTrpReComp'
plot_comparer(ax1_nrprp,ax2_nrprp, RP_exp, FT_rp, 'RP',figsize=(16,4),ax_lim = ax_lim,save_mode = save_mode, file_name = save_name,scan_folder = scan_folder_nrprp,weight_func_mode=weight_func_mode)
    
    #%
    # save_mode = save_mode
    # save_name = 'sim_' + scan_folder_dqc +'_FTdqcImComp'
    # plot_comparer(ax1, np.conjugate(DQC_exp), FT_dqc, 'DQC',compare_mode = 'imag',figsize=(16,4),ax_lim = ax_lim,save_mode = save_mode, file_name = save_name,scan_folder = scan_folder_dqc,weight_func_mode=weight_func_mode)
    # save_name = 'sim_' + scan_folder_nrprp +'_FTnrpImComp'
    # plot_comparer(ax1, np.conjugate(NRP_exp), FT_nrp, 'NRP',compare_mode = 'imag',figsize=(16,4),ax_lim = ax_lim,save_mode = save_mode, file_name = save_name,scan_folder = scan_folder_nrprp,weight_func_mode=weight_func_mode)
    # save_name = 'sim_' + scan_folder_nrprp +'_FTrpImComp'
    # plot_comparer(ax1, RP_exp, FT_rp, 'RP',compare_mode = 'imag',figsize=(16,4),ax_lim = ax_lim,save_mode = save_mode, file_name = save_name,scan_folder = scan_folder_nrprp,weight_func_mode=weight_func_mode)


#%%
# ax_lim = [28,30]
# sim_idx_28to30 = np.array([459, 642])
# ax1_28to30 = ax1[sim_idx_28to30[0]:sim_idx_28to30[1]]
# plot_comparer(ax1_28to30, DQC_exp_interp, FT_dqc[sim_idx_28to30[0]:sim_idx_28to30[1],sim_idx_28to30[0]:sim_idx_28to30[1]], 'DQC',figsize=(16,4))
# plot_comparer(ax1_28to30, NRP_exp_interp, FT_nrp[sim_idx_28to30[0]:sim_idx_28to30[1],sim_idx_28to30[0]:sim_idx_28to30[1]], 'NRP',figsize=(16,4))
# plot_comparer(ax1_28to30, RP_exp_interp, FT_rp[sim_idx_28to30[0]:sim_idx_28to30[1],sim_idx_28to30[0]:sim_idx_28to30[1]], 'RP',figsize=(16,4))

# plot_comparer(ax1_28to30, DQC_exp_interp, FT_dqc[sim_idx_28to30[0]:sim_idx_28to30[1],sim_idx_28to30[0]:sim_idx_28to30[1]], 'DQC',compare_mode = 'imag',figsize=(16,4))
# plot_comparer(ax1_28to30, NRP_exp_interp, FT_nrp[sim_idx_28to30[0]:sim_idx_28to30[1],sim_idx_28to30[0]:sim_idx_28to30[1]], 'NRP',compare_mode = 'imag',figsize=(16,4))
# plot_comparer(ax1_28to30, RP_exp_interp, FT_rp[sim_idx_28to30[0]:sim_idx_28to30[1],sim_idx_28to30[0]:sim_idx_28to30[1]], 'RP',compare_mode = 'imag',figsize=(16,4))


#%%

t21 = np.linspace(0,116.0805,num=Ntimesteps) 


# # lam1_bounds = [337,340] # 450, 420
# # lam2_bounds = [334,337] 
# laser_lam_bounds = [673, 680] #[650, 700]
# laser_fwhm_bounds = [25,30]#[5,30]#[12, 17] #[3, 30]
# mu1_6MI_bounds = [2,5]
# mu2_6MI_bounds = [2,5]
# Gam_bounds = [30, 1e2] #[1e-3,3e-2] #[1e-2,5e-1]
# sigI_bounds = [30, 1e2] #[0.5e-3,3e-2] #[2e-2,5e-1]
# # delta_bounds =  [180, 250]#[10, 1000] #[100,500] #[0.02, 50] #[-0.05 ,-0.02 ]
# monoC_lam_bounds = [698, 702]
# theta12_bounds = [0, 0] #90] #[30, 40]
# # lam1_bounds = [330, 370]
# # lam2_bounds = [330, 370]
# omega_ge_bounds = [14000,14800]
# omega_gep_bounds = [14800,15600]

# MNS bounds params
laser_lam_bounds = [674.5, 675.5] #[650, 700]
laser_fwhm_bounds = [29,30]#[5,30]#[12, 17] #[3, 30]
mu1_6MI_bounds = [3,4]
mu2_6MI_bounds = [3,4]
Gam_bounds = [30, 1e2] #[1e-3,3e-2] #[1e-2,5e-1]
sigI_bounds = [30, 1e2] #[0.5e-3,3e-2] #[2e-2,5e-1]
monoC_lam_bounds = [699.5, 700.5]
# theta12_bounds = [0, 0] #90] #[30, 40]
omega_ge_bounds = [14000,14800]
omega_gep_bounds = [14800,15600]

#MNT bounds
laser_lam_bounds = [674.5, 678.5] #[650, 700]
laser_fwhm_bounds = [29,32]#[5,30]#[12, 17] #[3, 30]
mu1_6MI_bounds = [3,4]
mu2_6MI_bounds = [3,4]
Gam_bounds = [70, 5e2] #[1e-3,3e-2] #[1e-2,5e-1]
sigI_bounds = [70, 5e2] #[0.5e-3,3e-2] #[2e-2,5e-1]
monoC_lam_bounds = [699.5, 700.5]
# theta12_bounds = [0, 0] #90] #[30, 40]
omega_ge_bounds = [14000,14800]
omega_gep_bounds = [14800,15600]

# bounds = np.vstack([laser_lam_bounds, laser_fwhm_bounds, mu1_6MI_bounds, mu2_6MI_bounds, Gam_bounds, sigI_bounds, delta_bounds])
# bounds = np.vstack([laser_lam_bounds, laser_fwhm_bounds, mu1_6MI_bounds, mu2_6MI_bounds, Gam_bounds, sigI_bounds, delta_bounds,monoC_lam_bounds, theta12_bounds])
# bounds = np.vstack([laser_lam_bounds, laser_fwhm_bounds, mu1_6MI_bounds, mu2_6MI_bounds, Gam_bounds, sigI_bounds, delta_bounds,monoC_lam_bounds, theta12_bounds, lam1_bounds, lam2_bounds])
# bounds = np.vstack([laser_lam_bounds, laser_fwhm_bounds, mu1_6MI_bounds, mu2_6MI_bounds, Gam_bounds, sigI_bounds, delta_bounds,monoC_lam_bounds, theta12_bounds, omega_ge_bounds, omega_gep_bounds])
# bounds = np.vstack([laser_lam_bounds, laser_fwhm_bounds, mu1_6MI_bounds, mu2_6MI_bounds, Gam_bounds, sigI_bounds,monoC_lam_bounds, theta12_bounds, omega_ge_bounds, omega_gep_bounds])
bounds = np.vstack([laser_lam_bounds, laser_fwhm_bounds, mu1_6MI_bounds, mu2_6MI_bounds, Gam_bounds, sigI_bounds,monoC_lam_bounds, omega_ge_bounds, omega_gep_bounds])

# =============================================================================
# # # set a starting point
# =============================================================================
# # MNS starting point
# laser_lam = 675#677 #675#10**7/14700 #674 # 675
# laser_fwhm = 30#25#30#8 #30 #15 #15 #6.37 #50 #30
# mu1_6MI =3.5#4.49 #2.42
# mu2_6MI =3# 3#3.5 #3.3
# Gam =80#99.9971 #2.6e-2
# sigI =55#35.9397 # 70#48.3 #5e-4
# # delta = 177 #400 #4e-2 #4.55e-2
# monoC_lam = 700#701.9994 #700
# # theta12 = 0 #30 #36.8
# # lam1 = 334 #336
# # lam2 = 344 #346
# # omega_ge = 10**7/(334*2)
# # omega_gep = 10**7/(344*2)
# omega_ge = 14606.9616#10**7/(344*2)
# omega_gep = 15080#15005.2009#10**7/(332*2)

# MNT starting point
laser_lam = 677#677 #675#10**7/14700 #674 # 675
laser_fwhm = 31#25#30#8 #30 #15 #15 #6.37 #50 #30
mu1_6MI =3.8 #5#4.49 #2.42
mu2_6MI =3# 3#3.5 #3.3
Gam =110 #80#99.9971 #2.6e-2
sigI =105 #55#35.9397 # 70#48.3 #5e-4
# delta = 177 #400 #4e-2 #4.55e-2
monoC_lam = 700#701.9994 #700
# theta12 = 0 #30 #36.8
# lam1 = 334 #336
# lam2 = 344 #346
# omega_ge = 10**7/(334*2)
# omega_gep = 10**7/(344*2)
omega_ge = 14550#10**7/(344*2)
omega_gep = 15100#15005.2009#10**7/(332*2)



# x0 = np.array([laser_lam, laser_fwhm, mu1_6MI, mu2_6MI, Gam, sigI, delta])
# x0 = np.array([laser_lam, laser_fwhm, mu1_6MI, mu2_6MI, Gam, sigI, delta, monoC_lam, theta12])
# x0 = np.array([laser_lam, laser_fwhm, mu1_6MI, mu2_6MI, Gam, sigI, delta, monoC_lam, theta12, lam1, lam2])
# x0 = np.array([laser_lam, laser_fwhm, mu1_6MI, mu2_6MI, Gam, sigI, delta, monoC_lam, theta12, omega_ge, omega_gep])
# x0 = np.array([laser_lam, laser_fwhm, mu1_6MI, mu2_6MI, Gam, sigI, monoC_lam, theta12, omega_ge, omega_gep])
x0 = np.array([laser_lam, laser_fwhm, mu1_6MI, mu2_6MI, Gam, sigI, monoC_lam, omega_ge, omega_gep])


# =============================================================================
# use the simple minimize()
# =============================================================================
# res = opt.minimize(chisq_calc_int2, x0, #args = (lam1, lam2, mu1_6MI, Gam, sigI, delta), 
#                         bounds=bounds, 
#                         method='SLSQP', 
#                         # tol=1e-15,
#                         callback=None, 
#                         options={'maxiter': 1000, 
#                                   # 'ftol': 1e-15, 
#                                   'xtol':1e-10,
#                                   'iprint': 2, 
#                                   'disp': True,
#                                   'acc' : 1e-15, 
#                                   'eps': 1.4901161193847656e-06}) 
# # laser_lam, laser_fwhm, mu1_6MI, mu2_6MI, Gam, sigI, delta = res.x
# laser_lam, laser_fwhm, mu1_6MI, mu2_6MI, Gam, sigI, delta, monoC_lam, theta12, lam1, lam2 = res.x


# =============================================================================
# use the more complicated differential_evolution() ... similar to genetic algorithm
# =============================================================================
t21 = np.linspace(0,116.0805,num=Ntimesteps) 
if __name__ == '__main__':
    res = opt.differential_evolution(func=chisq_calc_int2, 
                                      bounds=bounds,
                                      x0=x0,
                                      disp=True,
                                      workers=1,
                                      maxiter=1000,
                                      polish=True,
                                      # atol=1e-8, #1e-6, 1e-10,
                                      # tol = 1e-8, #1e-6, 10,
                                      # mutation=(0,1.9),
                                      # popsize=30,
                                      # updating='immediate',
                                      strategy = 'best1exp') 
                                    # 'best1exp' # this one did a decent job
                                    # 'rand1exp' didn't finish in 500 iter
                                    # 'randtobest1exp' did fine... copied res and plots into onenote
    # lam1, lam2, mu1_6MI, mu2_6MI, Gam, sigI, delta = res.x
    # laser_lam, laser_fwhm, mu1_6MI, mu2_6MI, Gam, sigI, delta, monoC_lam, theta12 = res.x
    # laser_lam, laser_fwhm, mu1_6MI, mu2_6MI, Gam, sigI, delta, monoC_lam, theta12, omega_ge, omega_gep = res.x
    # laser_lam, laser_fwhm, mu1_6MI, mu2_6MI, Gam, sigI, monoC_lam, theta12, omega_ge, omega_gep = res.x
    laser_lam, laser_fwhm, mu1_6MI, mu2_6MI, Gam, sigI, monoC_lam, omega_ge, omega_gep = res.x
    # t21 = t21.flatten()

# param_labels = ['laser wavelength', 'laser fwhm', '|mu1|', '|mu2|','Gam_H','sigma_I','delta','monoc wavelength','theta12', 'lam1', 'lam2']
# param_unit_labels = [' nm', ' nm', ' D', ' D',' cm^(-1)',' cm^(-1)',' cm^(-1)',' nm',' deg', ' nm', ' nm']
# param_labels = ['laser wavelength', 'laser fwhm', '|mu1|', '|mu2|','Gam_H','sigma_I','delta','monoc wavelength','theta12', 'omega_ge', 'omega_gep']
# param_unit_labels = [' nm', ' nm', ' D', ' D',' cm^(-1)',' cm^(-1)',' cm^(-1)',' nm',' deg', ' cm^(-1)', ' cm^(-1)']
# param_labels = ['laser wavelength', 'laser fwhm', '|mu1|', '|mu2|','Gam_H','sigma_I','monoc wavelength','theta12', 'omega_ge', 'omega_gep']
# param_unit_labels = [' nm', ' nm', ' D', ' D',' cm^(-1)',' cm^(-1)',' nm',' deg', ' cm^(-1)', ' cm^(-1)']
param_labels = ['laser wavelength', 'laser fwhm', '|mu1|', '|mu2|','Gam_H','sigma_I','monoc wavelength', 'omega_ge', 'omega_gep']
param_unit_labels = [' nm', ' nm', ' D', ' D',' cm^(-1)',' cm^(-1)',' nm', ' cm^(-1)', ' cm^(-1)']

t21 = np.linspace(0,116.0805,num=Ntimesteps) 
# t21, t43, cm_DQC, cm_NRP, cm_RP, ax1, ax2, FT_dqc, FT_nrp, FT_rp  = sim2Dspec2(t21, laser_lam, laser_fwhm, mu1_6MI, mu2_6MI, Gam, sigI, delta, monoC_lam, theta12)#, plot_mode=1)
# t21, t43, cm_DQC, cm_NRP, cm_RP, ax1, ax2, FT_dqc, FT_nrp, FT_rp  = sim2Dspec2(t21, laser_lam, laser_fwhm, mu1_6MI, mu2_6MI, Gam, sigI, delta, monoC_lam, theta12, lam1, lam2)#, plot_mode=1)
# t21, t43, cm_DQC, cm_NRP, cm_RP, ax1, ax2, FT_dqc, FT_nrp, FT_rp  = sim2Dspec2(t21, laser_lam, laser_fwhm, mu1_6MI, mu2_6MI, Gam, sigI, delta, monoC_lam, theta12, omega_ge, omega_gep)#, plot_mode=1)
# t21, t43, cm_DQC, cm_NRP, cm_RP, ax1, ax2, FT_dqc, FT_nrp, FT_rp  = sim2Dspec2(t21, laser_lam, laser_fwhm, mu1_6MI, mu2_6MI, Gam, sigI, monoC_lam, theta12, omega_ge, omega_gep)#, plot_mode=1)
t21, t43, cm_DQC, cm_NRP, cm_RP, ax1, ax2, FT_dqc, FT_nrp, FT_rp  = sim2Dspec2(t21, laser_lam, laser_fwhm, mu1_6MI, mu2_6MI, Gam, sigI, monoC_lam, omega_ge, omega_gep)#, plot_mode=1)

n_cont = 15
# ax_lim = [min(ax1), max(ax1)]
# ax_lim = [26, 29]
# ax_lim = [20,40]
# ax_lim = [28,30]
# ax_lim = [13.75, 15.5]
ax_lim = [14, 15.5]

#%
plot2Dspectra(t21, t43, cm_DQC, n_cont,ax_lim=[min(t21[0,:]), max(t21[0,:])], title=r'DQC($\tau$) with ($\tau_{32}$ = 0)', domain='time')
plot2Dspectra(t21, t43, cm_NRP, n_cont,ax_lim=[min(t21[0,:]), max(t21[0,:])], title=r'NRP($\tau$) with ($\tau_{32}$ = 0)', domain='time')
plot2Dspectra(t21, t43, cm_RP, n_cont,ax_lim=[min(t21[0,:]), max(t21[0,:])], title=r'RP($\tau$) with ($\tau_{32}$ = 0)', domain='time')

# ax_lim = [2.5, 5]
plot2Dspectra(ax1, ax2, FT_dqc, n_cont,ax_lim, title=r'DQC($\omega$) with ($\tau_{32}$ = 0)', domain='freq')
plot2Dspectra(ax1, ax2, FT_nrp, n_cont,ax_lim, title=r'NRP($\omega$) with ($\tau_{32}$ = 0)', domain='freq')
plot2Dspectra(ax1, ax2, FT_rp, n_cont,ax_lim, title=r'RP($\omega$) with ($\tau_{32}$ = 0)', domain='freq')


plot_comparer(ax1, DQC_exp, FT_dqc, 'DQC',figsize=(16,4), compare_mode = 'real',ax_lim = ax_lim)
plot_comparer(ax1, NRP_exp, FT_nrp, 'NRP',figsize=(16,4), compare_mode = 'real',ax_lim = ax_lim)
plot_comparer(ax1, RP_exp, FT_rp, 'RP',figsize=(16,4),  compare_mode = 'real',ax_lim = ax_lim)


# plot_comparer(ax1, DQC_exp, FT_dqc, 'DQC',compare_mode = 'imag',figsize=(16,4), ax_lim = ax_lim)
# plot_comparer(ax1, NRP_exp, FT_nrp, 'NRP',compare_mode = 'imag',figsize=(16,4), ax_lim = ax_lim)
# plot_comparer(ax1, RP_exp, FT_rp, 'RP',compare_mode = 'imag',figsize=(16,4), ax_lim = ax_lim)

plt.show()

# opt_vals = laser_lam, laser_fwhm, mu1_6MI, mu2_6MI, Gam, sigI, monoC_lam, theta12, omega_ge, omega_gep, scan_folder_nrprp, scan_folder_dqc 
opt_vals = laser_lam, laser_fwhm, mu1_6MI, mu2_6MI, Gam, sigI, monoC_lam, omega_ge, omega_gep, scan_folder_nrprp, scan_folder_dqc 

from datetime import time, datetime
save_opt_mode = 1

now = datetime.now()
# date_time = now.strftime("%m%d%Y_%H%M%S")
date_time = now.strftime("%Y%m%d_%H%M%S")
if save_opt_mode == 1:
    # file_path = os.path.join('/Users/clairealbrecht/Dropbox/Claire_Dropbox/Data/6MI MNS/2D scan/',date_folder)
    # file_path = os.path.join('/Users/clairealbrecht/Dropbox/Claire_Dropbox/Data_FPGA/MNS_4uM/2D_scans/',date_folder)
    
    # file_path = os.path.join('/Users/calbrec2/Dropbox/Claire_Dropbox/Data_FPGA/MNS_4uM/2D_scans/',date_folder) # used prior to 20231116
    
    scan_params = scan_folder_nrprp[len('20230101-120000-'):len(scan_folder)-len('_2DFPGA_FFT')-1]
    stages = scan_params[len(scan_params)-2:]
    scan_type = scan_params[:len(scan_params)-3]
    file_path = '/Users/calbrec2/Dropbox/Claire_Dropbox/Data_FPGA/'+ sample_name+ '/2D_scans/' + date_folder #+ '/' + scan_folder

    # opt_vals = laser_lam, laser_fwhm, mu1_6MI, mu2_6MI, Gam, sigI, delta, monoC_lam, theta12, lam1, lam2, scan_folder_nrprp, scan_folder_dqc 
    # opt_vals = laser_lam, laser_fwhm, mu1_6MI, mu2_6MI, Gam, sigI, delta, monoC_lam, theta12, omega_ge, omega_gep, scan_folder_nrprp, scan_folder_dqc 
    # opt_vals = laser_lam, laser_fwhm, mu1_6MI, mu2_6MI, Gam, sigI, monoC_lam, theta12, omega_ge, omega_gep, scan_folder_nrprp, scan_folder_dqc 
    opt_vals = laser_lam, laser_fwhm, mu1_6MI, mu2_6MI, Gam, sigI, monoC_lam, omega_ge, omega_gep, scan_folder_nrprp, scan_folder_dqc 
    # folderName = 'optimized_simulation'
    # folderName = 'optimized_simulation_wSelecRules'    
    # folderName = 'optimized_simulation_wSelecRules_wAlphaFix'
    folderName = date_folder+'_'+stages+'_'+'optimized_simulation_wSelecRules_wAlphaFix'
    if os.path.isdir(os.path.join(file_path,folderName)):
        # np.save(os.path.join(file_path,'optimized_simulation_wSelecRules')+'/'+str(date.today())+'_optimized_params', opt_vals)
        np.save(os.path.join(file_path+'/'+folderName+'/'+date_time+'_optimized_params'), opt_vals)
        print('...saving in: '+os.path.join(file_path+'/'+folderName))
        print('    as: ' + date_time+'_optimized_params')
    else:
        os.makedirs(os.path.join(file_path+'/'+folderName))
        print('...making folder: '+os.path.join(file_path+'/'+folderName))
        # np.save(os.path.join(file_path,'optimized_simulation_wSelecRules')+'/'+str(date.today())+'_optimized_params', opt_vals)
        np.save(os.path.join(file_path+'/'+folderName+'/'+date_time+'_optimized_params'), opt_vals)

    # if os.path.isdir(os.path.join(file_path,'optimized_simulation_wSelecRules_wAlphaFix')):
    #     # np.save(os.path.join(file_path,'optimized_simulation_wSelecRules')+'/'+str(date.today())+'_optimized_params', opt_vals)
    #     np.save(os.path.join(file_path,'optimized_simulation_wSelecRules_wAlphaFix'+'/'+date_time+'_optimized_params', opt_vals)
    # else:
    #     os.makedirs(os.path.join(file_path,'optimized_simulation_wSelecRules_wAlphaFix'))
    #     print('...making folder: '+os.path.join(file_path,'optimized_simulation_wSelecRules_wAlphaFix'))
    #     # np.save(os.path.join(file_path,'optimized_simulation_wSelecRules')+'/'+str(date.today())+'_optimized_params', opt_vals)
    #     np.save(os.path.join(file_path,'optimized_simulation_wSelecRules_wAlphaFix')+'/'+date_time+'_optimized_params', opt_vals)

#%

# vals = res.x
vals = opt_vals[:len(opt_vals)-2]#.astype(float)
# vals = opt_vals
print('  ')
print('**** Optimized parameters: ****')
print('  ')
for i in range(len(vals)):
    print(param_labels[i]+':'+' '*(19-len(param_labels[i]))+str(np.round(vals[i],4))+' '*(10-len(str(np.round(vals[i],4))))+param_unit_labels[i])
print('  ')
# print('2*omega_ge-delta'+':'+' '*(19-len('2*omega_ge-delta'))+str(np.round((2*omega_ge)-delta,4))+ ' cm^(-1)')
# print('2*omega_gep-delta'+':'+' '*(19-len('2*omega_gep-delta'))+str(np.round((2*omega_gep)-delta,4))+ ' cm^(-1)')
# print('|omega_f2 - omega_f0|'+':'+' '*(17-len('omega_f2 - omega_f0'))+ str(np.abs(np.round((2*omega_gep)-delta - (2*omega_ge)-delta,4)))+ ' '*(11-len(str(np.round((2*omega_gep)-delta - (2*omega_ge)-delta,4))))+' cm^(-1)')


delta = (omega_gep - omega_ge)/2 # 20231018 update
omega_ef = omega_ge - delta
omega_epfp = omega_gep - delta
omega_gf = omega_ge + omega_ef
omega_gfp = omega_gep + omega_epfp
omega_efp = omega_gfp - omega_ge
omega_epf = omega_gf - omega_gep
omega_ee = omega_epep = 0
omega_eep = omega_gep - omega_ge # does this equal 2 * delta?

print('  ')
print('**** Calculated parameters: ****')
print('  ')
omega_labels = ['omega0/2','omega_ge','omega_gep','omega_ef','omega_epfp','omega_efp','omega_epf','omega_gf','omega_gfp','omega_ee','omega_eep','omega_gfp - omega_gf']
omegas = [delta, omega_ge, omega_gep, omega_ef, omega_epfp, omega_efp, omega_epf, omega_gf, omega_gfp, omega_ee, omega_eep,omega_gfp - omega_gf]
for i in range(len(omegas)):
    print(omega_labels[i]+':'+' '*(19-len(omega_labels[i]))+str(np.round(omegas[i],4))+' '*(10-len(str(np.round(omegas[i],4))))+' cm^(-1)')
print('  ')

#%
