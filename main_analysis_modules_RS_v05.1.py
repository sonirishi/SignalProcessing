# -*- coding: utf-8 -*-
"""
Created on Wed Jun 15 12:12:37 2016

@author: rsoni106
"""

import pandas as pd
import numpy as np
import scipy.fftpack as sig
from scipy.signal import kaiserord, lfilter, firwin, hanning, filtfilt
import scipy.stats.stats as st
from collections import defaultdict
import pywt
from pykalman import KalmanFilter
from scipy.signal import hilbert
from dtw import dtw
from python_speech_features import mfcc
from sklearn.linear_model import LinearRegression
from scipy.spatial.distance import euclidean
from sklearn.decomposition import FactorAnalysis

#  4-8, 8-13, 13-30, and 30-100 Hz

""" Overlapping windowed FFT applied with hanning function """
""" Normalize the FFT output with window length """
    
def fft_func(data_1,window_length,window_type,overlap_fac):
    wind_func = window_type(window_length)
    fft_split = defaultdict(list)
    data_points = data_1.shape[1]
    for i in range(data_1.shape[0]):
        j = 0
        iter_count = 0
        while j < data_points:
            fft_split[str(i)+"_"+str(iter_count)] = np.fft.rfft(((data_1[i][j:j+window_length])*wind_func))
            j += int(window_length*overlap_fac) if (j+window_length) < data_points else data_points
            iter_count += 1
    return fft_split, iter_count
        
def pow_spectrum(fft_split,data_points,window_length,window_type,orig_freq,channelcount,iter_count):
    power_spectrum = defaultdict(list)
    power_spectrum_density = defaultdict(list)    
    avg_power_spectrum = np.zeros((channelcount,fft_split["0_0"].shape[0]))
    avg_power_spectrum_density = np.zeros((channelcount,fft_split["0_0"].shape[0]))
    wind_func = window_type(window_length)
    band1 = np.zeros(channelcount)
    band2 = np.zeros(channelcount)
    band3 = np.zeros(channelcount)
    band1_density = np.zeros(channelcount)
    band2_density = np.zeros(channelcount)
    band3_density = np.zeros(channelcount)
    S1 = wind_func.sum()
    S2 = (wind_func**2).sum()
    for i in fft_split.keys():
        power_spectrum[i] = 2*((np.abs(fft_split[i]))**2)/S1**2
        power_spectrum_density[i] = 2*((np.abs(fft_split[i]))**2)/(orig_freq*S2)
    for i in range(channelcount):
        for j in range(iter_count):
            avg_power_spectrum[i,] += power_spectrum[str(i) + "_" + str(j)]
            avg_power_spectrum_density[i,] += power_spectrum_density[str(i) + "_" + str(j)]
    avg_power_spectrum = avg_power_spectrum/iter_count
    avg_power_spectrum_density = avg_power_spectrum_density/iter_count
    for i in range(channelcount):
        """ Create 3 bands of energy"""
        band1[i] = np.sum(avg_power_spectrum[i,0:3])
        band2[i] = np.sum(avg_power_spectrum[i,4:6])
        band3[i] = np.sum(avg_power_spectrum[i,7:])
        """ Create 3 bands of energy density"""
        band1_density[i] = np.sum(avg_power_spectrum_density[i,0:3])
        band2_density[i] = np.sum(avg_power_spectrum_density[i,4:6])
        band3_density[i] = np.sum(avg_power_spectrum_density[i,7:])
    return band1,band2,band3,band1_density,band2_density,band3_density
        
def normalize(data_1):
    for i in range(data_1.shape[0]):
        data_1[i] = data_1[i] - np.mean(data_1[i])
    return data_1
    
def base_stats(data_1):
    stats_dict = np.zeros((data_1.shape[0],4))
    for i in range(data_1.shape[0]):
        stats_dict[i,0] = st.skew(data_1[i],bias=False)
        stats_dict[i,1] = st.kurtosis(data_1[i],bias=False)
        stats_dict[i,2] = np.max(data_1[i])
        stats_dict[i,3] = np.std(data_1[i])
    return stats_dict

def correlation_vals(data_1):
    """ Keep eigenvalues of the correlation matrix """
    eigen_corr,t = np.linalg.eig(np.corrcoef(data_1))
    eigen_corr.sort(); eigen_corr = eigen_corr[::-1]
    return eigen_corr
    
""" ripple defines ripple in passband and stopband
    Ripple DB = 20*log(Ao/Ai), we would want low ripple DB, 60"""    
"""https://haseebsohail.files.wordpress.com/2013/02/discrete-time-signal-processing-by-alan-v-oppenheim.pdf
    Ripple value = 60 page 475-476"""
""" threshold frequency = cut-off = 100, tran_width = 0.2*nyq"""    

def lowpassfilter(data_1,freq_signal,cutoff_hz,ripple_db,transition_width):
    nyquist_freq = freq_signal/2
    width = transition_width/nyquist_freq
    N, beta = kaiserord(ripple_db, width)
    taps = firwin(N, cutoff_hz/nyquist_freq, window=('kaiser', beta))
    filter_data = lfilter(taps, 1.0, data_1) 
    return filter_data

def lpf_fb(data_1,freq_signal,cutoff_hz,ripple_db,transition_width):
    nyquist_freq = freq_signal/2
    width = transition_width/nyquist_freq
    N, beta = kaiserord(ripple_db, width)
    taps = firwin(N, cutoff_hz/nyquist_freq, window=('kaiser', beta))
    filter_data = filtfilt(taps, 1.0, data_1) 
    return filter_data

""" Derivative calculated using the gradient function in numpy """

def petrosian_fractal(data_1):
    N = len(data_1[1])
    pfd = np.zeros(data_1.shape[0])
    cnt_sign_change = np.zeros(data_1.shape[0])
    for i in range(data_1.shape[0]):
        gradient_val = np.gradient(data_1[i])
        cnt_sign_change[i] = ((np.roll(np.sign(gradient_val),1) - np.sign(gradient_val)) !=0).astype(int).sum()
        pfd[i] = np.log10(N)/(np.log10(N) + np.log10(N/(N+0.4*cnt_sign_change[i])))
    return pfd

""" K is similar to embedding dimension, usually taken as 5"""

def higuchi_fractal(data_1,k):
    N = len(data_1[1])
    higuchi = np.zeros((data_1.shape[0],1))
    for z in range(data_1.shape[0]):
        df1 = np.zeros((k,1))
        df2 = np.zeros((k,1))
        for i in range(1,k+1):
            TL = np.zeros(i)
            for m in range(1,i+1):
                frac = int((N-m)/i)
                fnorm = (N-1)/(frac*i)
                TL[m-1] = np.sum(np.abs(data_1[z][np.arange(m+i,m+frac*i,i)] - data_1[z][np.arange(m,m+(frac-1)*i,i)]))*fnorm
            df2[i-1] = np.log(np.mean(TL)+1e-12)
            df1[i-1] = np.log(1/i)
        lm = LinearRegression()
        lm.fit(df1,df2)
        higuchi[z] = lm.coef_
    return higuchi
            
""" Tested with a random sequence, value of 0.48 achieved so seems fine """           
            
def hurst_exp(data_1,freq):
    hurst = np.zeros(data_1.shape[0])
    t = pd.DataFrame(np.log(np.arange(1,freq+1,1)))
    t = np.nan_to_num(t)
    for i in range(data_1.shape[0]):
        data_2 = []
        cum_series = []
        Rval=[]
        Sval=[]
        mean = np.mean(data_1[i])
        data_2 = data_1[i] - mean
        cum_series = np.cumsum(np.array(data_2))
        Rval = np.maximum.accumulate(np.array(cum_series)) - np.minimum.accumulate(np.array(cum_series))
        Sval = (pd.DataFrame(data_1[i])).expanding().std()
        Sval = np.ravel(np.nan_to_num(Sval))
        R_Sval = np.zeros((len(Rval),1))
        R_Sval = np.log(np.divide(Rval,Sval+1e-12)+1e-12)
        """ Using linear regression for power law
            log of the variables used -------"""
        lm = LinearRegression()
        lm.fit(t,R_Sval)
        hurst[i] = lm.coef_
    return hurst

def shannon_entropy(fft,channelcount,iter_count):
    entropy = defaultdict(list)
    power = defaultdict(list)
    entropy_avg = np.zeros(channelcount)
    for j in range(channelcount):
        for i in range(iter_count):
            power[str(j)+"_"+str(i)] = np.abs(fft[str(j)+"_"+str(i)])/np.sum(np.abs(fft[str(j)+"_"+str(i)]))
        for i in range(iter_count):
            entropy[str(j)+"_"+str(i)] = -np.sum(np.multiply(power[str(j)+"_"+str(i)],np.log2(power[str(j)+"_"+str(i)])))
        for i in range(iter_count):
            entropy_avg[j] += entropy[str(j)+"_"+str(i)]
        entropy_avg[j] = entropy_avg[j]/iter_count
    return entropy_avg
    
def renyi_entropy(fft,channelcount,iter_count):
    rentropy = defaultdict(list)
    rpower = defaultdict(list)
    rentropy_avg = np.zeros(channelcount)
    for j in range(channelcount):
        """ Normalizing the power component"""
        for i in range(iter_count):
            rpower[str(j)+"_"+str(i)] = np.abs(fft[str(j)+"_"+str(i)])/np.sum(np.abs(fft[str(j)+"_"+str(i)]))
        """ Entropy calculation """
        for i in range(iter_count):
            rentropy[str(j)+"_"+str(i)] = -np.log2(np.sum(np.multiply(rpower[str(j)+"_"+str(i)],rpower[str(j)+"_"+str(i)])))
        for i in range(iter_count):
            rentropy_avg[j] += rentropy[str(j)+"_"+str(i)]
        rentropy_avg[j] = rentropy_avg[j]/iter_count
    return rentropy_avg
    
def approx_entropy(data_1,embed_dim):
    corr_dim_fin = np.zeros(data_1.shape[0])
    corr_dim_fin2 = np.zeros(data_1.shape[0])
    approx_entropy = np.zeros(data_1.shape[0])
    for i in range(data_1.shape[0]):
        r = 0.2*np.std(data_1[i])
        Xt= np.zeros((data_1.shape[1]+1-embed_dim,embed_dim))
        Yt = np.zeros((data_1.shape[1]-embed_dim,embed_dim+1))
        for j in range(0,len(data_1[i])-embed_dim+1):
            Xt[j,] = data_1[i][np.arange(j,j+embed_dim)]
        for j in range(0,len(data_1[i])-embed_dim):
            Yt[j,] = data_1[i][np.arange(j,j+embed_dim+1)]
        "Correlation integral for embed"         
        total_cases_gt0 = np.zeros(len(Xt))
        for k in range(len(Xt)):
            diff_matrix = Xt[np.arange(0,len(Xt)),]
            """ Subtract 1D from array from 2D array"""
            matrix_sub = Xt[k,]-diff_matrix
            """ Norm of individual rows of the matrix, using L2 norm not max
                as mentioned by most of the papers"""
            matrix_norm = np.linalg.norm(matrix_sub,axis=1)
            compare_norm = r - matrix_norm
            """ Total cases where r > matrixnorm """
            total_cases_gt0[k] = len(np.where(compare_norm >= 0)[0])
        total_cases_gt0 = total_cases_gt0/(len(data_1[i])-embed_dim+1)
        corr_dim_fin[i] = np.sum(np.log(total_cases_gt0))/(len(data_1[i])-embed_dim+1)
        "Correlation integral for embed + 1"
        total_cases_gt0_1 = np.zeros(len(Yt))
        for k in range(len(Yt)):
            diff_matrix = Yt[np.arange(0,len(Yt)),]
            """ Subtract 1D from array from 2D array"""
            matrix_sub = Yt[k,]-diff_matrix
            """ Norm of individual rows of the matrix, using L2 norm not max
                as mentioned by most of the papers"""
            matrix_norm = np.linalg.norm(matrix_sub,axis=1)
            compare_norm = r - matrix_norm
            """ Total cases where r > matrixnorm """
            total_cases_gt0_1[k] = len(np.where(compare_norm >= 0)[0])
        total_cases_gt0_1 = total_cases_gt0_1/(len(data_1[i])-embed_dim)
        corr_dim_fin2[i] = np.sum(np.log(total_cases_gt0_1))/(len(data_1[i])-embed_dim)
        approx_entropy[i] = corr_dim_fin[i] - corr_dim_fin2[i]
    return approx_entropy
    
""" Theiler window in the t, start from i+1+theiler; theiler = 15 
    embedding dimension = 5"""
     
def correlation_dimension(data_1,embed_dim, countr, theiler):
    correlation_dimension = np.zeros(data_1.shape[0])
    corr_dim_fin2 = np.zeros(countr)
    for i in range(data_1.shape[0]):
        rmin = 0.5*np.std(data_1[i])
        rmax = 0.7*np.std(data_1[i])
        Xt = np.zeros((data_1.shape[1]+1-embed_dim,embed_dim))
        xaxis = np.zeros(countr)
        for j in range(0,len(data_1[i])-embed_dim+1):
            Xt[j,] = data_1[i][np.arange(j,j+embed_dim)]
        iter_c = 0
        for r in np.linspace(rmin, rmax, countr):
            total_cases_gt0 = 0
            for k in range(len(Xt)):
                if len(np.arange(k+1+theiler,len(Xt))) == 0:
                    diff_matrix = Xt[k]
                    """ 1-1 comaprison only """
                    total_cases_gt0 += 1
                else:
                    diff_matrix = Xt[np.arange(k+1+theiler,len(Xt)),]
                    """ Subtract 1D from array from 2D array"""
                    matrix_sub = Xt[k,]-diff_matrix
                    """ Norm of individual rows of the matrix """
                    matrix_norm = np.linalg.norm(matrix_sub,axis=1)
                    compare_norm = r - matrix_norm
                    """ Total cases where r > matrix-norm """
                    total_cases_gt0 += len(np.where(compare_norm >= 0)[0])
            total_cases_gt0 = 2*total_cases_gt0/(len(Xt)-theiler)
            """ Final correlation formula GP alorithm """
            corr_dim_fin2[iter_c] = np.log(total_cases_gt0/(len(Xt)-theiler-1))
            xaxis[iter_c] = np.log(r)
            iter_c +=  1
        """ Using regression for power regression 
            Taken log of the data-------"""
        lm = LinearRegression()
        xaxis = xaxis.reshape(-1,1)
        lm.fit(xaxis,np.ravel(corr_dim_fin2))
        correlation_dimension[i] = lm.coef_
    return correlation_dimension
    
def wavelet_transform(data_1,type_w,level_w):
    coeff_mean = np.zeros((data_1.shape[0],level_w+1))
    coeff_std = np.zeros((data_1.shape[0],level_w+1))
    coeff_skew = np.zeros((data_1.shape[0],level_w+1))
    coeff_kurt = np.zeros((data_1.shape[0],level_w+1))
    for i in range(data_1.shape[0]):
        """ Wavelet decomposition """
        w = pywt.wavedec(data_1[i],wavelet=type_w,level=level_w)
        for j in range(0,len(w)):
            coeff_mean[i,j] = np.mean(w[j])
            coeff_std[i,j] = np.std(w[j])
            coeff_skew[i,j] = st.skew(w[j],bias=False)
            coeff_kurt[i,j] = st.kurtosis(w[j],bias=False)
    """ Factor analysis on the wavelet coefficients
        Taking the first component"""
    fa_mean_coeff = FactorAnalysis(n_components = 1).fit(coeff_mean).transform(coeff_mean)
    fa_std_coeff = FactorAnalysis(n_components = 1).fit(coeff_std).transform(coeff_std)
    fa_skew_coeff = FactorAnalysis(n_components = 1).fit(coeff_skew).transform(coeff_skew)
    fa_kurt_coeff = FactorAnalysis(n_components = 1).fit(coeff_kurt).transform(coeff_kurt)
    return fa_mean_coeff, fa_std_coeff, fa_skew_coeff, fa_kurt_coeff
    
def hjorth_param(data_1):
    tp = 0
    m2 = 0
    m4 = 0
    hjorth_mob = np.empty((data_1.shape[0]))
    hjorth_complex = np.empty((data_1.shape[0]))
    for i in range(data_1.shape[0]):
        """ Squared to avoid cases where data's mean = 0"""
        tp = np.sqrt(np.sum(data_1[i]**2))
        m2 = np.sqrt(np.sum((np.diff(data_1[i]))**2)/len(data_1[i]))
        m4 = np.sqrt(np.sum(np.diff(np.diff(data_1[i]))**2)/len(data_1[i]))
        hjorth_mob[i] = np.sqrt(m2/tp)
        hjorth_complex[i] = np.sqrt(m4*tp/(m2**2))
    return hjorth_complex, hjorth_mob

def kalman_filter(data_1,niter):
    kalman_param = np.zeros((data_1.shape[0],2))
    kalman_smooth = np.zeros((data_1.shape[0],data_1.shape[1]))
    """ EM algorithm is slow for Kalman initialization"""
    for i in range(data_1.shape[0]):
        kf = KalmanFilter(em_vars=['transition_matrices', 'observation_matrices',
        'transition_covariance', 'observation_covariance',
        'observation_offsets', 'initial_state_mean','initial_state_covariance'])
        kf = kf.em(X = data_1[i], n_iter = niter)
        kalman_smooth[i,] = np.ravel(kf.smooth(data_1[i])[0])
        kalman_param[i,0] = np.max(kalman_smooth[i,])
        kalman_param[i,1] = np.std(kalman_smooth[i,]) 
    """ Eigen Value for correlated smoothed series """
    eigen_corr,t = np.linalg.eig(np.corrcoef(kalman_smooth))
    eigen_corr.sort(); eigen_corr = eigen_corr[::-1]
    return eigen_corr, kalman_param
        
def phase_sync_angle(data_1):
    """ Hilbert transform of the series """
    analytic_signal = data_1 + (1j*hilbert(data_1))
    bin_count = np.round(np.exp(0.626+0.4*np.log(len(data_1[0])-1)))
    """ Based on shannon entropy, Smax is the max possible shannon entropy"""
    Smax = np.log(bin_count)
    sync_coeff = np.zeros((data_1.shape[0],data_1.shape[0]))
    for i in range(data_1.shape[0]):
        for j in range(i,data_1.shape[0]):
            instantaneous_phase1 = np.unwrap(np.angle(analytic_signal[i]))
            instantaneous_phase2 = np.unwrap(np.angle(analytic_signal[j]))
            "-----Remove cyclicity at 2pi-----"
            phase_diff = np.mod(np.abs(instantaneous_phase1 - instantaneous_phase2),2*np.pi)  
            hist_val = np.histogram(phase_diff,bins = bin_count)[0]
            hist_val = hist_val/np.sum(hist_val)
            """ Formula for shannon entropy calculation"""
            S = -np.sum(hist_val*np.log(hist_val+1e-10))
            sync_coeff[i,j] = (Smax - S)/Smax
            sync_coeff[j,i] = (Smax - S)/Smax
    """Eigen Value of the sychronization coeffiecient
        across all the channels"""
    eigen_val,t = np.linalg.eig(sync_coeff)
    eigen_val.sort(); eigen_val = eigen_val[::-1]    
    return eigen_val

def phase_sync_fft(data_1):
    sync_coeff_fft = np.zeros((data_1.shape[0],data_1.shape[0]))
    Z= np.zeros(int(data_1.shape[1]/2))
    D = np.zeros(int(data_1.shape[1]/2))
    E = np.zeros(int(data_1.shape[1]/2)-1)
    for i in range(data_1.shape[0]):
        for j in range(i,data_1.shape[0]):
            """Entire signal's FFT is taken,
            ---This might get us some spurious frequencies"""
            fft_1 = np.fft.rfft(data_1[i])
            fft_2 = np.fft.rfft(data_1[j])
            for k in range(int(data_1.shape[1]/2)):
                Z[k] = np.real(fft_1[k])*np.real(fft_2[k]) + np.imag(fft_1[k])*np.imag(fft_2[k])
                if Z[k] != 0:
                    D[k] = (np.real(fft_1[k])*np.imag(fft_2[k]) + np.imag(fft_1[k])*np.real(fft_2[k]))/Z[k]
                else:
                    print("asynchronous")
            for k in range(int(data_1.shape[1]/2) - 1):
                E[k] = np.abs(D[k+1] - D[k])
            sync_coeff_fft[i,j] = 1/(1+np.mean(E) + np.std(E))
            sync_coeff_fft[j,i] = 1/(1+np.mean(E) + np.std(E))
    """Eigen Value of the sychronization coeffiecient
        across all the channels"""
    eigen_val,t = np.linalg.eig(sync_coeff_fft)
    eigen_val.sort(); eigen_val = eigen_val[::-1]  
    return eigen_val

#################### MFCC Features and Dynamic Time Warping ###################

def audio_like_features(data_1,freq_signal):
    avg_dtw_coeff = np.zeros(data_1.shape[0])
    mfcc_coeff = defaultdict(list)
    fa_mfcc_coeff = defaultdict(list)
    signal_freq_dist = np.zeros((data_1.shape[0],data_1.shape[0]))
    for i in range(data_1.shape[0]):
        mfcc_coeff[i] = mfcc(data_1[i],freq_signal)
        temp_mfcc = np.array(mfcc_coeff[i])
        """ Calculating latent variable of individual channel"""
        fa_mfcc_coeff[i] = FactorAnalysis(n_components = 1).fit(temp_mfcc).transform(temp_mfcc)
    fa_mfcc_coeff_chann = np.zeros((data_1.shape[0],len(fa_mfcc_coeff[1])))
    for i in range(data_1.shape[0]):
        fa_mfcc_coeff_chann[i,] = np.ravel(fa_mfcc_coeff[i])
    """ Again Factor analysis to get top components """
    fa_fin_mfcc = FactorAnalysis(n_components = 2).fit(fa_mfcc_coeff_chann).transform(fa_mfcc_coeff_chann)
    for i in range(data_1.shape[0]):
        for j in range(i+1,data_1.shape[0]):
            """ Euclidean distance between the mfcc coefficients"""
            temp = dtw(mfcc_coeff[i].T, mfcc_coeff[j].T, dist = euclidean)[0]
            signal_freq_dist[i,j] = temp
            signal_freq_dist[j,i] = temp
        """ Taking the mean of all the distances for 1 channel"""
        avg_dtw_coeff[i] = np.sum(signal_freq_dist[i])/(data_1.shape[0]-1)
    return avg_dtw_coeff, fa_fin_mfcc
        

 
