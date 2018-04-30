# -*- coding: utf-8 -*-
"""
Created on Wed Aug  3 13:37:32 2016

@author: rsoni106
"""

""" ############### Initial variable creation for Dogs, will create separate model #############"""

import sys
import os

""" ############## To import python code as a package, __init__.py is required ##############"""

sys.path.append(os.path.abspath("C:/Users/rsoni106/Documents/Work/Methodology Work/Seizure Detection/Pythoncode"))

import main_analysis_modules_RS_v05 as eeg_func
import hdf5storage as rd

eeg_data_dir = "C:/Users/rsoni106/Documents/Work/Methodology Work/Seizure Detection/Volumes_win/test"
patient_data = "/mapr/projects/Rishabh_Work/EEG/Patient_DP"

"""########### Calculate all variables from the data and append into a dataframe ################"""

list_dir = []

for (dirpath, dirnames, filename) in os.walk(eeg_data_dir):
    list_dir.append(filename)
    break

final_eeg_data = eeg_func.pd.DataFrame(eeg_func.np.zeros((1,119)))

def data_prep_python(name):    
    data = rd.loadmat(os.path.join(eeg_data_dir,name))
    orig_freq = int(eeg_func.np.round(data["freq"]))
    data_1 = eeg_func.np.array(data["data"])
    data_1 = eeg_func.normalize(data_1)
    window_length = orig_freq/5
    overlap_frac = 0.5
    window_type = eeg_func.hanning
    data_points = data_1.shape[1]
    channelcount = data_1.shape[0]
    data_1 = eeg_func.lowpassfilter(data_1,orig_freq,100,60,20)
    """ FFT split using hamming window """
    fft_split, iter_count = eeg_func.fft_func(data_1,window_length,window_type,overlap_frac)
    """ Power spectrum split by bands"""
    b1,b2,b3,bd1,bd2,bd3 = eeg_func.pow_spectrum(fft_split,data_points,window_length,window_type,orig_freq,channelcount,iter_count)
    """ Channel invariant power spectrum using moments """
    max_power_b1 = max(b1); min_power_b1 = min(b1); mean_power_b1 = eeg_func.np.mean(b1)
    median_power_b1 = eeg_func.np.median(b1); std_power_b1 = eeg_func.np.std(b1)
    max_power_b2 = max(b2); min_power_b2 = min(b2); mean_power_b2 = eeg_func.np.mean(b2)
    median_power_b2 = eeg_func.np.median(b2); std_power_b2 = eeg_func.np.std(b2)
    max_power_b3 = max(b3); min_power_b3 = min(b3); mean_power_b3 = eeg_func.np.mean(b3)
    median_power_b3 = eeg_func.np.median(b3); std_power_b3 = eeg_func.np.std(b3)
    max_power_bd1 = max(bd1); min_power_bd1 = min(bd1); mean_power_bd1 = eeg_func.np.mean(bd1)
    median_power_bd1 = eeg_func.np.median(bd1); std_power_bd1 = eeg_func.np.std(bd1)
    max_power_bd2 = max(bd2); min_power_bd2 = min(bd2); mean_power_bd2 = eeg_func.np.mean(bd2)
    median_power_bd2 = eeg_func.np.median(bd2); std_power_bd2 = eeg_func.np.std(bd2)
    max_power_bd3 = max(bd3); min_power_bd3 = min(bd3); mean_power_bd3 = eeg_func.np.mean(bd3)
    median_power_bd3 = eeg_func.np.median(bd3); std_power_bd3 = eeg_func.np.std(bd3)
    """ Channel invariant entropy using moments """
    shannon_entropy = eeg_func.shannon_entropy(fft_split,channelcount,iter_count)
    renyi_entropy = eeg_func.renyi_entropy(fft_split,channelcount,iter_count)
    max_shannon = max(shannon_entropy); min_shannon = min(shannon_entropy); mean_shannon = eeg_func.np.mean(shannon_entropy)
    median_shannon = eeg_func.np.median(shannon_entropy); std_shannon = eeg_func.np.std(shannon_entropy)
    max_renyi = max(renyi_entropy); min_renyi = min(renyi_entropy); mean_renyi = eeg_func.np.mean(renyi_entropy)
    median_renyi = eeg_func.np.median(renyi_entropy); std_renyi = eeg_func.np.std(renyi_entropy)
    """ Basic Time series stats """
    ts_stats = eeg_func.base_stats(data_1)
    ts_skew_mean = eeg_func.np.mean(ts_stats[:,0]); ts_skew_std = eeg_func.np.std(ts_stats[:,0])
    ts_kurt_mean = eeg_func.np.mean(ts_stats[:,1]); ts_kurt_std = eeg_func.np.std(ts_stats[:,1])
    ts_max_mean = eeg_func.np.mean(ts_stats[:,2]); ts_max_std = eeg_func.np.std(ts_stats[:,2])
    ts_std_mean = eeg_func.np.mean(ts_stats[:,3]); ts_std_std = eeg_func.np.std(ts_stats[:,3])
    """ Time series correlation eigenvalues """
    correlation_matrix = eeg_func.correlation_vals(data_1)
    corr_eigen1 = correlation_matrix[0]; corr_eigen2 = correlation_matrix[1]; corr_eigen3 = correlation_matrix[2]
    corr_eigen4 = correlation_matrix[3]; corr_eigen5 = correlation_matrix[4]
    """ Fractal dimension """
    petrosian_fractal = eeg_func.petrosian_fractal(data_1)
    mean_pet_fractal = eeg_func.np.mean(petrosian_fractal); std_pet_fractal = eeg_func.np.std(petrosian_fractal);
    median_pet_fractal = eeg_func.np.median(petrosian_fractal); max_pet_fractal = max(petrosian_fractal)
    min_pet_fractal = min(petrosian_fractal)
    
    higuchi_fractal = eeg_func.higuchi_fractal(data_1,5)
    mean_hig_fractal = eeg_func.np.mean(higuchi_fractal); std_hig_fractal = eeg_func.np.std(higuchi_fractal);
    median_hig_fractal = eeg_func.np.median(higuchi_fractal); max_hig_fractal = max(higuchi_fractal)
    min_hig_fractal = min(higuchi_fractal)
    
    hurst_exp = eeg_func.hurst_exp(data_1,orig_freq)
    mean_hurst_exp = eeg_func.np.mean(hurst_exp); std_hurst_exp = eeg_func.np.std(hurst_exp);
    median_hurst_exp = eeg_func.np.median(hurst_exp);  max_hurst_exp = max(hurst_exp)
    min_hurst_exp = min(hurst_exp)
    """ 3rd parameter is the iteration count (100), 4th is the theiler window """        
    correlation_dimension = eeg_func.correlation_dimension(data_1,5,25,15)
    mean_corr_dim = eeg_func.np.mean(correlation_dimension); std_corr_dim = eeg_func.np.std(correlation_dimension);
    median_corr_dim = eeg_func.np.median(correlation_dimension); max_corr_dim = max(correlation_dimension)
    min_corr_dim = min(correlation_dimension)
    """ embedding dimension chosen as 5 """
    approx_entropy = eeg_func.approx_entropy(data_1,5)
    mean_approx_entropy = eeg_func.np.mean(approx_entropy); std_approx_entropy = eeg_func.np.std(approx_entropy);
    median_approx_entropy = eeg_func.np.median(approx_entropy); max_approx_entropy = max(approx_entropy)
    min_approx_entropy = min(approx_entropy)
    """ Hjorth Parameters """
    hjorth_complex, hjorth_mob = eeg_func.hjorth_param(data_1)
    mean_hjorth_complex = eeg_func.np.mean(hjorth_complex); std_hjorth_complex = eeg_func.np.std(hjorth_complex)
    median_hjorth_complex = eeg_func.np.median(hjorth_complex); max_hjorth_complex = max(hjorth_complex)
    min_hjorth_complex = min(hjorth_complex)
    mean_hjorth_mob = eeg_func.np.mean(hjorth_mob); std_hjorth_mob = eeg_func.np.std(hjorth_mob)
    median_hjorth_mob = eeg_func.np.median(hjorth_mob); max_hjorth_mob = max(hjorth_mob)
    min_hjorth_mob = min(hjorth_mob)
    """ Kalman Filter, number of iterations for EM, not running for now        
    kalman_eigen, kalman_ts = eeg_func.kalman_filter(data_1,10) 
    kalman_eigen1 = kalman_eigen[0]; kalman_eigen2 = kalman_eigen[1]; kalman_eigen3 = kalman_eigen[2]
    kalman_ts_max_mean = eeg_func.np.mean(kalman_ts[:,0]); kalman_ts_max_std = eeg_func.np.std(kalman_ts[:,0])
    kalman_ts_std_mean = eeg_func.np.mean(kalman_ts[:,0]); kalman_ts_std_std = eeg_func.np.std(kalman_ts[:,0]) """
    """ Eigen values of the wavelet coefficients' moment"""        
    wave_mean, wave_std, wave_skew, wave_kurt = eeg_func.wavelet_transform(data_1,'db4',4)
    wave_mean1 = wave_mean[0]; wave_mean2 = wave_mean[1]; wave_mean3 = wave_mean[2]
    wave_std1 = wave_std[0]; wave_std2 = wave_std[1]; wave_std3 = wave_std[2]
    wave_skew1 = wave_skew[0]; wave_skew2 = wave_skew[1]; wave_skew3 = wave_skew[2]
    wave_kurt1 = wave_kurt[0]; wave_kurt2 = wave_kurt[1]; wave_kurt3 = wave_kurt[2]
    """ Phase synchronization and Locking between channels"""
    eigen_phase_ang = eeg_func.phase_sync_angle(data_1)
    eigen_phase_ang1 = eigen_phase_ang[0]; eigen_phase_ang2 = eigen_phase_ang[1]; eigen_phase_ang3 = eigen_phase_ang[2]
    """ FFT based synchronization """
    eigen_phase_fft = eeg_func.phase_sync_fft(data_1)
    eigen_phase_fft1 = eigen_phase_fft[0]; eigen_phase_fft2 = eigen_phase_fft[1]; eigen_phase_fft3 = eigen_phase_fft[2]
    """ DTW Speech related Features """
    dtw_feature, speech_feature = eeg_func.audio_like_features(data_1,orig_freq)
    dtw_mean = eeg_func.np.mean(dtw_feature); dtw_std = eeg_func.np.std(dtw_feature)
    dtw_median = eeg_func.np.median(dtw_feature)
    """ MFCC Speech related Features """
    speech_fa1_mean = eeg_func.np.mean(speech_feature[:,0])
    speech_fa1_std = eeg_func.np.std(speech_feature[:,0]); speech_fa1_median = eeg_func.np.median(speech_feature[:,0])
    speech_fa1_max = max(speech_feature[:,0]); speech_fa1_min = min(speech_feature[:,0])
    speech_fa2_mean = eeg_func.np.mean(speech_feature[:,1]); speech_fa2_max = max(speech_feature[:,1])
    speech_fa2_std = eeg_func.np.std(speech_feature[:,1]); speech_fa2_median = eeg_func.np.median(speech_feature[:,1]);
    speech_fa2_min = min(speech_feature[:,1])
    """ Now append individual signal features into one """
    final_eeg_data[0,0] = float(max_power_b1); final_eeg_data[0,1] = float(median_power_b1)
    final_eeg_data[0,2] = float(max_power_b2); final_eeg_data[0,3] = float(median_power_b2)
    final_eeg_data[0,4] = float(max_power_b3); final_eeg_data[0,5] = float(median_power_b3) 
    final_eeg_data[0,6] = float(max_power_bd1); final_eeg_data[0,7] = float(median_power_bd1)
    final_eeg_data[0,8] = float(max_power_bd2); final_eeg_data[0,9] = float(median_power_bd2)
    final_eeg_data[0,10] = float(max_power_bd3); final_eeg_data[0,11] = float(median_power_bd3)
    final_eeg_data[0,12] = float(mean_power_b1); final_eeg_data[0,13] = float(mean_power_bd1)
    final_eeg_data[0,14] = float(mean_power_b3); final_eeg_data[0,15] = float(min_power_b1)
    final_eeg_data[0,16] = float(std_power_b1); final_eeg_data[0,17] = float(min_power_b2)
    final_eeg_data[0,18] = float(std_power_b2); final_eeg_data[0,19] = float(min_power_b3)
    final_eeg_data[0,20] = float(std_power_b3); final_eeg_data[0,21] = float(min_power_bd1)
    final_eeg_data[0,22] = float(std_power_bd1); final_eeg_data[0,23] = float(min_power_bd2)
    final_eeg_data[0,24] = float(std_power_bd2); final_eeg_data[0,25] = float(min_power_bd3)
    final_eeg_data[0,26] = float(std_power_bd3); final_eeg_data[0,27] = float(mean_power_b2)
    final_eeg_data[0,28] = float(mean_power_bd3); final_eeg_data[0,29] = float(mean_power_bd2)
    final_eeg_data[0,30] = float(max_shannon); final_eeg_data[0,31] = float(median_shannon)
    final_eeg_data[0,32] = float(max_renyi); final_eeg_data[0,33] = float(median_renyi)
    final_eeg_data[0,34] = float(min_shannon); final_eeg_data[0,35] = float(std_shannon)
    final_eeg_data[0,36] = float(min_renyi); final_eeg_data[0,37] = float(std_renyi)
    final_eeg_data[0,38] = float(mean_shannon); final_eeg_data[0,39] = float(mean_renyi)
    final_eeg_data[0,40] = float(ts_skew_mean); final_eeg_data[0,41] = float(ts_kurt_mean)
    final_eeg_data[0,42] = float(ts_max_mean); final_eeg_data[0,43] = float(ts_std_mean)
    final_eeg_data[0,44] = float(ts_skew_std); final_eeg_data[0,45] = float(ts_kurt_std)
    final_eeg_data[0,46] = float(ts_max_std); final_eeg_data[0,47] = float(ts_std_std)
    final_eeg_data[0,48] = float(corr_eigen1); final_eeg_data[0,49] = float(corr_eigen2)
    final_eeg_data[0,50] = float(corr_eigen3); final_eeg_data[0,51] = float(corr_eigen4)
    final_eeg_data[0,52] = float(corr_eigen5); final_eeg_data[0,53] = float(mean_pet_fractal)
    final_eeg_data[0,54] = float(median_pet_fractal); final_eeg_data[0,55] = float(min_pet_fractal)
    final_eeg_data[0,56] = float(max_pet_fractal); final_eeg_data[0,57] = float(std_pet_fractal)
    final_eeg_data[0,58] = float(mean_hig_fractal); final_eeg_data[0,59] = float(median_hig_fractal)
    final_eeg_data[0,60] = float(min_hig_fractal); final_eeg_data[0,61] = float(std_hig_fractal)
    final_eeg_data[0,62] = float(max_hig_fractal); final_eeg_data[0,63] = float(mean_hurst_exp)
    final_eeg_data[0,64] = float(median_hurst_exp); final_eeg_data[0,65] = float(min_hurst_exp)
    final_eeg_data[0,66] = float(std_hurst_exp); final_eeg_data[0,67] = float(max_hurst_exp)
    final_eeg_data[0,68] = float(mean_corr_dim); final_eeg_data[0,69] = float(median_corr_dim)
    final_eeg_data[0,70] = float(min_corr_dim); final_eeg_data[0,71] = float(std_corr_dim)
    final_eeg_data[0,72] = float(max_corr_dim); final_eeg_data[0,73] = float(mean_approx_entropy)
    final_eeg_data[0,74] = float(median_approx_entropy); final_eeg_data[0,75] = float(min_approx_entropy)
    final_eeg_data[0,76] = float(std_approx_entropy); final_eeg_data[0,77] = float(max_approx_entropy)
    final_eeg_data[0,78] = float(mean_hjorth_complex); final_eeg_data[0,79] = float(median_hjorth_complex)
    final_eeg_data[0,80] = float(min_hjorth_complex); final_eeg_data[0,81] = float(std_hjorth_complex)
    final_eeg_data[0,82] = float(max_hjorth_complex); final_eeg_data[0,83] = float(mean_hjorth_mob)
    final_eeg_data[0,84] = float(median_hjorth_mob); final_eeg_data[0,85] = float(min_hjorth_mob)
    final_eeg_data[0,86] = float(std_hjorth_mob); final_eeg_data[0,87] = float(max_hjorth_mob)
    final_eeg_data[0,88] = float(wave_mean1)
    final_eeg_data[0,89] = float(wave_std1); final_eeg_data[0,90] = float(wave_skew1)
    final_eeg_data[0,91] = float(wave_kurt1); final_eeg_data[0,92] = float(wave_mean2)
    final_eeg_data[0,93] = float(wave_std2); final_eeg_data[0,94] = float(wave_skew2)
    final_eeg_data[0,95] = float(wave_kurt2); final_eeg_data[0,96] = float(wave_mean3)
    final_eeg_data[0,97] = float(wave_std3); final_eeg_data[0,98] = float(wave_skew3)
    final_eeg_data[0,99] = float(wave_kurt3); final_eeg_data[0,100] = float(eigen_phase_ang1)
    final_eeg_data[0,101] = float(eigen_phase_ang2); final_eeg_data[0,102] = float(eigen_phase_ang3)
    final_eeg_data[0,103] = float(eigen_phase_fft1); final_eeg_data[0,104] = float(eigen_phase_fft2)
    final_eeg_data[0,105] = float(eigen_phase_fft3); final_eeg_data[0,106] = float(speech_fa1_std)
    final_eeg_data[0,107] = float(speech_fa1_max); final_eeg_data[0,108] = float(speech_fa2_mean)
    final_eeg_data[0,109] = float(speech_fa2_std); final_eeg_data[0,110] = float(speech_fa2_min)
    final_eeg_data[0,111] = float(speech_fa1_median); final_eeg_data[0,112] = float(speech_fa1_min)
    final_eeg_data[0,113] = float(speech_fa2_max); final_eeg_data[0,114] = float(speech_fa2_median)
    final_eeg_data[0,115] = float(dtw_mean); final_eeg_data[0,116] = float(dtw_std)
    final_eeg_data[0,117] = float(dtw_median); final_eeg_data[0,118] = float(speech_fa1_mean)
    """if name.find("interictal") == -1:
        y[1] = 1
    else:
        y[1] = 0
    print("Task " + str(((1/len(filename)))*100) + "% percent complete")"""
    
    return(final_eeg_data)
    
results = eeg_func.defaultdict(list)

import multiprocessing
def main():
    pool = multiprocessing.Pool()
    results = pool.map(data_prep_python, list_dir[0])
    print(len(results))
    
if __name__ == '__main__':
    main()

final_eeg_data_1 = eeg_func.pd.DataFrame(eeg_func.np.zeros((1,119)))

if __name__ == '__main__':
    pool = Pool(4)
    eeg_func.pd.concat((final_eeg_data_1,eeg_func.pd.DataFrame(pool.map(data_prep_python, list_dir[0]))),axis=1)
    
final_eeg_data_1 = eeg_func.pd.DataFrame()    
    
for i in range(len(temp_data)):
    """ List has dataframes in it, not sure why there are twice # row"""
    p = temp_data[0].iloc[:,119:238]
    final_eeg_data_1 = eeg_func.pd.concat((final_eeg_data_1,p))
    
final_eeg_data_1.columns = ["max_power_b1","median_power_b1","max_power_b2","median_power_b2","max_power_b3",
"median_power_b3","max_power_bd1","median_power_bd1","max_power_bd2","median_power_bd2","max_power_bd3","median_power_bd3",
"mean_power_b1","mean_power_bd1","mean_power_b3","min_power_b1","std_power_b1","min_power_b2","std_power_b2","min_power_b3",
"std_power_b3","min_power_bd1","std_power_bd1","min_power_bd2","std_power_bd2","min_power_bd3","std_power_bd3",
"mean_power_b2","mean_power_bd3","mean_power_bd2","max_shannon","median_shannon","max_renyi","median_renyi","min_shannon",
"std_shannon","min_renyi","std_renyi","mean_shannon","mean_renyi","ts_skew_mean","ts_kurt_mean","ts_max_mean","ts_std_mean",
"ts_skew_std","ts_kurt_std","ts_max_std","ts_std_std","corr_eigen1","corr_eigen2","corr_eigen3","corr_eigen4","corr_eigen5",
"mean_pet_fractal","median_pet_fractal","min_pet_fractal","max_pet_fractal","std_pet_fractal","mean_hig_fractal",
"median_hig_fractal","min_hig_fractal","std_hig_fractal","max_hig_fractal","mean_hurst_exp","median_hurst_exp",
"min_hurst_exp","std_hurst_exp","max_hurst_exp","mean_corr_dim","median_corr_dim","min_corr_dim","std_corr_dim",
"max_corr_dim","mean_approx_entropy","median_approx_entropy","min_approx_entropy","std_approx_entropy","max_approx_entropy",
"mean_hjorth_complex","median_hjorth_complex","min_hjorth_complex","std_hjorth_complex","max_hjorth_complex",
"mean_hjorth_mob","median_hjorth_mob","min_hjorth_mob","std_hjorth_mob","max_hjorth_mob",
"wave_mean1","wave_std1","wave_skew1","wave_kurt1","wave_mean2","wave_std2","wave_skew2","wave_kurt2","wave_mean3",
"wave_std3","wave_skew3","wave_kurt3","eigen_phase_ang1","eigen_phase_ang2","eigen_phase_ang3","eigen_phase1_fft1",
"eigen_phase1_fft2","eigen_phase1_fft3","speech_fa1_std","speech_fa1_max","speech_fa2_mean","speech_fa2_std",
"speech_fa2_min","speech_fa1_median","speech_fa1_min","speech_fa2_max","speech_fa2_median","dtw_mean","dtw_std","dtw_median",
"speech_fa1_mean"]

final_eeg_data_1.to_csv(patient_data + "/eeg_variables_python.csv")

if __name__ == "__main__":
    import multiprocessing as mp; mp.set_start_method('forkserver')
    pool = mp.Pool(2)
    results = [pool.apply_async(data_prep_python, files) for files in list_dir[0]]
    
for result in results:
    result.get()
    
from joblib import Parallel, delayed

if __name__ == "__main__":
    Parallel(n_jobs=4)(delayed(data_prep_python)(file)
                           for file in list_dir[0])

import multiprocessing
import time                               
def main ():
    list_dir = []
    for (dirpath, dirnames, filename) in os.walk(eeg_data_dir):
        list_dir.append(filename)
    multiprocessing.set_start_method('spawn')    
    with multiprocessing.Pool(2) as p:
        results = [p.apply_async (data_prep_python, name) for name in list_dir[0]]
        for r in results:
            print (r.get())

if __name__ == '__main__':
    main()
    
results = defaultdict(list) 

import sys
from collections import defaultdict
from multiprocessing import Pool, Queue

def build_concordance(list_dir):
    global results
    pool = Pool(2)
    results = pool.map(data_prep_python, list_dir)
    for result in results:
        results.append(final_eeg_data)
    print("done")

def main():
    list_dir = []
    for (dirpath, dirnames, filename) in os.walk(eeg_data_dir):
        list_dir.append(filename)
    build_concordance(list_dir[0])

if __name__ == "__main__":
    main()

p = eeg_func.defaultdict(eeg_func.pd.DataFrame())