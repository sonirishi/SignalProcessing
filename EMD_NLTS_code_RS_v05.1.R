rm(list=ls(all=TRUE))

library(R.matlab)
library(EMD)
library(dplyr)
library(fractal)
library(nonlinearTseries)
library(pracma)
library(moments)

######## EMD function to split the sereis #########

emd_split = function(eeg_data,spar_val,data_points){
  max_imf = as.data.frame(matrix(rep(0,nrow(eeg_data)),nrow(eeg_data),5))
  std_imf = as.data.frame(matrix(rep(0,nrow(eeg_data)),nrow(eeg_data),5))
  skew_imf = as.data.frame(matrix(rep(0,nrow(eeg_data)),nrow(eeg_data),5))
  kurt_imf = as.data.frame(matrix(rep(0,nrow(eeg_data)),nrow(eeg_data),5))
  ############ Actual EMD analysis ###############
  for (i in 1:nrow(eeg_data)){
    eeg_data_1 = eeg_data[i,]
    eeg_data_1 = as.numeric(eeg_data_1)
    IMF = EMD::emd(eeg_data_1, tt=seq(1:data_points),boundary="wave", sm = "spline", check=FALSE, spar=spar_val,
              max.imf=5, plot.imf=FALSE, interm=NULL)
    IMF_vals = as.data.frame(IMF["imf"])
    for (j in 1:ncol(IMF_vals)){
      max_imf[i,j] = max(IMF_vals[,j])
      std_imf[i,j] = sd(IMF_vals[,j])
      skew_imf[i,j] = moments::skewness(IMF_vals[,j])
      kurt_imf[i,j] = moments::kurtosis(IMF_vals[,j])
    }
  }
  rem = which(as.numeric(apply(max_imf,2,sum)) == 0)
  if(length(rem) > 0){
    max_fit = factanal(max_imf[,-rem], 1, rotation="varimax",scores="regression")$scores
    std_fit = factanal(std_imf[,-rem], 1, rotation="varimax",scores="regression")$scores
    skew_fit = factanal(skew_imf[,-rem], 1, rotation="varimax",scores="regression")$scores
    kurt_fit = factanal(kurt_imf[,-rem], 1, rotation="varimax",scores="regression")$scores
  } else{
    max_fit = factanal(max_imf, 1, rotation="varimax",scores="regression")$scores
    std_fit = factanal(std_imf, 1, rotation="varimax",scores="regression")$scores
    skew_fit = factanal(skew_imf, 1, rotation="varimax",scores="regression")$scores
    kurt_fit = factanal(kurt_imf, 1, rotation="varimax",scores="regression")$scores
  }
  return(list(max_fit,std_fit,skew_fit,kurt_fit))
}  

### sum.order = 1 means cumulative summation before subtracting the average ###

detrended_fluc = function(eeg_data){
  dfa_val = as.data.frame(matrix(rep(0,nrow(eeg_data)),nrow(eeg_data),1))
  for(i in 1:nrow(eeg_data)){
    eeg_data_1 = eeg_data[i,]
    eeg_data_1 = as.numeric(eeg_data_1)
    dfa_val[i,1] = as.numeric(fractal::DFA(eeg_data_1, detrend = "poly2", overlap = 0.5, sum.order = 1)[1])
  }
  return(dfa_val)
}

##### Grassberger-Procaccia algorithm, theiler = 15, taking mean of sample entropy####

sample_entropy = function(eeg_data){
  samp_entr = as.data.frame(matrix(rep(0,nrow(eeg_data)),nrow(eeg_data),1))
  for(k in 1:nrow(eeg_data)){
    eeg_data_1 = eeg_data[k,]
    eeg_data_1 = as.numeric(eeg_data_1)
    cdval = nonlinearTseries::corrDim(time.series=eeg_data_1,
                                      min.embedding.dim = 2, max.embedding.dim = 9,
                                      corr.order=2, time.lag=1,
                                      min.radius = 0.5*sd(eeg_data_1), max.radius = 0.7*sd(eeg_data_1),
                                      n.points.radius = 15,
                                      theiler.window = 15, do.plot = FALSE)
    samp_ent = nonlinearTseries::sampleEntropy(cdval,do.plot=FALSE)
    d = tryCatch(mean(nonlinearTseries::estimate(samp_ent,use.embeddings = 5:9,do.plot = FALSE)),
                 error=function(e){d <- NA})
    samp_entr[k,1] = d
  }
  return(samp_entr)
}

##### mean of mutual information excluding time lag = 0 ####

mutu_info = function(eeg_data){
  mut_info = as.data.frame(matrix(rep(0,nrow(eeg_data)),nrow(eeg_data),1))
  for(i in 1:nrow(eeg_data)){
    eeg_data_1 = eeg_data[i,]
    eeg_data_1 = as.numeric(eeg_data_1)
    m_vec = nonlinearTseries::mutualInformation(eeg_data_1, lag.max = 16, n.partitions = 32, units = "Bits", 
                              do.plot = FALSE)
    mut_info[i,1] = mean(m_vec$mutual.information[2:length(m_vec$mutual.information)])
  }
  return(mut_info)
}

##### Test analysis, not sure of the impact, kept r same as approx entropy #####

rqa_analysis = function(eeg_data){
  rqa_val = as.data.frame(matrix(rep(0,nrow(eeg_data)),nrow(eeg_data),4))
  for(i in 1:nrow(eeg_data)){
    eeg_data_1 = eeg_data[i,]
    eeg_data_1 = as.numeric(eeg_data_1)
    rqa.ana = nonlinearTseries::rqa(time.series = eeg_data_1, embedding.dim=5, time.lag=1,
                     radius=0.2*sd(eeg_data_1),lmin=2,do.plot=FALSE,distanceToBorder=2)
    rqa_val[i,1] = rqa.ana$REC
    rqa_val[i,2] = rqa.ana$ENTR
    rqa_val[i,3] = rqa.ana$Vmean
    rqa_val[i,4] = rqa.ana$RATIO
  }
  return(rqa_val)
}

###### Max Lyapunov exponent code, Experimental only, Not sure on max time steps######
###### Radius again taken as 0.2* std of the data ########

lyapunov_analysis = function(eeg_data){
  lyapunov_val = as.data.frame(matrix(rep(0,nrow(eeg_data)),nrow(eeg_data),1))
  for(i in 1:nrow(eeg_data)){
    eeg_data_1 = eeg_data[i,]
    eeg_data_1 = as.numeric(eeg_data_1)
    ml1=nonlinearTseries::maxLyapunov(time.series=eeg_data_1,
                    min.embedding.dim=2,
                    max.embedding.dim=2,
                    time.lag=1,
                    radius=0.2*sd(eeg_data_1),theiler.window=15,
                    min.neighs=5,min.ref.points=length(eeg_data_1),
                    max.time.steps=100,do.plot=FALSE)
    
    lyapunov_val[i,1] = estimate(ml1,do.plot = FALSE)
  }
  return(lyapunov_val)
}
