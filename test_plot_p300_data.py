#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 13 10:03:54 2024

@author: Varshney Gentela and Claire Leahy
"""

#%% Import functions to be called

import numpy as np
from plot_p300_data import load_erp_data, plot_confidence_intervals, bootstrap_erps, test_statistic, calculate_p_values, plot_false_discovery_rate, multiple_subject_evaluation, plot_subject_significance, plot_spatial_map

#%% Load data

is_target_event, eeg_epochs, erp_times, target_erp, nontarget_erp = load_erp_data(subject=3, data_directory='P300Data/', epoch_start_time=-0.5, epoch_end_time=1.0)

#%% Plot confidence intervals standard error

plot_confidence_intervals(eeg_epochs,erp_times, target_erp, nontarget_erp, is_target_event)

#%% Bootstrapping

# declare necessary variables
randomization_count = 3000
sample_count = eeg_epochs.shape[1]
channel_count = eeg_epochs.shape[2]

# preallocate arrays for resampled data 
sampled_target_erp = np.zeros([randomization_count,sample_count,channel_count])
sampled_nontarget_erp = np.zeros([randomization_count,sample_count,channel_count])

# perform the bootstrapping
for randomization_index in range(randomization_count):
        
    # resample targets
    sampled_target_erp[randomization_index,:] = bootstrap_erps(eeg_epochs, is_target_event)[0][:,:]
    
    # resample nontargets
    sampled_nontarget_erp[randomization_index,:] = bootstrap_erps(eeg_epochs, is_target_event)[1][:,:]
    
# find test statistics
bootstrapped_erp_difference = test_statistic(sampled_target_erp, sampled_nontarget_erp)
real_erp_difference = test_statistic(target_erp, nontarget_erp)

# determine p_values
p_values = calculate_p_values(sampled_target_erp, sampled_nontarget_erp,target_erp, nontarget_erp,randomization_count=3000)

#%% FDR correction

significant_times = plot_false_discovery_rate(eeg_epochs, erp_times, target_erp, nontarget_erp, is_target_event, p_values, subject=3, fdr_threshold = 0.05)

#%% Multiple subjects comparison
subject_significance = multiple_subject_evaluation(subjects=np.arange(3,11), data_directory='P300Data/', sample_count=384, channel_count=8, epoch_start_time=-0.5, epoch_end_time=1.0, randomization_count=3000, fdr_threshold=0.05)

plot_subject_significance(erp_times, subject_significance)

#%% Spatial map

#plot_spatial_map(eeg_epochs, is_target_event, subject=3)