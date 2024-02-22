#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 13 10:03:54 2024

test_plot_p300_data.py

This file serves as the test script for Project 1 for BCI-S24. The functions from the module script are first imported so that they can be called here. This script primarily calls the functions under their default conditions, which primarily indicates the data are evaluated for subject 3. The actions of the functions called include loading the training data, plotting the ERP data with their 95% confidence intervals, bootstrapping the data, finding test statistics and using them for p-value calculations, and obtaining signficance by adjusting the p-values for multiple corrections. After these actions are performed, virtually all of them are called again in the function multiple_subject_evaluation, which calls these functions on subjects 3-10 (per default conditions). A plot is then produced to depict how many subjects exhibit significance at a sample in a channel. A scalp map is also produced as a visual representation of voltage by channel location.

Relevant abbreviations:
    EEG: electroencephalography
    ERP: event-related potential
    FDR: false discovery rate

@author: Varshney Gentela and Claire Leahy
"""

#%% Import functions to be called

import numpy as np
from plot_p300_data import load_erp_data, plot_confidence_intervals, bootstrap_erps, test_statistic, calculate_p_values, plot_false_discovery_rate, multiple_subject_evaluation, plot_subject_significance, plot_spatial_map

#%% Load data

is_target_event, eeg_epochs, erp_times, target_erp, nontarget_erp = load_erp_data() # default conditions
print("shape of eeg_epochs:", eeg_epochs.shape)
print("shape of target_erp:", target_erp.shape)
print("shape of nontarget_erp:", nontarget_erp.shape)

#%% Plot confidence intervals with standard error

plot_confidence_intervals(eeg_epochs,erp_times, target_erp, nontarget_erp, is_target_event) # default conditions

#%% Bootstrapping

# declare necessary variables
randomization_count = 3000
sample_count = eeg_epochs.shape[1]
channel_count = eeg_epochs.shape[2]

# preallocate arrays for resampled data 
sampled_target_erp = np.zeros([randomization_count,sample_count,channel_count])
sampled_nontarget_erp = np.zeros([randomization_count,sample_count,channel_count])

# perform the bootstrapping
for randomization_index in range(randomization_count): # perform 3000 randomizations of the bootstrap function
        
    # resample targets
    sampled_target_erp[randomization_index,:] = bootstrap_erps(eeg_epochs, is_target_event)[0][:,:] # 1st return of bootstrap function is for target ERPs
    
    # resample nontargets
    sampled_nontarget_erp[randomization_index,:] = bootstrap_erps(eeg_epochs, is_target_event)[1][:,:] # 2nd return of bootstrap function is for nontarget ERPs
    
# find test statistics
bootstrapped_erp_difference = test_statistic(sampled_target_erp, sampled_nontarget_erp) # resampled data
real_erp_difference = test_statistic(target_erp, nontarget_erp) # actual data
print("real_erp_difference shape:", real_erp_difference.shape)
print("bootstrapped_erp_difference shape:", bootstrapped_erp_difference.shape)
# determine p_values
p_values = calculate_p_values(sampled_target_erp, sampled_nontarget_erp,target_erp, nontarget_erp) # default conditions

#%% FDR correction

# plotting function that returns time points for use within other functions
significant_times = plot_false_discovery_rate(eeg_epochs, erp_times, target_erp, nontarget_erp, is_target_event, p_values) # default conditions

#%% Multiple subjects comparison

# perform the loop on relevant subjects
subject_significance = multiple_subject_evaluation() # default conditions
# Test to make sure the shape is what it is supposed to be 
print("Shape of subject_significance:", subject_significance.shape)
#%%
# plot the number of subjects with significant sample time in a channel
plot_subject_significance(erp_times, subject_significance)

#%% Spatial map

# generate scalp maps
plot_spatial_map(eeg_epochs, is_target_event, erp_times)
#%% Printing out the docstrings of all functions
# Part A: Docstring for Load and Epoch the Data
print("Docstring for load_erp_data function:")
print(load_erp_data.__doc__)

# Part B: Docstring for Calculate & Plot Parametric Confidence Intervals
print("\nDocstring for plot_confidence_intervals function:")
print(plot_confidence_intervals.__doc__)

# Part C: Docstring for Bootstrap P Values
print("\nDocstring for bootstrap_erps function:")
print(bootstrap_erps.__doc__)

# Part D: Docstring for test_statistic function
print("\nDocstring for test_statistic function:")
print(test_statistic.__doc__)

# Part E: Docstring for calculate P Values function
print("\nDocstring for calculate_p_values function:")
print(calculate_p_values.__doc__)

# Part F: Docstring for Plot FDR-Corrected P Values
print("\nDocstring for plot_false_discovery_rate function:")
print(plot_false_discovery_rate.__doc__)

# Part G: Docstring for multiple_subject_evaluation function
print("Docstring for multiple_subject_evaluation function:")
print(multiple_subject_evaluation.__doc__)

# Part H: Docstring for plot_subject_significance function
print("\nDocstring for plot_subject_significance function:")
print(plot_subject_significance.__doc__)

# Part I: Docstring for plot_spatial_map function
print("\nDocstring for plot_spatial_map function:")
print(plot_spatial_map.__doc__)
