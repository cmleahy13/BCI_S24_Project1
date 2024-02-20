#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 12 22:23:10 2024

@authors: Varshney Gentela and Claire Leahy
"""

# import lab modules (Part A, Part D)
import numpy as np
from matplotlib import pyplot as plt
from load_p300_data import load_training_eeg
from plot_p300_erps import get_events, epoch_data, get_erps, plot_erps
from mne.stats import fdr_correction
import plot_topo

#%% Part A: Load and Epoch the Data

def load_erp_data(subject=3, data_directory='P300Data/', epoch_start_time=-0.5, epoch_end_time=1.0):
    
    # load in training data
    eeg_time, eeg_data, rowcol_id, is_target = load_training_eeg(subject, data_directory)
    
    
    # extract target and nontarget epochs
    event_sample, is_target_event = get_events(rowcol_id, is_target)
    eeg_epochs, erp_times = epoch_data(eeg_time, eeg_data, event_sample, epoch_start_time, epoch_end_time)
    
    # calculate ERPs
    target_erp, nontarget_erp = get_erps(eeg_epochs, is_target_event)
    
    return is_target_event, eeg_epochs, erp_times, target_erp, nontarget_erp

#%% Part B: Calculate & Plot Parametric Confidence Intervals

def plot_confidence_intervals(eeg_epochs, erp_times, target_erp, nontarget_erp, is_target_event, subject=3):
    
    target_count = len(eeg_epochs[is_target_event])
    nontarget_count = len(eeg_epochs[~is_target_event])
    sample_count = eeg_epochs.shape[1]  # number of channels
    channel_count = eeg_epochs.shape[2]  # number of channels
    
    # calculate statistics for target ERPs
    target_standard_deviation = np.std(target_erp, axis=1) 
    target_standard_error = target_standard_deviation / np.sqrt(target_count)
    
    # calculate statistics for nontarget ERPs
    nontarget_standard_deviation = np.std(nontarget_erp, axis=1)
    nontarget_standard_error = nontarget_standard_deviation / np.sqrt(nontarget_count)
    
    # transpose the erp data to plot, matches average at that sample time to the size of the time array
    target_erp_transpose = np.transpose(target_erp)
    nontarget_erp_transpose = np.transpose(nontarget_erp)
    
    # get channel count
    channel_count = len(target_erp_transpose) # same as if nontargets were used
    
    # plot ERPs for events for each channel
    figure, channel_plots = plt.subplots(3,3, figsize=(10, 6))
    channel_plots[2][2].remove()  # only 8 channels, 9th plot unnecessary
   
    for channel_index in range(channel_count):
        
        row_index, column_index = divmod(channel_index, 3)  # wrap around to column 0 for every 3 plots
        
        channel_plot = channel_plots[row_index][column_index] # subplot
        
        # plot dotted lines for time 0 and 0 voltage
        channel_plot.axvline(0, color='black', linestyle='dotted')
        channel_plot.axhline(0, color='black', linestyle='dotted')
        
        # plot target and nontarget erp data in the subplot
        target_handle, = channel_plot.plot(erp_times, target_erp_transpose[channel_index])
        nontarget_handle, = channel_plot.plot(erp_times, nontarget_erp_transpose[channel_index])
        
        # plot confidence intervals
        channel_plot.fill_between(erp_times,target_erp_transpose[channel_index] - 2 * target_standard_error, target_erp_transpose[channel_index] + 2 * target_standard_error, alpha=0.5)
        
        channel_plot.fill_between(erp_times,nontarget_erp_transpose[channel_index] - 2 * nontarget_standard_error, nontarget_erp_transpose[channel_index] + 2 * nontarget_standard_error, alpha=0.5)
        
        # workaround for legend to only display each entry once
        if channel_index == 0:
            target_handle.set_label('Target')
            nontarget_handle.set_label('Nontarget')
        
        # label each plot's axes and channel number
        channel_plot.set_title(f'Channel {channel_index}')
        channel_plot.set_xlabel('time from flash onset (s)')
        channel_plot.set_ylabel('Voltage (μV)')
    
    # formatting
    figure.suptitle(f'P300 Speller S{subject} Training ERPs')
    figure.legend(loc='lower right', fontsize='xx-large') # legend in space of nonexistent plot 9
    figure.tight_layout()  # stop axis labels overlapping titles
    
    # save image
    plt.savefig(f'P300_S{subject}_channel_plots.png')  # save as image

#%% Part C: Bootstrap P Values

def bootstrap_erps(eeg_epochs, is_target_event):
    
    event_count = eeg_epochs.shape[0]
    
    # Sample random indices for each channel separately
    random_indices = np.random.randint(event_count, size=event_count)
    
    eeg = eeg_epochs[random_indices]
        
    sampled_target_erp = np.mean(eeg[is_target_event], axis=0)
    sampled_nontarget_erp = np.mean(eeg[~is_target_event], axis=0)
    
    return sampled_target_erp, sampled_nontarget_erp

def bootstrapped_test_statistic(sampled_target_erp, sampled_nontarget_erp):
    
    erp_difference = np.abs(sampled_target_erp-sampled_nontarget_erp, axis=0)
    
    return erp_difference

def bootstrap_confidence_interval(sampled_target_erp, sampled_nontarget_erp):
    
    sampled_target_erp.sort(axis=0)
    sampled_nontarget_erp.sort(axis=0)
    randomizations = len(sampled_target_erp) # same as nontarget
    
    target_upper_confidence_interval = sampled_target_erp[int(0.95*randomizations),:,:] # upper 5% because of absolute value
    
    nontarget_lower_confidence_interval = sampled_nontarget_erp[int(0.025*randomizations),:,:]
    nontarget_upper_confidence_interval = sampled_nontarget_erp[int(0.975*randomizations),:,:]

#     eeg_stack = np.vstack((target_erp, nontarget_erp))
    
#     max_abs_diff_samples = []
#     for _ in range(iterations): 
#         sample_count = len(eeg_epochs[1])
#         if size is None:
#             size = sample_count
#         i = np.random.randint(sample_count, size=size)
#         eeg0 = eeg_stack[i]
#         mean_stacked_epochs = eeg0.mean(1)
        
#         i = np.random.randint(sample_count, size=size)
#         eeg1 = eeg_stack[i]
#         mean_stacked_epochs_two = eeg1.mean(1)
        
#         random_mean_diff = np.subtract(mean_stacked_epochs, mean_stacked_epochs_two)
#         max_abs_diff_samples.append(np.max(np.abs(random_mean_diff)))
    
#     return max_abs_diff_samples, stat

# def calculate_p_values(eeg_epochs, is_target_event, target_erp, nontarget_erp, size=None, iterations=3000):
    
#     # Perform bootstrapping to obtain the distribution of differences
#     max_abs_diff_samples, observed_statistic = bootstrapping(eeg_epochs, is_target_event, target_erp, nontarget_erp, size, iterations)
    
#     # Calculate p-values
#     p_values = []
#     for channel_index in range(eeg_epochs.shape[2]):  # Iterate over channels
#         for sample_index in range(eeg_epochs.shape[1]):  # Iterate over time points
#             # Count how many bootstrapped samples have a statistic greater than or equal to the observed statistic
#             count = sum(diff >= observed_statistic for diff in max_abs_diff_samples)
#             # Calculate the p-value as the proportion of samples with a statistic greater than or equal to the observed statistic
#             p_value = count / iterations
#             p_values.append(p_value)
    
#     # Reshape p-values array to match the shape of the target_erp and nontarget_erp arrays
#     p_values = np.array(p_values).reshape(eeg_epochs[is_target_event].shape[1:])
    
#     return p_values

# # Perform bootstrapping and obtain max_abs_diff samples and initial stat value
# max_abs_diff_samples, initial_stat = bootstrapping(eeg_epochs, is_target_event, target_erp, nontarget_erp)
# p_values = calculate_p_values(eeg_epochs, is_target_event, target_erp, nontarget_erp)

# #print(p_values)

# # Plot histogram of max_abs_diff samples and initial stat value
# plt.figure(figsize=(8, 6))
# plt.hist(max_abs_diff_samples, bins=10, color='teal', label='Max Absolute Difference')  # Plot histogram of max_abs_diff
# plt.axvline(x=initial_stat, color='orange', linestyle='--', label='Initial Stat Value')  # Plot initial stat value
# plt.xlabel('Max Absolute Difference')
# plt.ylabel('Frequency')
# plt.title('Distribution of Max Absolute Difference from Bootstrap')
# plt.legend()
# plt.show()

# # at a given sample index, we want to know if there's overlap between the bootstrapped confidence intervals

# # target_erp_bootstrap_lci > nontarget_erp_bootstrap_uci OR 


# #%% Part D: Plot FDR-Corrected P Values

# # False Discovery Rate correction to correct p values for multiple comparisons

# # function goals:
#     # add black dot on x-axis (ERP and CI) when ERP difference is significant at FDR-correct p value 0.05
    
# # inputs:
    
# # returns:

# def fdr_correction(erp_times, target_erp, nontarget_erp, p_values, fdr_threshold = 0.05):
#     res, corrected_p_values = fdr_correction(p_values.flatten(), alpha=fdr_threshold)
#     print(res)
#     meantarget_erp = np.mean(target_erp, axis=1)  # Compute the mean ERP across trials
#     std_erp = np.std(target_erp, axis=1)  # Compute the standard deviation across trials
#     sdmn1 = std_erp / np.sqrt(len(eeg_epochs[is_target_event]))  # Standard error of the mean
#     mean_nontarget_erp = np.mean(nontarget_erp, axis=1) # Compute the mean ERP across trials
#     std_erp2 = np.std(nontarget_erp, axis=1)  # Compute the standard deviation across trials
#     sdmn2 = std_erp2 / np.sqrt(len(eeg_epochs[~is_target_event]))  # Standard error of the mean
    
#     #corrected_p_values = corrected_p_values.reshape(eeg_epochs[is_target_event].shape[1:])
#     #print(corrected_p_values.shape)
#     # Plot ERPs and confidence intervals for each channel
#     fig, axs = plt.subplots(nrows=3, ncols=3, figsize=(12, 10))
#     axs = axs.flatten()  # Flatten the 2D array of subplots
#     #significant_time_points_indices = np.where(corrected_p_values < fdr_threshold)
#     erp_times = np.array(erp_times)
#     for i in range(8):
#         ax = axs[i]
#         #ax.plot(erp_times[i], meantarget_erp[i], label='Mean Target ERP', color='teal')
#         #ax.plot(erp_times[i], mean_nontarget_erp[i], label='Mean Non Target ERP', color='pink')
#         ax.fill_between(erp_times, meantarget_erp - 2 * sdmn1, meantarget_erp + 2 * sdmn1,  alpha=0.2, label = 'Target confidence intervals' , lw = 5)
#         ax.fill_between(erp_times, mean_nontarget_erp - 2 * sdmn2, mean_nontarget_erp + 2 * sdmn2, alpha=0.2, label = 'Nontarget confidence intervals', lw = 200)
#         ax.plot(erp_times, target_erp[:, i], label='Target ERP' )
#         ax.plot(erp_times, nontarget_erp[:, i], label='Non-target ERP')
#         ax.axvline(0, color='black', linestyle='--')  # Mark stimulus onset
#         ax.axhline(0, color='black', linestyle=':')  # Mark zero voltage
#         ax.set_title(f'Channel {i+1}')
#         ax.set_xlabel('Time (s)')
#         ax.set_ylabel('Voltage (μV)')
        
#         significant_time_points = erp_times[np.where(corrected_p_values < fdr_threshold)]
#         ax.plot(significant_time_points, np.zeros_like(significant_time_points), 'ko')  # Plot black dots on x-axis
        
#         plt.tight_layout()
        
    
#     ax.legend(loc='lower right')
#     fig.delaxes(axs[8])

#     # Save the resulting image file
#     plt.show()
    
#     return None

# #%% Part E: Evaluate Across Subjects

# # function goals:
#     # loop through above code on subjects 3-10
#     # save different image file for each
#     # record time channels/time points that pass FDR-corrected significance threshold
#     # make/save new subplots (for each channel) to show number of subjects that passed significance threshold at each time point and channel
    
# # inputs:
#     # subject array (optional), default array np.arange(3,11)

# # returns:
#     # none?
    

# #%% Part F: Plot a Spatial Map

# # function goals:
#     # get group median ERP across all trials 
#     # get subject median ERP across all trials
#     # assign channels spatial location
#     # use spatial location assignments as input to plot_topo.plot_topo() function
#     # use above function to plot spatial distribution of median voltages in N2 time range (one plot) and P3b time range (second plot)
#     # adjust channel name order to obtain reasonable output

# # inputs:

# # returns: