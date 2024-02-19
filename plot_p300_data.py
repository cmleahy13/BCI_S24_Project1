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
    channel_count = eeg_epochs.shape[1]  # number of channels
    
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
        channel_plot.fill_between(erp_times,target_erp_transpose[channel_index] - 2 * target_standard_error, target_erp_transpose[channel_index] + 2 * target_standard_error, alpha=0.25)
        channel_plot.fill_between(erp_times,nontarget_erp_transpose[channel_index] - 2 * nontarget_standard_error, nontarget_erp_transpose[channel_index] + 2 * nontarget_standard_error, alpha=0.25)
        
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
        
# call to trouble shoot
is_target_event, eeg_epochs, erp_times, target_erp, nontarget_erp = load_erp_data(subject=3, data_directory='P300Data/', epoch_start_time=-0.5, epoch_end_time=1.0)

plot_confidence_intervals(eeg_epochs,erp_times, target_erp, nontarget_erp, is_target_event)

# #%%
# def plot_confidence_intervals(eeg_epochs,erp_times, target_erp, nontarget_erp, is_target_event, subject=3):
      
#     # target statistics
#     mean_target_erp = np.mean(target_erp, axis=1)
#     std_target_erp = np.std(target_erp, axis=1)
#     se_target_erp = std_target_erp / np.sqrt(len(eeg_epochs[is_target_event]))  # Standard error of the mean
    
#     # nontarget statistics
#     mean_nontarget_erp = np.mean(nontarget_erp, axis=1)
#     std_nontarget_erp = np.std(nontarget_erp, axis=1)  # Compute the standard deviation across trials
#     se_nontarget_erp = std_nontarget_erp / np.sqrt(len(eeg_epochs[~is_target_event]))  # Standard error of the mean
    
    
#     # Plot ERPs and confidence intervals for each channel
#     fig, axs = plt.subplots(nrows=3, ncols=3, figsize=(12, 10))
#     axs = axs.flatten()  # Flatten the 2D array of subplots
    
#     for i in range(8):
#         ax = axs[i]
#         #ax.plot(erp_times[i], meantarget_erp[i], label='Mean Target ERP', color='teal')
#         #ax.plot(erp_times[i], mean_nontarget_erp[i], label='Mean Non Target ERP', color='pink')
#         ax.fill_between(erp_times, mean_target_erp - 2 * se_target_erp, mean_target_erp + 2 * se_target_erp,  alpha=0.2, label = 'Target confidence intervals' , lw = 5)
#         ax.fill_between(erp_times, mean_nontarget_erp - 2 * se_nontarget_erp, mean_nontarget_erp + 2 * se_nontarget_erp, alpha=0.2, label = 'Nontarget confidence intervals', lw = 200)
#         ax.plot(erp_times, target_erp[:, i], label='Target ERP' )
#         ax.plot(erp_times, nontarget_erp[:, i], label='Non-target ERP')
#         ax.axvline(0, color='black', linestyle='--')  # Mark stimulus onset
#         ax.axhline(0, color='black', linestyle=':')  # Mark zero voltage
#         ax.set_title(f'Channel {i+1}')
#         ax.set_xlabel('Time (s)')
#         ax.set_ylabel('Voltage (μV)')
        
        
#         plt.tight_layout()
        
#     fig.suptitle(f'P300 Speller S{subject} Training ERPs')
#     fig.tight_layout()
#     ax.legend(loc = 'lower right')    
#     fig.delaxes(axs[8])
#     plt.show()

# # erp_times,is_target_event, target_erp, nontarget_erp, eeg_epochs = load_erp_data(subject=3,data_directory='P300Data/',epoch_start_time=-0.5, epoch_end_time=1.0)
# # plot_confidence_intervals(eeg_epochs,is_target_event,erp_times, target_erp, nontarget_erp)

#%% Part C: Bootstrap P Values

# null hypothesis: no difference between trial types
# each bootstrapping iteration has same number of target/nontarget trials as real data

# function goals:
    # 3000 bootstrapping iterations to find distribution of target-minus-nontarget ERPs
        # number of iterations can be input, default 3000
    # calculate p value for each time point on each channel
        # p value: chance that absolute value as high as that of real data observed by chance

# inputs:
    # iterations (optional), default 3000
    # target_erp?
    # nontarget_erp?
    
# returns:
    # p value?


#%% Part D: Plot FDR-Corrected P Values

# False Discovery Rate correction to correct p values for multiple comparisons

# function goals:
    # add black dot on x-axis (ERP and CI) when ERP difference is significant at FDR-correct p value 0.05
    
# inputs:
    
# returns:

#%% Part E: Evaluate Across Subjects

# function goals:
    # loop through above code on subjects 3-10
    # save different image file for each
    # record time channels/time points that pass FDR-corrected significance threshold
    # make/save new subplots (for each channel) to show number of subjects that passed significance threshold at each time point and channel
    
# inputs:
    # subject array (optional), default array np.arange(3,11)

# returns:
    # none?
    

#%% Part F: Plot a Spatial Map

# function goals:
    # get group median ERP across all trials 
    # get subject median ERP across all trials
    # assign channels spatial location
    # use spatial location assignments as input to plot_topo.plot_topo() function
    # use above function to plot spatial distribution of median voltages in N2 time range (one plot) and P3b time range (second plot)
    # adjust channel name order to obtain reasonable output

# inputs:

# returns:
