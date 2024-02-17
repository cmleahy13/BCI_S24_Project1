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

# function goals:
    # load in data
    # extract target/nontarget epochs
    # calculate ERPs
    
# inputs: 
    # subject number
    # does it make sense to include data_directory?
    # does it also make snese to include epoch_start_time and epoch_end_time?
    
# returns: 
    # erp_times
    # target_erp
    # non_target_erp
    # does not make snese to include eeg_epochs?

def load_erp_data(subject=3, data_directory='P300Data/', epoch_start_time=-0.5, epoch_end_time=1.0):
    
    # load in training data
    eeg_time, eeg_data, rowcol_id, is_target = load_training_eeg(subject, data_directory)
    
    
    # extract target and nontarget epochs
    event_sample, is_target_event = get_events(rowcol_id, is_target)
    eeg_epochs, erp_times = epoch_data(eeg_time, eeg_data, event_sample, epoch_start_time, epoch_end_time)
    
    # calculate ERPs
    target_erp, nontarget_erp = get_erps(eeg_epochs, is_target_event)
    
    return erp_times, target_erp, nontarget_erp, eeg_epochs, is_target_event

#%% Part B: Calculate & Plot Parametric Confidence Intervals

# assuming normal distribution of voltages for each channel and time point
# define 95% CI based on SE of mean

# function goals:
    # plot ERPs on each channel for target/nontarget events
    # plot confidence intervals as error bars

# inputs:
    # target_erp
    # nontarget_erp
    
# returns:
    # none?
def plot_confidence_intervals(eeg_epochs,erp_times, target_erp, nontarget_erp, is_target_event, subject=3):
    
    target_count = len(eeg_epochs[is_target_event])
    nontarget_count = len(eeg_epochs[~is_target_event])
    channel_count = eeg_epochs.shape[1]  # Number of channels
    
    # calculate statistics for target ERPs
    target_standard_deviation = np.std(target_erp, axis=1)  # Compute the standard deviation across trials
    target_standard_error = target_standard_deviation / np.sqrt(target_count)  # Standard error of the mean
    
    # calculate statistics for nontarget ERPs
    nontarget_standard_deviation = np.std(nontarget_erp, axis=1)  # Compute the standard deviation across trials
    nontarget_standard_error = nontarget_standard_deviation / np.sqrt(nontarget_count)  # Standard error of the mean
    
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
erp_times, target_erp, nontarget_erp, eeg_epochs, is_target_event = load_erp_data(subject=3,data_directory='P300Data/',epoch_start_time=-0.5, epoch_end_time=1.0)
plot_confidence_intervals(eeg_epochs,erp_times, target_erp, nontarget_erp, is_target_event)
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
