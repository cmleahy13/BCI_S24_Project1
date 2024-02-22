#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 12 22:23:10 2024

@authors: Varshney Gentela and Claire Leahy

Sources:
    
    Nick Bosley gave tips on approach to standard deviation calculations
    
"""

# import lab modules (Part A, Part D)
import numpy as np
from matplotlib import pyplot as plt
from load_p300_data import load_training_eeg
from plot_p300_erps import get_events, epoch_data, get_erps
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
    
    # identify necessary counts
    target_count = len(eeg_epochs[is_target_event])
    nontarget_count = len(eeg_epochs[~is_target_event])
    channel_count = eeg_epochs.shape[2]
    
    # statistics calculated for given channel
    for channel_index in range(channel_count):
        
        # calculate statistics for target ERPs
        target_standard_deviation = np.std(eeg_epochs[is_target_event,:,channel_index], axis=0) # standard deviation of EEG data for target events at a given sample time point
        target_standard_error = target_standard_deviation / np.sqrt(target_count) # standard error for targets
        
        # calculate statistics for nontarget ERPs
        nontarget_standard_deviation = np.std(eeg_epochs[~is_target_event,:,channel_index], axis=0) # standard deviation of EEG data for nontarget events at a given sample time point
        nontarget_standard_error = nontarget_standard_deviation / np.sqrt(nontarget_count) # standard error for nontargets
    
    # transpose the erp data to plot, matches average at that sample time to the size of the time array
    target_erp_transpose = np.transpose(target_erp)
    nontarget_erp_transpose = np.transpose(nontarget_erp)
    
    # plot ERPs for events for each channel
    figure, channel_plots = plt.subplots(3,3, figsize=(10, 6))
    channel_plots[2][2].remove()  # only 8 channels, 9th plot unnecessary
   
    for channel_index in range(channel_count):
        
        row_index, column_index = divmod(channel_index, 3)  # wrap around to column 0 for every 3 plots
        
        channel_plot = channel_plots[row_index][column_index] # subplot
        
        # plot dotted lines for time 0 and 0 voltage
        channel_plot.axvline(0, color='black', linestyle='dotted')
        channel_plot.axhline(0, color='black', linestyle='dotted')
        
        # plot ERP data in the subplot
        target_handle, = channel_plot.plot(erp_times, target_erp_transpose[channel_index])
        
        nontarget_handle, = channel_plot.plot(erp_times, nontarget_erp_transpose[channel_index])
        
        # plot confidence intervals
        target_confidence_interval_handle = channel_plot.fill_between(erp_times,target_erp_transpose[channel_index] - 2 * target_standard_error, target_erp_transpose[channel_index] + 2 * target_standard_error, alpha=0.25)
        
        nontarget_confidence_interval_handle = channel_plot.fill_between(erp_times,nontarget_erp_transpose[channel_index] - 2 * nontarget_standard_error, nontarget_erp_transpose[channel_index] + 2 * nontarget_standard_error, alpha=0.25)
        
        # workaround for legend to only display each entry once
        if channel_index == 0:
            target_handle.set_label('Target')
            nontarget_handle.set_label('Nontarget')
            target_confidence_interval_handle.set_label('Target +/- 95% CI')
            nontarget_confidence_interval_handle.set_label('Nontarget +/- 95% CI')
        
        
        # label each plot's axes and channel number
        channel_plot.set_title(f'Channel {channel_index}')
        channel_plot.set_xlabel('time from flash onset (s)')
        channel_plot.set_ylabel('Voltage (μV)')
    
    # formatting
    figure.suptitle(f'P300 Speller S{subject} Training ERPs')
    figure.legend(loc='lower right', fontsize='x-large') # legend in space of nonexistent plot 9
    figure.tight_layout()  # stop axis labels overlapping titles
    
    # save image
    plt.savefig(f'P300_S{subject}_channel_plots.png')

#%% Part C: Bootstrap P Values

def bootstrap_erps(eeg_epochs, is_target_event):
    
    event_count = eeg_epochs.shape[0]
    
    # Sample random indices for each channel separately
    random_indices = np.random.randint(event_count, size=event_count)
    
    eeg = eeg_epochs[random_indices]
        
    sampled_target_erp = np.mean(eeg[is_target_event], axis=0)
    sampled_nontarget_erp = np.mean(eeg[~is_target_event], axis=0)
    
    return sampled_target_erp, sampled_nontarget_erp

def test_statistic(target_erp_data, nontarget_erp_data):
    
    erp_difference = abs(target_erp_data-nontarget_erp_data)
    
    return erp_difference


def calculate_p_values(sampled_target_erp, sampled_nontarget_erp,target_erp, nontarget_erp,randomization_count=3000):
    
    sample_count = sampled_target_erp.shape[1]  # number of samples
    channel_count = sampled_target_erp.shape[2]  # number of channels
    
    # finding the test statistic
    real_erp_difference = test_statistic(target_erp, nontarget_erp) # identify real test statistic, the difference between the target_erp and nontarget_erp data 
    bootstrapped_erp_difference = test_statistic(sampled_target_erp, sampled_nontarget_erp) # identify bootsrapped test statistic, the difference between the sampled target and nontarget data
    
    # preallocate p-value array
    p_values = np.zeros([sample_count, channel_count])
    
    # perform p-value calculation on each sample of each channel
    for channel_index in range(channel_count):
        
        for sample_index in range(sample_count):
           
            # count number of bootstrapped test statistics that are greater than real test statistic
            difference_count = sum(bootstrapped_erp_difference[:, sample_index, channel_index] > real_erp_difference[sample_index, channel_index]) # apply to all 3000 randomizations at each sample index for each channel
            
            # calculate the p-value
            p_value = difference_count / randomization_count # find the proportion of bootstrapped statistics that are greater than the real test statistic
            
            # add the p-value to the preallocated array for the given sample number and channel
            p_values[sample_index, channel_index] = p_value
    
    return p_values

#%% Part D: Plot FDR-Corrected P Values

def plot_false_discovery_rate(eeg_epochs, erp_times, target_erp, nontarget_erp, is_target_event, p_values, subject=3, fdr_threshold=0.05):
    
    # identify necessary counts
    target_count = len(eeg_epochs[is_target_event])
    nontarget_count = len(eeg_epochs[~is_target_event])
    channel_count = eeg_epochs.shape[2]
    
    corrected_p_values = np.array(fdr_correction(p_values, alpha=fdr_threshold))
    
    significant_times = [[] for _ in range(channel_count)]
    
    # statistics calculated for given channel
    for channel_index in range(channel_count):
        
        # calculate statistics for target ERPs
        target_standard_deviation = np.std(eeg_epochs[is_target_event,:,channel_index], axis=0) # standard deviation of EEG data for target events at a given sample time point
        target_standard_error = target_standard_deviation / np.sqrt(target_count) # standard error for targets
        
        # calculate statistics for nontarget ERPs
        nontarget_standard_deviation = np.std(eeg_epochs[~is_target_event,:,channel_index], axis=0) # standard deviation of EEG data for nontarget events at a given sample time point
        nontarget_standard_error = nontarget_standard_deviation / np.sqrt(nontarget_count) # standard error for nontargets
    
    # transpose the erp data to plot, matches average at that sample time to the size of the time array
    target_erp_transpose = np.transpose(target_erp)
    nontarget_erp_transpose = np.transpose(nontarget_erp)
    
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
        
        # generate times to plot
        is_significant = np.array(np.where(corrected_p_values[0, :, channel_index] == 1)) # evaluate across all samples for a channel when  p_value < fdr_threshold is true
        #significant_times = []
        for significant_index in is_significant[0]:
            #significant_times.append(erp_times[significant_index])
            significant_times[channel_index].append(erp_times[significant_index])
        significant_count = len(np.array(significant_times[channel_index]).T)
        
        # plot significant points
        #significance_handle, = channel_plot.plot(significant_times, np.zeros(significant_count), 'ko', markersize=3)
        significance_handle, = channel_plot.plot(np.array(significant_times[channel_index]), np.zeros(significant_count), 'ko', markersize=3)
        
        # plot confidence intervals
        target_confidence_interval_handle = channel_plot.fill_between(erp_times,target_erp_transpose[channel_index] - 2 * target_standard_error, target_erp_transpose[channel_index] + 2 * target_standard_error, alpha=0.25)
        
        nontarget_confidence_interval_handle = channel_plot.fill_between(erp_times,nontarget_erp_transpose[channel_index] - 2 * nontarget_standard_error, nontarget_erp_transpose[channel_index] + 2 * nontarget_standard_error, alpha=0.25)
        
        # workaround for legend to only display each entry once
        if channel_index == 0:
            target_handle.set_label('Target')
            nontarget_handle.set_label('Nontarget')
            significance_handle.set_label('$p_{FDR}$ < 0.05')
            target_confidence_interval_handle.set_label('Target +/- 95% CI')
            nontarget_confidence_interval_handle.set_label('Nontarget +/- 95% CI')
        
        # label each plot's axes and channel number
        channel_plot.set_title(f'Channel {channel_index}')
        channel_plot.set_xlabel('time from flash onset (s)')
        channel_plot.set_ylabel('Voltage (μV)')
    
    # formatting
    figure.suptitle(f'P300 Speller S{subject} Training ERPs')
    figure.legend(loc='lower right', fontsize='x-large') # legend in space of nonexistent plot 9
    figure.tight_layout()  # stop axis labels overlapping titles
    
    # save image
    plt.savefig(f'P300_S{subject}_channel_plots_with_significance.png')  # save as image
    
    return significant_times

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

def multiple_subject_evaluation(subjects=np.arange(3,11), data_directory='P300Data/', sample_count=384, channel_count=8, epoch_start_time=-0.5, epoch_end_time=1.0, randomization_count=3000, fdr_threshold=0.05):
    
    # preallocate array for editing with each subject
    # set to empty list since necessary inputs obtained in subject loop for dynamic variables
    subject_significance = np.zeros([channel_count, sample_count])
    
    for subject in subjects:
        
        # loading data
        is_target_event, eeg_epochs, erp_times, target_erp, nontarget_erp = load_erp_data(subject, data_directory, epoch_start_time, epoch_end_time)
        
        # bootstrapping

        # preallocate arrays for resampled data 
        sampled_target_erp = np.zeros([randomization_count,sample_count,channel_count])
        sampled_nontarget_erp = np.zeros([randomization_count,sample_count,channel_count])

        # perform the bootstrapping
        for randomization_index in range(randomization_count):
                
            # resample targets
            sampled_target_erp[randomization_index,:] = bootstrap_erps(eeg_epochs, is_target_event)[0][:,:]
            
            # resample nontargets
            sampled_nontarget_erp[randomization_index,:] = bootstrap_erps(eeg_epochs, is_target_event)[1][:,:]
            
        # find p_values
        p_values = calculate_p_values(sampled_target_erp, sampled_nontarget_erp,target_erp, nontarget_erp,randomization_count)
        
        # FDR correction and plotting
        significant_times = plot_false_discovery_rate(eeg_epochs, erp_times, target_erp, nontarget_erp, is_target_event, p_values, subject, fdr_threshold)
        
        # track number of subjects where a point in time is significant for each channel
        # do this with boolean indexing: true if the sample in time for that subject and channel is significant
        # use np.sum to track total, likely want (384,8) array
        # return the sum, take into next function
        
        # for significant_index in is_significant[0]:
        #     significant_times.append(erp_times[significant_index])
        # significant_count = len(np.array(significant_times))
        
        # add 1 to count at a sample index if a subject has a value for that time
        # for a time in erp_times
            # if that time is also in significant times
                # add 1 to that time point in subject_significance
        
        for time_index in range(sample_count):
            for channel_index in range(channel_count):
                for i in range(len(significant_times[channel_index])):
                    if significant_times[channel_index][i] == erp_times[time_index]:
                        subject_significance[channel_index,time_index] = subject_significance[channel_index,time_index]+1
                        
    return erp_times, subject_significance
            

def plot_subject_significance(erp_times, subject_significance):
    
    channel_count = len(subject_significance)
    
    figure, channel_plots = plt.subplots(3,3, figsize=(10, 6), sharey=True)
    
    channel_plots[2][2].remove()  # only 8 channels, 9th plot unnecessary
    
    for channel_index in range(channel_count):
        
        row_index, column_index = divmod(channel_index, 3)  # wrap around to column 0 for every 3 plots
        
        channel_plot = channel_plots[row_index][column_index] # subplot
        
        channel_plot.plot(erp_times, subject_significance[channel_index])
        
        # label each plot's axes and channel number
        channel_plot.set_title(f'Channel {channel_index}')
        channel_plot.set_xlabel('time (s)')
        channel_plot.set_ylabel('# subjects significant')
        channel_plot.grid(True)
    
    # formatting
    figure.suptitle('Number of Significant Subjects by Channel')
    figure.tight_layout()  # stop axis labels overlapping titles
    
    # save image
    plt.savefig('subject_significance_channel_plots.png')  
        

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
    
def plot_spatial_map(eeg_epochs, is_target_event, subject=3):
    
    median_target_erp = np.median(eeg_epochs[is_target_event], axis=(0,1))
    median_nontarget_erp = np.median(eeg_epochs[~is_target_event], axis=(0,1))
    
    plot_topo.plot_topo(['Fz', 'Cz', 'P3', 'Pz', 'P4', 'P7', 'P8', 'Oz'], median_target_erp, f'Subject{subject} P300 Spatial Map');
    
        