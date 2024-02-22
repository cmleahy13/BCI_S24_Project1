#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 12 22:23:10 2024

plot_p300_data.py

This file serves as the module for Project 1 in BCI-S24. The ultimate purpose of this script is to generate plots that depict ERP data from a P300 Speller with 95% confidence intervals (under the assumption of a normal distribution) and number of subjects that are significant for a particular sample in a given channel (where the data for the test statistic is bootstrapped, creating a normal distribution of means). These plots generate observable patterns pertaining to the usefulness of a P300 Speller as a potentially effective BCI. In addition, a function generating scalp maps for the ERP data is included, which provides a visual interpretation of channel location spatially (as voltage from ERPs are color-coded).

Relevant abbreviations:
    EEG: electroencephalography
    ERP: event-related potential
    FDR: false discovery rate

@authors: Varshney Gentela and Claire Leahy

Sources:
    
    Nick Bosley gave tips on approach to standard deviation calculations
    
"""
#%% Imports

# import lab modules (Part A, Part D)
import numpy as np
from matplotlib import pyplot as plt
from load_p300_data import load_training_eeg
from plot_p300_erps import get_events, epoch_data, get_erps
from mne.stats import fdr_correction
from plot_topo import plot_topo, get_channel_names
import itertools
#%% Part A: Load and Epoch the Data

def load_erp_data(subject=3, data_directory='P300Data/', epoch_start_time=-0.5, epoch_end_time=1.0):
    """
    Description
    -----------
    This function loads the training data for a given subject from a given directory using the load_training_eeg function defined in an earlier module. After obtaining the relevant EEG data for the subject, this function also gets the samples where events (flashes) occur using get_events as well as the epochs (using epoch_data), which is a designated amount of time surrounding an event occurrence as defined by epoch_start_time and epoch_end_time. Finally, the ERP data, or mean EEG data, is obtained using get_erps. The function returns were identified based on the needs of subsequent functions, primarily those involved in plotting the EEG and/or ERP data.

    Parameters
    ----------
    subject : int, optional
        Input to define the subject to be evaluated. The default is 3.
    data_directory : str, optional
        Input string directory to the location of the data files. The default is 'P300Data/'.
    epoch_start_time : float, optional
        Beginning of relative range to collect samples around an event, in seconds. The default is -0.5.
    epoch_end_time : float, optional
        Ending of relative range to collect samples around an event, in seconds. The default is 1.0.

    Returns
    -------
    is_target_event : Ex1 Boolean array, where E is the number of samples in which an event occurred
        Array holding truth data pertaining to whether each event that occurred was a target (True) or nontarget (False).
    eeg_epochs : ExSxC array of floats, where E is the number of epochs, S is the number of samples in each epoch, and C is the number of channels
        Array containing the sample EEG data from each channel that occurs at each event (epoch).
    erp_times : Sx1 array of floats, where S is the number of samples in each epoch
        Array containing the times of each sample relative to the event onset in seconds.
    target_erp : SxC array of floats, where S is the number of samples in each epoch, and C is the number of channels
        Array containing mean EEG data at each sample point in time for each channel for epochs that constitute targets.
    nontarget_erp : SxC array of floats, where S is the number of samples in each epoch, and C is the number of channels
        Array containing mean EEG data at each sample point in time for each channel for epochs that constitute nontargets.

    """
    
    # load in training data
    eeg_time, eeg_data, rowcol_id, is_target = load_training_eeg(subject, data_directory)
    
    
    # extract target and nontarget epochs
    event_sample, is_target_event = get_events(rowcol_id, is_target) # find events, which events are targets
    eeg_epochs, erp_times = epoch_data(eeg_time, eeg_data, event_sample, epoch_start_time, epoch_end_time) # generate epochs given events, corresponding relative tiems
    
    # calculate ERPs
    target_erp, nontarget_erp = get_erps(eeg_epochs, is_target_event) # get ERPs (mean EEG data)
    
    
    return is_target_event, eeg_epochs, erp_times, target_erp, nontarget_erp

#%% Part B: Calculate & Plot Parametric Confidence Intervals

def plot_confidence_intervals(eeg_epochs, erp_times, target_erp, nontarget_erp, is_target_event, subject=3):
    """
    Description
    -----------
    This function is utilized to plot the ERP data as well as the 95% confidence intervals of the EEG data. To plot the confidence intervals, it is necessary to calculate standard error (found by obtaining the standard deviation of the EEG data at target and nontarget epochs). By calculating standard error, a normal distribution is assumed; however, it is important to note that the data is likely not normally distributed. After calculating the standard error, the 95% confidence intervals are calculated roughly be computing the (ERP data) +/- (2*standard error) for target and nontarget events.

    Parameters
    ----------
    eeg_epochs : ExSxC array of floats, where E is the number of epochs, S is the number of samples in each epoch, and C is the number of channels
        Array containing the sample EEG data from each channel that occurs at each event (epoch).
    erp_times : Sx1 array of floats, where S is the number of samples in each epoch
        Array containing the times of each sample relative to the event onset in seconds.
    target_erp : SxC array of floats, where S is the number of samples in each epoch, and C is the number of channels
        Array containing mean EEG data at each sample point in time for each channel for epochs that constitute targets.
    nontarget_erp : SxC array of floats, where S is the number of samples in each epoch, and C is the number of channels
        Array containing mean EEG data at each sample point in time for each channel for epochs that constitute nontargets.
    is_target_event : Ex1 Boolean array, where E is the number of samples in which an event occurred
        Array holding truth data pertaining to whether each event that occurred was a target (True) or nontarget (False).
    subject : int, optional
        Input to define the subject to be evaluated. The default is 3.

    Returns
    -------
    None.

    """
    
    # identify necessary counts for easy sizing and indexing
    target_count = len(eeg_epochs[is_target_event])
    nontarget_count = len(eeg_epochs[~is_target_event])
    channel_count = eeg_epochs.shape[2]
    
    # Printing counts to make sure they are the numbers that we want 
    print("shape of target_count to find the target_standard error:", target_count)
    print("shape of target_count to find the nontarget_standard error:", nontarget_count)
    print("channel_count for indexing and plotting the confidence intervals:", channel_count)
    
    # plot ERPs for events for each channel
    figure, channel_plots = plt.subplots(3,3, figsize=(10, 6))
    channel_plots[2][2].remove()  # 9th plot unnecessary
   
    for channel_index in range(channel_count):
        
        # calculate statistics for target ERPs
        target_standard_deviation = np.std(eeg_epochs[is_target_event,:,channel_index], axis=0) # standard deviation of EEG data for target events at a given sample time point
        target_standard_error = target_standard_deviation / np.sqrt(target_count) # standard error for targets
        
        # calculate statistics for nontarget ERPs
        nontarget_standard_deviation = np.std(eeg_epochs[~is_target_event,:,channel_index], axis=0) # standard deviation of EEG data for nontarget events at a given sample time point
        nontarget_standard_error = nontarget_standard_deviation / np.sqrt(nontarget_count) # standard error for nontargets
        
        # plot accessing
        row_index, column_index = divmod(channel_index, 3)  # wrap around to column 0 for every 3 plots
        
        channel_plot = channel_plots[row_index][column_index] # subplot
        
        # plot dotted lines for time 0 and 0 voltage
        channel_plot.axvline(0, color='black', linestyle='dotted')
        channel_plot.axhline(0, color='black', linestyle='dotted')
        
        # plot ERP data in the subplot
        target_handle, = channel_plot.plot(erp_times, target_erp.T[channel_index])
        
        nontarget_handle, = channel_plot.plot(erp_times, nontarget_erp.T[channel_index])
        
        # plot confidence intervals
        target_confidence_interval_handle = channel_plot.fill_between(erp_times,target_erp.T[channel_index] - 2 * target_standard_error, target_erp.T[channel_index] + 2 * target_standard_error, alpha=0.25)
        
        nontarget_confidence_interval_handle = channel_plot.fill_between(erp_times,nontarget_erp.T[channel_index] - 2 * nontarget_standard_error, nontarget_erp.T[channel_index] + 2 * nontarget_standard_error, alpha=0.25)
        
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
    """
    Description
    -----------
    This function begins by generating random indices that will be used to sample from the EEG data with replacement to get a normal distribution of means (mean EEG data, therefore ERP data) per the central limit theorem. The randomly resampled EEG data is put into an array that represents eeg_epochs sampled at those random indices. Under the null hypothesis that there is no difference between target and nontarget ERP data, the random samples are pooled, and to test whether there is any difference between target and nontarget data, the same proportion of target-to-nontarget data is applied when generating the sampled target and nontarget data (i.e. the sampled target ERP data is the mean of the sampled EEG data at indices where there is a target event, even though there is theoretically no difference at these indices under the null hypothesis). Sampled target and nontarget data are returned for use in calculating relevant test statistics.

    Parameters
    ----------
    eeg_epochs : ExSxC array of floats, where E is the number of epochs, S is the number of samples in each epoch, and C is the number of channels
        Array containing the sample EEG data from each channel that occurs at each event (epoch).
    is_target_event : Ex1 Boolean array, where E is the number of samples in which an event occurred
        Array holding truth data pertaining to whether each event that occurred was a target (True) or nontarget (False).

    Returns
    -------
    sampled_target_erp : RxSxC array of floats, where R is the number of randomizations, S is the number of samples per epoch, and C is the number of channels
        Array containing the bootstrapped iterations for target indices (not inherently target data due to resampling).
    sampled_nontarget_erp : RxSxC array of floats, where R is the number of randomizations, S is the number of samples per epoch, and C is the number of channels
        Array containing the bootstrapped iterations for nontarget indices (not inherently nontarget data due to resampling).

    """
    
    # get event count for number of random indices
    event_count = eeg_epochs.shape[0]
    
    # sample random indices for each channel separately
    random_indices = np.random.randint(event_count, size=event_count)
   
    
    # randomly sampled EEG data
    eeg = eeg_epochs[random_indices]
    
    # get ERP by taking the mean of the resampled EEG data
    sampled_target_erp = np.mean(eeg[is_target_event], axis=0) # assumption of no difference: use target indices on randomly sampled data 
    sampled_nontarget_erp = np.mean(eeg[~is_target_event], axis=0) # assumption of no difference: use nontarget indices on randomly sampled data 
    
    return sampled_target_erp, sampled_nontarget_erp

def test_statistic(target_erp_data, nontarget_erp_data):
    """
    Description
    -----------
    This function calculates the test statistic of interest, which is the absolute value of the difference between target and nontarget ERP data.

    Parameters
    ----------
    target_erp_data : RxSxC array of floats, where R is the number of randomizations (N/A for real data, number of bootstrap iterations for bootstrapped data), S is the number of samples per epoch, and C is the number of channels
        Averaged EEG data over epochs (ERP) data for target events.
    nontarget_erp_data : RxSxC array of floats, where R is the number of randomizations (N/A for real data, number of bootstrap iterations for bootstrapped data), S is the number of samples per epoch, and C is the number of channels
        Averaged EEG data over epochs (ERP) data for nontarget events.

    Returns
    -------
    erp_difference : RxSxC array of floats, 
    target_erp_data : RxSxC array of floats, where R is the number of randomizations (N/A for real data, number of bootstrap iterations for bootstrapped data), S is the number of samples per epoch, and C is the number of channels
        The absolute value of the difference between target and nontarget ERP data at each sample for each channel.

    """
    
    # absolute value of the difference between target and nontarget data
    erp_difference = abs(target_erp_data-nontarget_erp_data)
    
    return erp_difference


def calculate_p_values(sampled_target_erp, sampled_nontarget_erp,target_erp, nontarget_erp,randomization_count=3000):
    """
    Description
    -----------
    This function uses the test statistics - the absolute value of the difference between target and nontarget ERP data, both resampled and real - to calculate p-values. Since p-values represent the proportion of bootstrapped statistics are greater than the real statistics, a count where this condition holds is divided by the number of randomizations performed. To effectively do this, the condition must be checked across each of the randomizations. The result is an array that contains a p-value at each sample for each channel.

    Parameters
    ----------
    sampled_target_erp : RxSxC array of floats, where R is the number of randomizations, S is the number of samples per epoch, and C is the number of channels
        Array containing the bootstrapped iterations for target indices (not inherently target data due to resampling).
    sampled_nontarget_erp : RxSxC array of floats, where R is the number of randomizations, S is the number of samples per epoch, and C is the number of channels
        Array containing the bootstrapped iterations for nontarget indices (not inherently nontarget data due to resampling).
    target_erp : SxC array of floats, where S is the number of samples in each epoch, and C is the number of channels
        Array containing mean EEG data at each sample point in time for each channel for epochs that constitute targets.
    nontarget_erp : SxC array of floats, where S is the number of samples in each epoch, and C is the number of channels
        Array containing mean EEG data at each sample point in time for each channel for epochs that constitute nontargets.
    randomization_count : int, optional
        The number of times the bootstrapping procedure is performed. The default is 3000.

    Returns
    -------
    p_values : SxC array of floats, where S is the number of samples per epoch and C is the number of channels
        Array containing p-values based on the test statistic calculated for the real data and the bootstrapped data.

    """
    
    # counts for easy access
    sample_count = sampled_target_erp.shape[1]  # number of samples
    channel_count = sampled_target_erp.shape[2]  # number of channels
    
    # finding the test statistic
    real_erp_difference = test_statistic(target_erp, nontarget_erp) # identify real test statistic, difference between the target_erp and nontarget_erp data 
    bootstrapped_erp_difference = test_statistic(sampled_target_erp, sampled_nontarget_erp) # identify bootsrapped test statistic, difference between the sampled target and nontarget data
    
   
    # preallocate p-value array
    p_values = np.zeros([sample_count, channel_count])
    
    # perform p-value calculation on each sample of each channel
    for channel_index in range(channel_count):
        
        for sample_index in range(sample_count):
           
            # count number of bootstrapped test statistics that are greater than real test statistic
            difference_count = sum(bootstrapped_erp_difference[:, sample_index, channel_index] > real_erp_difference[sample_index, channel_index]) # apply to all 3000 randomizations at each sample index for each channel
            
            # calculate the p-value
            p_value = difference_count / randomization_count # proportion of bootstrapped statistics greater than real test statistic
            
            # add p-value to the preallocated array for the given sample number and channel
            p_values[sample_index, channel_index] = p_value
    
    return p_values

#%% Part D: Plot FDR-Corrected P Values

def plot_false_discovery_rate(eeg_epochs, erp_times, target_erp, nontarget_erp, is_target_event, p_values, subject=3, fdr_threshold=0.05):
    """
    Description
    -----------
    Correcting for false discovery rate implements a method that limits false positives being reported as significant. This effectively increases the p-values, potentially decreasing the significance observed in a sample of a given channel. This function employs false discovery rate correction to find where the data is potentially significant under stricter conditions. After the corrected p-values are found, their values are compared with the fdr_threshold value alpha (0.05 in this case); when the p-values are found to be lower than alpha, the sample point is deemed significant (for that given channel). These indices make up a sub-portion of erp_times, and a significant time can be obtained by indexing the significant index in erp_times. A black dot is plotted along the x-axis of a channel subplot on the sample time for each occasion of significance. An array where the sample times are significant in each channel is returned for future use in counting the number of subjects that experience significant differences at those times.
    
    Parameters
    ----------
    eeg_epochs : ExSxC array of floats, where E is the number of epochs, S is the number of samples in each epoch, and C is the number of channels
        Array containing the sample EEG data from each channel that occurs at each event (epoch).
    erp_times : Sx1 array of floats, where S is the number of samples in each epoch
        Array containing the times of each sample relative to the event onset in seconds.
    target_erp : SxC array of floats, where S is the number of samples in each epoch, and C is the number of channels
        Array containing mean EEG data at each sample point in time for each channel for epochs that constitute targets.
    nontarget_erp : SxC array of floats, where S is the number of samples in each epoch, and C is the number of channels
        Array containing mean EEG data at each sample point in time for each channel for epochs that constitute nontargets.
    is_target_event : Ex1 Boolean array, where E is the number of samples in which an event occurred
        Array holding truth data pertaining to whether each event that occurred was a target (True) or nontarget (False).
    p_values : SxC array of floats, where S is the number of samples per epoch and C is the number of channels
        Array containing p-values based on the test statistic calculated for the real data and the bootstrapped data.
    subject : int, optional
        Input to define the subject to be evaluated. The default is 3.
    fdr_threshold : float, optional
        The value of alpha used as the threshold value to determine signficance. The default is 0.05.

    Returns
    -------
    significant_times : list of size C, where C is the number of channels
        List of arrays containing time points relative to event onset where the p-value was lower than fdr_threshold for a given subject where. The array within each channel is size Sx1, where S is the number of samples that are significant for that channel.

    """
    
    # identify counts for easy access
    target_count = len(eeg_epochs[is_target_event])
    nontarget_count = len(eeg_epochs[~is_target_event])
    channel_count = eeg_epochs.shape[2]
    
    # generate array of corrected p_values using fdr_correction
    corrected_p_values = np.array(fdr_correction(p_values, alpha=fdr_threshold))
    
    # create a list that will hold significant times for each channel
    significant_times = [[] for _ in range(channel_count)]
    
    # plot ERPs for events for each channel
    figure, channel_plots = plt.subplots(3,3, figsize=(10, 6))
    channel_plots[2][2].remove()  # 9th plot unnecessary
   
    for channel_index in range(channel_count):
        
        # calculate statistics for target ERPs
        target_standard_deviation = np.std(eeg_epochs[is_target_event,:,channel_index], axis=0) # standard deviation of EEG data for target events at a given sample time point
        target_standard_error = target_standard_deviation / np.sqrt(target_count) # standard error for targets
        
        # calculate statistics for nontarget ERPs
        nontarget_standard_deviation = np.std(eeg_epochs[~is_target_event,:,channel_index], axis=0) # standard deviation of EEG data for nontarget events at a given sample time point
        nontarget_standard_error = nontarget_standard_deviation / np.sqrt(nontarget_count) # standard error for nontargets
        
        # plot access
        row_index, column_index = divmod(channel_index, 3)  # wrap around to column 0 for every 3 plots
        
        channel_plot = channel_plots[row_index][column_index] # subplot
        
        # plot dotted lines for time 0 and 0 voltage
        channel_plot.axvline(0, color='black', linestyle='dotted')
        channel_plot.axhline(0, color='black', linestyle='dotted')
        
        # plot target and nontarget ERP data in the subplot
        target_handle, = channel_plot.plot(erp_times, target_erp.T[channel_index])
        nontarget_handle, = channel_plot.plot(erp_times, nontarget_erp.T[channel_index])
        
        # generate times to plot
        is_significant = np.array(np.where(corrected_p_values[0, :, channel_index] == 1)) # evaluate across all samples for a channel when  p_value < fdr_threshold
    
        
        for significant in is_significant[0]: # this index [0] is the boolean index
            
            significant_times[channel_index].append(erp_times[significant]) # add the list of significant time points (conversion obtained by indexing erp_times) to significant_times in the channel index
        significant_count = len(np.array(significant_times[channel_index]).T) # count how many values are significant to size plot of zeros
        
        # plot significant points
        significance_handle, = channel_plot.plot(np.array(significant_times[channel_index]), np.zeros(significant_count), 'ko', markersize=3) # plot the significant times for the channel as zeros along x-axis
        
        # plot confidence intervals
        target_confidence_interval_handle = channel_plot.fill_between(erp_times,target_erp.T[channel_index] - 2 * target_standard_error, target_erp.T[channel_index] + 2 * target_standard_error, alpha=0.25)
        
        nontarget_confidence_interval_handle = channel_plot.fill_between(erp_times,nontarget_erp.T[channel_index] - 2 * nontarget_standard_error, nontarget_erp.T[channel_index] + 2 * nontarget_standard_error, alpha=0.25)
        
        # limit legend display
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

def multiple_subject_evaluation(subjects=np.arange(3,11), data_directory='P300Data/', sample_count=384, channel_count=8, epoch_start_time=-0.5, epoch_end_time=1.0, randomization_count=3000, fdr_threshold=0.05):
    """
    Description
    -----------
    This function loops through the functions defined above for each subject (with the exception of plot_confidence_intervals since plot_false_discovery_rate has very similar functionality in addition to adding significant points.) After each function is called, the number of subjects in which a given sample of a given channel exhibits significance is counted.  

    Parameters
    ----------
    subjects : Px1 array of integers, where P is the subject number, optional
        Array containing the subject numbers to evaluate. The default is np.arange(3,11).
    data_directory : str, optional
        Input string directory to the location of the data files. The default is 'P300Data/'.
    sample_count : int, optional
        The number of samples per epoch, taken as an argument due to necessary use before dynamic coding possible. The default is 384.
    channel_count : int, optional
        The number of channels, taken as an argument due to necessary use before dynamic coding possible. The default is 8.
    epoch_start_time : float, optional
        Beginning of relative range to collect samples around an event, in seconds. The default is -0.5.
    epoch_end_time : float, optional
        Ending of relative range to collect samples around an event, in seconds. The default is 1.0.
    randomization_count : int, optional
        The number of times the bootstrapping procedure is performed. The default is 3000.
    fdr_threshold : float, optional
        The value of alpha used as the threshold value to determine signficance. The default is 0.05.
        
    Returns
    -------
    erp_times : Sx1 array of floats, where S is the number of samples in each epoch
        Array containing the times of each sample relative to the event onset in seconds.
    subject_significance : CxS array of floats, where C is the number of channels and S is the number of samples per epoch
        Array containing the number of subjects that possess statistically significant data (the corrected p-value is less than the fdr_threshold value) for each sample in each channel.

    """
    
    # preallocate array for editing with each subject
    subject_significance = np.zeros([channel_count, sample_count])
    
    for subject in subjects: # for each subject defined in the input
        
        # load data
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
        
        # counting the number of subjects that are significant for a given sample on a given channel
        for time_index in range(sample_count): # index itself will be a sample number, but used to find the corresponding relative time
        
            for channel_index in range(channel_count): # for each channel
            
                for significant_index in range(len(significant_times[channel_index])): # for each significant time point in a channel
                    if significant_times[channel_index][significant_index] == erp_times[time_index]: # compare if the time matches the erp_times index
                        subject_significance[channel_index,time_index] = subject_significance[channel_index,time_index]+1 # if the times do match, add 1 each time a subject is significant for that channel's sample
                        

    return erp_times, subject_significance

def plot_subject_significance(erp_times, subject_significance):
    """
    Description
    -----------
    Using the data from the function above where the number of significant subjects in a given sample of a given channel is counted, a plot (for each channel) depicting these counts is generated. 

    Parameters
    ----------
    erp_times : Sx1 array of floats, where S is the number of samples in each epoch
        Array containing the times of each sample relative to the event onset in seconds.
    subject_significance : CxS array of floats, where C is the number of channels and S is the number of samples per epoch
        Array containing the number of subjects that possess statistically significant data (the corrected p-value is less than the fdr_threshold value) for each sample in each channel.
        
    Returns
    -------
    None.

    """
    
    # find number of channels to plot
    channel_count = len(subject_significance[1])
    
    # generate figure
    figure, channel_plots = plt.subplots(3,3, figsize=(10, 6), sharey=True)
    
    channel_plots[2][2].remove()  # 9th plot unnecessary
    
    for channel_index in range(channel_count): # perform for each channel
        
        row_index, column_index = divmod(channel_index, 3)  # wrap around to column 0 for every 3 plots
        
        channel_plot = channel_plots[row_index][column_index] # subplot

        # plot the number of subjects that are significant for a channel's sample
        channel_plot.plot(erp_times, subject_significance[1][channel_index])
        
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

def plot_spatial_map(eeg_epochs, is_target_event, erp_times, subjects=np.arange(3,11), data_directory='P300Data/'):
    
    """
    Description
    -----------
    This function plots spatial maps for the median target and nontarget ERPs (Event-Related Potentials)
    across all subjects for N2 (200-300 ms) and P3b (300-500 ms) time ranges.

    Parameters
    ----------
    eeg_epochs : ExSxC array of floats
        EEG data from each channel at each event (epoch), where E is the number of epochs, S is the number of samples in each epoch, and C is the number of channels.
    is_target_event : Ex1 Boolean array
        Array indicating whether each event is a target (True) or nontarget (False).
    erp_times : Sx1 array of floats
        Array containing the times of each sample relative to the event onset in seconds.
    subjects : array-like, optional
        Subjects to include. Default is np.arange(3,11).
    data_directory : str, optional
        Directory where the ERP data is stored. Default is 'P300Data/'.

    Returns
    -------
    None
    """
    # declare lists for storing ERP data for each subject
    individual_median_erp_target = []
    individual_median_erp_nontarget = []    
    # convert erp_times to an array
    #erp_times = np.array(erp_times)
    for subject in np.arange(4,6):
        
        # load the erp data for each subject
        is_target_event, eeg_epochs, erp_times, target_erp, nontarget_erp = load_erp_data(subject, data_directory,  epoch_start_time=-0.5, epoch_end_time=1.0)
        median_target_eeg = np.median(eeg_epochs[is_target_event], axis=0)
        median_nontarget_eeg = np.median(eeg_epochs[~is_target_event], axis=0)
        erp_times = np.array(erp_times)
        n2_time_range = (0.2, 0.3)  # 200-300 ms
        n2_start_time, n2_end_time = n2_time_range
        n2_start_index = np.where(erp_times >= n2_start_time)[0][0]
        n2_end_index = np.where(erp_times<=n2_end_time)[0][-1]
        
        p3b_time_range = (0.3, 0.5)  # 300-500 ms
        p3b_start_time, p3b_end_time = p3b_time_range
        p3b_start_index = np.where(erp_times >=p3b_start_time)[0][0]
        p3b_end_index = np.where(erp_times <= p3b_end_time)[0][-1]
        
        # extract target and nontarget ERPs for N2 and P3b time ranges
        target_n2 = median_target_eeg[n2_start_index:n2_end_index +1,:]
        target_p3b = median_target_eeg[p3b_start_index:p3b_end_index + 1,:]
        
        nontarget_n2 = median_nontarget_eeg[n2_start_index:n2_end_index +1,:]
        nontarget_p3b = median_nontarget_eeg[p3b_start_index:p3b_end_index +1,:]
       
        # make channel array to edit order once 
        channel_array = ['P4','Pz','Cz','Fz','P8','Oz','P3','P7']        
       
        # plot and save topomaps for N2 and P3b time ranges
        plt.figure(figsize=(10, 6))
        plot_topo(channel_array, channel_data=target_n2.T, title='Group Median Target N2', montage_name='standard_1020')
        plt.savefig(f'target_n2_topomap_subject_{subject}.png') 
        
        plt.figure(figsize=(10, 6))
        plot_topo(channel_array, channel_data=nontarget_n2.T, title='Group Median Nontarget N2')
        plt.savefig(f'nontarget_n2_topomap_subject_{subject}.png') 
            
        plt.figure(figsize=(10, 6))
        plot_topo(channel_array, channel_data=target_p3b.T, title='Group Median Target P3b')
        plt.savefig(f'target_p3b_topomap_subject_{subject}.png') 
        
        plt.figure(figsize=(10, 6))
        plot_topo(channel_array, channel_data=nontarget_p3b.T, title='Group Median Nontarget P3b')
        plt.savefig(f'nontarget_p3b_topomap_subject_{subject}.png') 
            


    # find individual median ERP data
    for subject in subjects: # for each subject
        
        # load the erp data for each subject
        is_target_event, eeg_epochs, erp_times, target_erp, nontarget_erp = load_erp_data(subject, data_directory,  epoch_start_time=-0.5, epoch_end_time=1.0)
 
        # add the median ERP data for the subject to the list
        individual_median_erp_target.append(np.median(eeg_epochs[is_target_event],axis = 0))
        
        individual_median_erp_nontarget.append(np.median(eeg_epochs[~is_target_event],axis =0))
    erp_times = np.array(erp_times)
    # convert from list to array
    individual_median_erp_target = np.array(individual_median_erp_target)
    individual_median_erp_nontarget = np.array(individual_median_erp_nontarget)
    
    # print to observe shape of array
    print("The shape of the median ERP for individual subject:", individual_median_erp_target.shape) # (subject_count, sample_count, channel_count)
    
    # calculate group median ERPs
    group_median_erp_target = np.median(individual_median_erp_target, axis=0)
    group_median_erp_nontarget = np.median(individual_median_erp_nontarget, axis=0)
   
    # get time ranges
    
    
    n2_time_range = (0.2, 0.3)  # 200-300 ms
    n2_start_time, n2_end_time = n2_time_range
    n2_start_index = np.where(erp_times >= n2_start_time)[0][0]
    n2_end_index = np.where(erp_times<=n2_end_time)[0][-1]
    
    p3b_time_range = (0.3, 0.5)  # 300-500 ms
    p3b_start_time, p3b_end_time = p3b_time_range
    p3b_start_index = np.where(erp_times >=p3b_start_time)[0][0]
    p3b_end_index = np.where(erp_times <= p3b_end_time)[0][-1]
    
    # extract target and nontarget ERPs for N2 and P3b time ranges
    target_n2 = group_median_erp_target[n2_start_index:n2_end_index +1,:]
    target_p3b = group_median_erp_target[p3b_start_index:p3b_end_index + 1,:]
    
    nontarget_n2 = group_median_erp_nontarget[n2_start_index:n2_end_index +1,:]
    nontarget_p3b = group_median_erp_nontarget[p3b_start_index:p3b_end_index +1,:]
   
    # make channel array to edit order once
    channel_array = ['P4','Pz','Cz','Fz','P8','Oz','P3','P7']
    # plot and save topomaps for N2 and P3b time ranges
    plt.figure(figsize=(10, 6))
    plot_topo(channel_array, channel_data=target_n2.T, title='Group Median Target N2', montage_name='standard_1020')
    plt.savefig('target_n2_topomap.png') 
    
    plt.figure(figsize=(10, 6))
    plot_topo(channel_array, channel_data=nontarget_n2.T, title='Group Median Nontarget N2')
    plt.savefig('nontarget_n2_topomap.png') 
        
    plt.figure(figsize=(10, 6))
    plot_topo(channel_array, channel_data=target_p3b.T, title='Group Median Target P3b')
    plt.savefig('target_p3b_topomap.png') 
    
    plt.figure(figsize=(10, 6))
    plot_topo(channel_array, channel_data=nontarget_p3b.T, title='Group Median Nontarget P3b')
    plt.savefig('nontarget_p3b_topomap.png') 
    
    return None
