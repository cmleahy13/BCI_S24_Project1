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
from plot_topo import plot_topo,get_channel_names

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
    
    return erp_times, is_target_event, target_erp, nontarget_erp, eeg_epochs

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
def plot_confidence_intervals(eeg_epochs,is_target_event,erp_times, target_erp, nontarget_erp,subject =3) :
      
    
    meantarget_erp = np.mean(target_erp, axis=1)  # Compute the mean ERP across trials
    std_erp = np.std(target_erp, axis=1)  # Compute the standard deviation across trials
    sdmn1 = std_erp / np.sqrt(len(eeg_epochs[is_target_event]))  # Standard error of the mean
    mean_nontarget_erp = np.mean(nontarget_erp, axis=1) # Compute the mean ERP across trials
    std_erp2 = np.std(nontarget_erp, axis=1)  # Compute the standard deviation across trials
    sdmn2 = std_erp2 / np.sqrt(len(eeg_epochs[~is_target_event]))  # Standard error of the mean
    
    
    # Plot ERPs and confidence intervals for each channel
    fig, axs = plt.subplots(nrows=3, ncols=3, figsize=(12, 10))
    axs = axs.flatten()  # Flatten the 2D array of subplots
    
    for i in range(8):
        ax = axs[i]
        #ax.plot(erp_times[i], meantarget_erp[i], label='Mean Target ERP', color='teal')
        #ax.plot(erp_times[i], mean_nontarget_erp[i], label='Mean Non Target ERP', color='pink')
        ax.fill_between(erp_times, meantarget_erp - 2 * sdmn1, meantarget_erp + 2 * sdmn1,  alpha=0.2, label = 'Target confidence intervals' , lw = 5)
        ax.fill_between(erp_times, mean_nontarget_erp - 2 * sdmn2, mean_nontarget_erp + 2 * sdmn2, alpha=0.2, label = 'Nontarget confidence intervals', lw = 200)
        ax.plot(erp_times, target_erp[:, i], label='Target ERP' )
        ax.plot(erp_times, nontarget_erp[:, i], label='Non-target ERP')
        ax.axvline(0, color='black', linestyle='--')  # Mark stimulus onset
        ax.axhline(0, color='black', linestyle=':')  # Mark zero voltage
        ax.set_title(f'Channel {i+1}')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Voltage (μV)')
        
        
        plt.tight_layout()
        
    fig.suptitle(f'P300 Speller S{subject} Training ERPs')
    fig.tight_layout()
    ax.legend(loc = 'lower right')    
    fig.delaxes(axs[8])
    plt.show()

"""
<<<<<<< Updated upstream
def plot_erps_and_intervals(erp_times, target_erp, nontarget_erp):
    # would it make sense to call plot_erps here?
    

=======
        """
erp_times,is_target_event, target_erp, nontarget_erp, eeg_epochs = load_erp_data(subject=3,data_directory='P300Data/',epoch_start_time=-0.5, epoch_end_time=1.0)
plot_confidence_intervals(eeg_epochs,is_target_event,erp_times, target_erp, nontarget_erp)
#>>>>>>> Stashed changes
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
def bootstrapping(eeg_epochs, is_target_event, target_erp, nontarget_erp, size=None, iterations=3000):
    mean_target = np.mean(target_erp, 1)
    nontarget_mean = np.mean(nontarget_erp, 1)
    meandiff = mean_target - nontarget_mean
    stat = max(abs(meandiff))
    print('stat = {:.4f}'.format(stat))
    eeg_stack = np.vstack((target_erp, nontarget_erp))
    #np.random.seed(123)
    max_abs_diff_samples = []
    for _ in range(iterations): 
        ntrials = len(target_erp)
        if size is None:
            size = ntrials
        i = np.random.randint(ntrials, size=size)
        eeg0 = eeg_stack[i]
        mean_stacked_epochs = eeg0.mean(1)
        
        i = np.random.randint(ntrials, size=size)
        eeg1 = eeg_stack[i]
        mean_stacked_epochs_two = eeg1.mean(1)
        
        random_mean_diff = np.subtract(mean_stacked_epochs, mean_stacked_epochs_two)
        max_abs_diff_samples.append(np.max(np.abs(random_mean_diff)))
    
    return max_abs_diff_samples, stat

def calculate_p_values(eeg_epochs, is_target_event, target_erp, nontarget_erp, size=None, iterations=3000):
    # Perform bootstrapping to obtain the distribution of differences
    print(eeg_epochs.shape[0])
    max_abs_diff_samples, observed_statistic = bootstrapping(eeg_epochs, is_target_event, target_erp, nontarget_erp, size, iterations)
    
    # Calculate p-values
    p_values = []
    for i in range(eeg_epochs.shape[2]):  # Iterate over channels
        for j in eeg_epochs[0]:  # Iterate over time points
            # Count how many bootstrapped samples have a statistic greater than or equal to the observed statistic
            count = sum(diff >= observed_statistic for diff in max_abs_diff_samples)
            # Calculate the p-value as the proportion of samples with a statistic greater than or equal to the observed statistic
            p_value = count / iterations
            p_values.append(p_value)
    
    # Reshape p-values array to match the shape of the target_erp and nontarget_erp arrays
    p_values = np.array(p_values).reshape(eeg_epochs[is_target_event].shape[1:])
    
    return p_values

# Perform bootstrapping and obtain max_abs_diff samples and initial stat value
max_abs_diff_samples, initial_stat = bootstrapping(eeg_epochs, is_target_event, target_erp, nontarget_erp)
p_values = calculate_p_values(eeg_epochs, is_target_event, target_erp, nontarget_erp)

#print(p_values)

# Plot histogram of max_abs_diff samples and initial stat value
plt.figure(figsize=(8, 6))
plt.hist(max_abs_diff_samples, bins=10, color='teal', label='Max Absolute Difference')  # Plot histogram of max_abs_diff
plt.axvline(x=initial_stat, color='orange', linestyle='--', label='Initial Stat Value')  # Plot initial stat value
plt.xlabel('Max Absolute Difference')
plt.ylabel('Frequency')
plt.title('Distribution of Max Absolute Difference from Bootstrap')
plt.legend()
plt.show()
                       
#%% Part D: Plot FDR-Corrected P Values

# False Discovery Rate correction to
def fdr(erp_times, target_erp, nontarget_erp, p_values, fdr_threshold = 0.05):
    res, corrected_p_values = fdr_correction(p_values.flatten(), alpha=fdr_threshold)
    print(res)
    meantarget_erp = np.mean(target_erp, axis=1)  # Compute the mean ERP across trials
    std_erp = np.std(target_erp, axis=1)  # Compute the standard deviation across trials
    sdmn1 = std_erp / np.sqrt(len(eeg_epochs[is_target_event]))  # Standard error of the mean
    mean_nontarget_erp = np.mean(nontarget_erp, axis=1) # Compute the mean ERP across trials
    std_erp2 = np.std(nontarget_erp, axis=1)  # Compute the standard deviation across trials
    sdmn2 = std_erp2 / np.sqrt(len(eeg_epochs[~is_target_event]))  # Standard error of the mean
    
    #corrected_p_values = corrected_p_values.reshape(eeg_epochs[is_target_event].shape[1:])
    #print(corrected_p_values.shape)
    # Plot ERPs and confidence intervals for each channel
    fig, axs = plt.subplots(nrows=3, ncols=3, figsize=(12, 10))
    axs = axs.flatten()  # Flatten the 2D array of subplots
    #significant_time_points_indices = np.where(corrected_p_values < fdr_threshold)
    erp_times = np.array(erp_times)
    for i in range(8):
        ax = axs[i]
        #ax.plot(erp_times[i], meantarget_erp[i], label='Mean Target ERP', color='teal')
        #ax.plot(erp_times[i], mean_nontarget_erp[i], label='Mean Non Target ERP', color='pink')
        ax.fill_between(erp_times, meantarget_erp - 2 * sdmn1, meantarget_erp + 2 * sdmn1,  alpha=0.2, label = 'Target confidence intervals' , lw = 5)
        ax.fill_between(erp_times, mean_nontarget_erp - 2 * sdmn2, mean_nontarget_erp + 2 * sdmn2, alpha=0.2, label = 'Nontarget confidence intervals', lw = 200)
        ax.plot(erp_times, target_erp[:, i], label='Target ERP' )
        ax.plot(erp_times, nontarget_erp[:, i], label='Non-target ERP')
        ax.axvline(0, color='black', linestyle='--')  # Mark stimulus onset
        ax.axhline(0, color='black', linestyle=':')  # Mark zero voltage
        ax.set_title(f'Channel {i+1}')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Voltage (μV)')
        
        significant_time_points = erp_times[np.where(corrected_p_values < fdr_threshold)]
        ax.plot(significant_time_points, np.zeros_like(significant_time_points), 'ko')  # Plot black dots on x-axis
        
        plt.tight_layout()
        
    
    ax.legend(loc='lower right')
    fig.delaxes(axs[8])

    # Save the resulting image file
    plt.show()
    
    return None

fdr(erp_times, target_erp, nontarget_erp, p_values, fdr_threshold = 0.05)

#correct p values for multiple comparisons
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
<<<<<<< Updated upstream
    
=======

def multiple_subject_evaluation(subjects=np.arange(3,11), data_directory='P300Data/', epoch_start_time=-0.5, epoch_end_time=1.0, randomization_count=3000):
    combined_eeg =[]
    is_target_combined = []
    for subject in subjects:
        
        # loading data
        is_target_event, eeg_epochs, erp_times, target_erp, nontarget_erp = load_erp_data(subject, data_directory, epoch_start_time, epoch_end_time)
        
        # bootstrapping
        # declare necessary variables
        sample_count = eeg_epochs.shape[1]
        channel_count = eeg_epochs.shape[2]

        # preallocate arrays for resampled data 
        sampled_target_erp = np.zeros([randomization_count,sample_count,channel_count])
        sampled_nontarget_erp = np.zeros([randomization_count,sample_count,channel_count])
        combined_eeg.append(eeg_epochs)
        is_target_combined.append(is_target_event)
        # perform the bootstrapping
        """for randomization_index in range(randomization_count):
                
            # resample targets
            sampled_target_erp[randomization_index,:] = bootstrap_erps(eeg_epochs, is_target_event)[0][:,:]
            
            # resample nontargets
            sampled_nontarget_erp[randomization_index,:] = bootstrap_erps(eeg_epochs, is_target_event)[1][:,:]
            
        # find p_values
        p_values = calculate_p_values(sampled_target_erp, sampled_nontarget_erp,target_erp, nontarget_erp,randomization_count=3000)
        
        # FDR correction and plotting
        plot_false_discovery_rate(eeg_epochs, erp_times, target_erp, nontarget_erp, is_target_event, p_values, subject, fdr_threshold = 0.05)
        
        # declare an array of zeros to determine significance
        # track number of subjects where a point in time is significant for each channel
        # do this with boolean indexing: true if the sample in time for that subject and channel is significant
        # use np.sum to track total, likely want (384,8) array
        # return the sum, take into next function"""
            
    return combined_eeg,is_target_combined
#def subject_significance_plots():
>>>>>>> Stashed changes

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
<<<<<<< Updated upstream
=======
# Function to compute group median ERP
def compute_group_median(eeg_epochs, is_target_event):
    target_median = np.median(eeg_epochs[is_target_event], axis=0)
    nontarget_median = np.median(eeg_epochs[~is_target_event], axis=0)
    return target_median, nontarget_median

channel_names = get_channel_names(montage_name='biosemi64')


def plot_spatial_map(eeg_epochs, is_target_event,erp_times,combined_eeg, is_target_combined, subject=3):
    
    combined_eeg = np.array(combined_eeg)
    is_target_combined = np.array(is_target_combined)
    print(is_target_combined.shape)
    combined_erp_target = np.mean(combined_eeg[is_target_combined],axis = 0)
    combined_erp_nontarget = np.mean(combined_eeg[~is_target_combined],axis = 0)
    
    median_target_erp = np.median(combined_erp_target,axis=(0,1))
    median_nontarget_erp = np.median(combined_erp_nontarget, axis=(0,1))
    
    n2_time_range = (0.2, 0.3)  # 200-300 ms
    p3b_time_range = (0.3, 0.5)  # 300-500 ms
    n2_start_time, n2_end_time = n2_time_range
    n2_start_index = np.where(erp_times == n2_start_time)[0][0]
    n2_end_index = np.where(erp_times == n2_end_time)[0][0]
    
    
    # Extract voltage data for N2 time range
    target_median_n2 = target_median[:, n2_start_index:n2_end_index]
    nontarget_median_n2 = nontarget_median[:, n2_start_index:n2_end_index]
    
    # Plot topomap for N2 time range
    plt.figure(figsize=(10, 6))
    plot_topo(channel_names=channel_names, channel_data=target_median_n2, title='Target Median N2')
    plt.figure(figsize=(10, 6))
    plot_topo(channel_names=channel_names, channel_data=nontarget_median_n2, title='Nontarget Median N2')
    
    # Similarly, plot topomap for P3b time range
    # Assuming you have the P3b time range defined as p3b_start_time and p3b_end_time
    p3b_start_time, p3b_end_time = p3b_time_range
    p3b_start_index = np.where(erp_times == p3b_start_time)[0][0]
    p3b_end_index = np.where(erp_times == p3b_end_time)[0][0]

    # Extract voltage data for P3b time range
    target_median_p3b = target_median[:, p3b_start_index:p3b_end_index]
    nontarget_median_p3b = nontarget_median[:, p3b_start_index:p3b_end_index]
    
    # Plot topomap for P3b time range
    plt.figure(figsize=(10, 6))
    plot_topo(channel_names=channel_names, channel_data=target_median_p3b, title='Target Median P3b')
    plt.figure(figsize=(10, 6))
    plot_topo(channel_names=channel_names, channel_data=nontarget_median_p3b, title='Nontarget Median P3b')
    
    plt.show()
    plot_topo.plot_topo(['Fz', 'Cz', 'P3', 'Pz', 'P4', 'P7', 'P8', 'Oz'], median_target_erp, f'Subject{subject} P300 Spatial Map');
    
        
>>>>>>> Stashed changes
