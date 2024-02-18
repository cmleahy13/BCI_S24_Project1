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
