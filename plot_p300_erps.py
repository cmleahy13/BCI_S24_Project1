#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 29 23:16:04 2024

plot_p300_erps.py

@authors: Claire Leahy and Lexi Reinsborough

This script, plot_p300_erps.py, is the module for Lab02 for BCIs S24. This file contains a series of functions that can help isolate EEG data based on event occurrences (i.e. flashes of a row or column). The first function, get_events, isolates when an event, target or nontarget, occurs. Using the returned data, the subsequent function epoch_data creates epochs, a sample range of data surrounding an event; for this particular case, the half-second leading up to an event and the second following event constitute the range of interest, and the EEG data in that range is gathered. Given the target data from get_events and the epochs from epoch_data, ERPs (event-related potentials) are calculated by taking the mean EEG voltages at each channel for a subject at an event (target or nontarget), as well as the times at which the ERPs occur. The data are subsequently plotted in plot_erps, where the whole plot represents a particular subject and each subplot depicts the ERPs for target and nontarget events at a particular channel.

Sources and collaborators: N/A

"""

#%% Part 1: Load the Data

# import statements
import numpy as np
from matplotlib import pyplot as plt

#%% Part 2: Extract the Event Times and Labels

# write function get_events(rowcol_id, is_target)
def get_events(rowcol_id, is_target):
    """
    Description
    -----------
    This function identifies sample indices in which there are events. These events can either be targets or nontargets. 

    Parameters
    ----------
    rowcol_id : Sx1 array of integers, where S is the total number of samples
        Input of current event type. Integers 1-6 correspond to the columns of the same number, 7-12 correspond to the rows numbered 1-6 in ascending order.
    is_target : Sx1 Boolean array, where S is the total number of samples
        Input of true when the row/column flashed contains the target letter, False otherwise. Serves as y axis (0 False, 1 True) for one of the subplots.

    Returns
    -------
    event_sample : Nx1 array of integers, where N is the number of samples in which an event occurred
        This array contains all of the sample numbers at which an event, a target flash or nontarget flash, occurred.
    is_target_event : Nx1 Boolean array, where N is the number of samples in which an event occurred
        This array contains a Boolean value of the event type in each of the relevant indices determined via event_sample. is_target_event[index] is True in cases in which the event was a target, False in the case that the event was not a target.

    """
    
    # array event_sample of samples when event ID went upward
    event_sample = np.where(np.diff(rowcol_id)>0)[0]+1 # index [0] accesses array from tuple
    
    # is_target_event[i] is True if event i was a target event
    is_target_event = is_target[event_sample] 
    
    # return event_sample and is_target_event
    return event_sample, is_target_event

#%% Part 3: Extract the Epochs

def epoch_data(eeg_time, eeg_data, event_sample, epoch_start_time=-0.5, epoch_end_time=1.0):
    """
    Description
    -----------
    Collates EEG data into epochs, which are collections of data from specific time ranges around each P300 event.
    Takes all the samples relative to each P300 Speller event's index (from event_sample) and stacks them next to each other in an array so that all the events line up and we can compare data from before and after each one.
    Also creates a lookup of the time offsets from each event to determine how far before or after an event any given data point occurred.

    Parameters
    ----------
    eeg_time : Sx1 array of floats, where S is the total number of samples
        The time when each sample was recorded, in seconds.
    eeg_data : CxS array of floats, where C is the number of channels and S is the total number of samples
        The recorded samples for each EEG channel in microvolts.
    event_sample : Nx1 array of integers, where N is the number of samples in which an event occurred
        This array contains all of the sample indices at which an event occurred.
    epoch_start_time : float, optional
        Beginning of relative range to collect samples around an event, in seconds. (-1.0 would be 1 second before the event's onset). The default is -0.5.
    epoch_end_time : float, optional
        Ending of relative range. (1.0 would be 1 second after the event's onset). The default is 1.0.

    Returns
    -------
    eeg_epochs : NxPxC array of floats, where N is the number of epochs, P is the number of samples in each epoch, and C is the number of channels.
        Each epoch corresponds to one sample in event_sample, and each epoch contains the data from before and after each event from all C channels.
    erp_times : Px1 array of floats, where P is the number of samples in each epoch
        The time offsets at which any given sample in eeg_epochs occurred relative to its corresponding event onset, along dimension 1.
        
    """
    
    # find size of each dimension for eeg_epochs 
    # number of epochs (dimension 0)
    epoch_count = len(event_sample)
    # samples per epoch (dimension 1)
    samples_per_second = 1/(eeg_time[1]-eeg_time[0])
    seconds_per_epoch = epoch_end_time - epoch_start_time
    samples_per_epoch = int(samples_per_second*seconds_per_epoch) # cast as integer for dimensioning
    # number of channels (dimension 2)
    channel_count = len(eeg_data)
    
    # declare array to contain epoched data
    eeg_epochs = np.zeros((epoch_count, samples_per_epoch, channel_count))
    # load epochs into array
    for epoch_index in range(epoch_count): 
        for channel_index in range(channel_count):
            start_offset = int(event_sample[epoch_index]+(epoch_start_time*samples_per_second)) # samples to go back before event
            for sample_index in range(samples_per_epoch):
                raw_sample = sample_index + start_offset # sample number at the start of the epoch (not event)
                eeg_epochs[epoch_index, sample_index, channel_index] = eeg_data[channel_index, raw_sample]
    
    # declare array to contain times of event-related potentials (ERPs)
    erp_times = []
    # load in times in seconds for each sample in the epoch 
    for time_index in range(int(samples_per_epoch)):
        erp_times.append(epoch_start_time+(time_index*(1/samples_per_second)))
    
    return eeg_epochs, erp_times

#%% Part 4: Calculate the ERPs

def get_erps(eeg_epochs, is_target_event):
    """
    Description
    -----------
    Separate data from events which are target events (i.e. the flash was of the correct row or column) and ones which are not, then take the mean voltage for each channel and sample position for each of these new separated datasets.

    Parameters
    ----------
    eeg_epochs : NxPxC array of floats, where N is the number of epochs, P is the number of samples in each epoch, and C is the number of channels
        Each epoch corresponds to one sample in event_sample, and each epoch contains the data from before and after each event from all C channels.
    is_target_event : Nx1 Boolean array, where N is the number of samples in which an event occurred
       This array contains a Boolean value of the event type in each of the relevant indices determined via event_sample. is_target_event[index] is True in cases in which the event was a target, False in the case that the event was not a target.

    Returns
    -------
    target_erp : PxC array of floats, where P is the number of samples in each epoch, and C is the number of channels
        Mean values for each channel and sample position in all epochs where the epoch corresponds to a target event.
    nontarget_erp : PxC array of floats, where P is the number of samples in each epoch, and C is the number of channels
        Mean values where the epoch doesn't correspond to a target event.

    """
    
    # separate epochs by target or nontarget event
    target_epochs = eeg_epochs[is_target_event] # (target_count, samples_per_epoch, channel_count)
    nontarget_epochs = eeg_epochs[~is_target_event] # (nontarget_count, samples_per_epoch, channel_count)
    
    # mean response on each channel for each event
    target_erp = np.mean(target_epochs,axis=0)
    nontarget_erp = np.mean(nontarget_epochs,axis=0)
    
    return target_erp, nontarget_erp

#%% Part 5: Plot the ERPs

def plot_erps(target_erp, nontarget_erp, erp_times, subject=3):
    """
    Description
    -----------
    Transforms mean channel data into 8 matplotlib plots and saves them in a .png image.

    Parameters
    ----------
    target_erp : PxC array of floats, where P is the number of samples in each epoch, and C is the number of channels
        Mean values for each channel and sample position in all epochs where the epoch corresponds to a target event.
    nontarget_erp : PxC array of floats, where P is the number of samples in each epoch, and C is the number of channels
        Mean values where the epoch doesn't correspond to a target event.
     erp_times : Px1 array of floats, where P is the number of samples in each epoch
        The time offsets at which any given sample in eeg_epochs occurred relative to its corresponding event onset, along dimension 1.
    subject : int, optional
        index of the subject which the data comes from. used only for labeling plot and filename of output. The default is 3.

    Returns
    -------
    None.

    """
    
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
        channel_plot = channel_plots[row_index][column_index]
        
        # plot dotted lines for time 0 and 0 voltage
        channel_plot.axvline(0, color='black', linestyle='dotted')
        channel_plot.axhline(0, color='black', linestyle='dotted')
        
        # plot target and nontarget erp data in the subplot
        target_handle, = channel_plot.plot(erp_times, target_erp_transpose[channel_index])
        nontarget_handle, = channel_plot.plot(erp_times, nontarget_erp_transpose[channel_index])
        
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
    
