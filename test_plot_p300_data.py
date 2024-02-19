#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 13 10:03:54 2024

@author: ClaireLeahy
"""

# import functions to be called
from plot_p300_data import load_erp_data

# calling the functions defined in the module

# load data
is_target_event, eeg_epochs, erp_times, target_erp, nontarget_erp = load_erp_data(subject=3, data_directory='P300Data/', epoch_start_time=-0.5, epoch_end_time=1.0)

# plot standard error
plot_confidence_intervals(eeg_epochs,erp_times, target_erp, nontarget_erp, is_target_event)

#%%

import numpy as np
from matplotlib import pyplot as plt

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