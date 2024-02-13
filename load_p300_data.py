#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#%% Part 7: Create Headers, Docstrings, and Comments

"""

    load_p300_data.py
    
    This file is a Python script containing the function definitions to be called in test_load_p300_data.py. These functions effectively load and plot the data, with a culminating function that can loop through multiple subjects. There is an additional function that determines the string the subjects were instructed to spell given truth and target data. This is the module script for Lab 1.
    
    Written by Claire Leahy, 01/19/2024
    
    Sources: 
        GeeksForGeeks: Refresher on default arguments (for load_training_eeg)
            https://www.geeksforgeeks.org/default-arguments-in-python/

"""


#%% importing modules
import numpy as np
from matplotlib import pyplot as plt
import loadmat

#%% Part 5: Declare and Call a Function

# function to load training data
def load_training_eeg(subject=3, data_directory='P300Data/'): # default values for inputs set
    """
    Description
    -----------
    This function loads the training EEG data from a folder containing P300 Speller data files. The data are loaded into a dictionary variable, data, and subsequently extracted into another dictionary (via the nested dictionary) called train_data.
    
    Parameters
    ----------
    subject (Optional) : int
        Input to define the subject number to be evaluated. The default is 3.
    data_directory (Optional) : str
        Input string directory to the location of the data files. The default is 'P300Data/'.
    
    Returns
    -------
    eeg_time : Array of float, size Sx1, where S is the number of samples
        Time of the sample in seconds.
    eeg_data : Array of float, size CxS , where C is the number of channels and S is the number of samples.
        Raw electroencephalography (EEG) data in uV.
    rowcol_id : Array of integers, size Sx1, where S is the number of samples
        Current event type. Integers 1-6 correspond to the rows of the same number, 7-12 correspond to the columns numbered 1-6 in ascending order.
    is_target : Array of Boolean, size Sx1, where S is the number of samples
        True when the row/column flashed contains the target letter, False otherwise.
        
    """
    
    data_file = f'{data_directory}s{subject}.mat'
    
    # create subject string for ease of use
    subject_string = f's{subject}';
    
    # loading data
    # use loadmat function to load data into data variable of type dict
    data = {subject_string: {'train': loadmat.loadmat(data_file)}};
    # extract training data from dict, put in train_data variable
    train_data = data[subject_string]['train'];
    
    # row 0: time (s)
    eeg_time = np.array(train_data[subject_string]['train'][0]); # eeg = electroencephalography
    
    # rows 1-8: EEG data (uV)
    eeg_data = np.array(train_data[subject_string]['train'][1:9]); # Python is exclusive at end
    
    # row 9: current event type
    rowcol_id = np.array(train_data[subject_string]['train'][9],dtype=int);
    
    # row 10: 1 when current row/col flashed includes target letter, 0 otherwise
    is_target = np.array(train_data[subject_string]['train'][10],dtype=bool);
    
    return eeg_time, eeg_data, rowcol_id, is_target

# function to plot data
def plot_raw_eeg(subject, eeg_time, eeg_data, rowcol_id, is_target):
    """
    Description
    -----------
    This function plots the data obtained using the load_training_eeg function for a particular subject number input. For a given subject, a figure with three subplots, comprised of the the eeg_time (x) data and the eeg_data, rowcol_id, is_target (the latter three all output (y) data). The function also saves the figure as a .png to the same directory as the source code.

    Parameters
    ----------
    subject : Integer
        Input to define the subject number to have its data plotted.
    eeg_time : Array of floats, 61866x1 (number of samples)
        Input array of the sample in seconds. Serves as x axis on all plots.
    eeg_data : Array of floats, 8x61866 (number of EEG channels by number of samples)
        Input of raw electroencephalography (EEG) data in uV. Serves as y axis on one of the subplots.
    rowcol_id : Array of integers, 61866x1 (number of samples)
        Input of current event type. Integers 1-6 correspond to the columns of the same number, 7-12 correspond to the rows numbered 1-6 in ascending order. Serves as y axis on one of the subplots.
    is_target : Array of boolean, 61866x1 (number of samples)
        Input of true when the row/column flashed contains the target letter, False otherwise. Serves as y axis (0 False, 1 True) for one of the subplots.

    Returns
    -------
    None.

    """
    
    # creating figures and plotting data
    fig, axs = plt.subplots(nrows=3, ncols=1, sharex=True);
    axs[0].plot(eeg_time, rowcol_id);
    axs[1].plot(eeg_time, is_target);
    axs[2].plot(eeg_time, eeg_data.T); # plot transpose of eeg_data

    # formatting plots
    # axis limits
    axs[2].set_xlim(48,53); # 5 second window starting 1 second before first flash
    axs[2].set_ylim(-25, 25);
    # add grid
    for subplot_index in [0,1,2]:
        axs[subplot_index].grid(which='both')
        
    # axis labels and title
    plt.suptitle(f'P300 Speller Subject {subject} Raw Data');
    plt.xlabel('time (s)');
    axs[0].set_ylabel('row/column ID');
    axs[1].set_ylabel('target ID');
    axs[2].set_ylabel('voltage (uV)');

    # save figure
    figure_file_name = f'P300_S{subject}_training_rawdata.png'
    plt.savefig(figure_file_name);
    
#%% Part 6: Create a Loop

def load_and_plot_all(data_directory, subjects):
    """
    Description
    -----------
    Function that loops through load_training_eeg and plot_raw_eeg functions to load and plot data for multiple subjects at once.

    Parameters
    ----------
    data_directory : String
        Input string directory to the location of the data files.
    subjects : Array of integers, 1x(number of subjects)
        Input array of integers to plot and load data for multiple subjects with corresponding subject IDs. Possible number of subjects include 1-10. Subjects 1 and 2 use the single character speller (triggers up to 36), while subjects 3-10 use the row/column (matrix) speller (triggers up to 12). 

    Returns
    -------
    None.

    """
    
    # loop through two functions
    for subject_index in subjects:
        eeg_time, eeg_data, rowcol_id, is_target = load_training_eeg(subject_index, data_directory)
        plot_raw_eeg(subject_index, eeg_time, eeg_data, rowcol_id, is_target)
        
#%% Part 8: Decode the Message

def decode_letters(is_target, rowcol_id, subject=3,data_directory='P300Data/',string_size=5):
    """
    Description
    -----------
    This function takes in the arrays generated by the functions that load the data, as well as a string size, to determine the string being spelled in the P300 Speller. The primary means of obtaining relevant data include evaluating when an event occurred (when is_target was True). At the indices where is_target was indeed True, more information, such as the row/column ID, was obtained. After isolating the rows and columns and converting this information to match the character matrix (generated as a variable within this function), the function prints the string spelled by the subject.
    
    Parameters
    ----------
    is_target : Array of boolean, 61866x1 (number of samples)
        Input of true when the row/column flashed contains the target letter, False otherwise.
    rowcol_id : Array of integers, 61866x1 (number of samples)
        Input of current event type. Integers 1-6 correspond to the columns of the same number, 7-12 correspond to the rows numbered 1-6 in ascending order.
    subject (Optional) : Integer
        Input to define the subject number to be evaluated. The default is 3.
    data_directory (optional) : String
        Input string directory to the location of the data files. The default is 'P300Data/'.
    string_size (Optional) : Integer
        Size of the string being typed with the P300 Speller. The default is 5.

    Returns
    -------
    None.

    """

    # generate character matrix that can be converted to what's in paper
    character_matrix = [['A', 'G', 'M', 'S', 'Y', '4'],
                        ['B', 'H', 'N', 'T', 'Z', '5'],
                        ['C', 'I', 'O', 'U', '0', '6'],
                        ['D', 'J', 'P', 'V', '1', '7'],
                        ['E', 'K', 'Q', 'W', '2', '8'],
                        ['F', 'L', 'R', 'X', '3', '9']]

    # find cases where target event took place
    true_target = is_target.nonzero(); # returns tuple, isolate indices
    separated_true_target = true_target[0]; # accesses first index of tuple, which is the whole array

    targets = []; # generate empty array for the target rows/columns to be entered
    for target_index in range(0,len(separated_true_target)):
        targets.append(rowcol_id[separated_true_target[target_index]])
        
    # isolate groupings by string size
    # all flashes of a row and column PAIR will be contained in a group
    character_groupings = []; # generate empty array for grouped rows/columns (each character)
    # split the targets array based on total number of True events and expected string size
    for splitting_index in range(0, string_size): # splitting is essentially fraction of events based on string size
        start_index = int(splitting_index*(len(separated_true_target)/string_size))
        end_index = int((splitting_index+1)*(len(separated_true_target)/string_size)-1)
        character_groupings.append(targets[start_index:end_index])

    # isolate single row and column intersection for each character
    unique_characters = []; # generate empty array to store character row/column individually
    for character_index in range(0,len(character_groupings)):
        unique_characters.append(np.unique(character_groupings[character_index]))

    # update rows/columns to match character_matrix
    for rowcol_index in range(0,len(unique_characters)):
        unique_characters[rowcol_index][0] = unique_characters[rowcol_index][0]-1; # -1 converts column
        unique_characters[rowcol_index][1] = unique_characters[rowcol_index][1]-7; # -7 converts row
        
    final_string = ''; # generate empty string to add actual letters based on row/column
    for final_rowcol_index in range(string_size):
        final_string = final_string + character_matrix[unique_characters[final_rowcol_index][0]][unique_characters[final_rowcol_index][1]];
        
    print(f'Subject {subject} string: {final_string}')
        
    # find the characters a subject can spell per minute
    # find total sampling time once flashes begin
    true_targets = is_target.nonzero(); # find where the targets are flashed
    target_times = true_targets[0]; # extract data as array from the tuple
    starting_sample = target_times[0]; # first sample number when a target was flashed
    ending_sample = target_times[-1]; # last sample number when target was flashed
    
    # find period where samples where actively involved in generating string
    samples_count = ending_sample - starting_sample; # total number of samples
    sampling_frequency = 256;
    seconds_of_data = samples_count/sampling_frequency;
    
    # convert to desired units of characters/minute
    minutes_of_data = seconds_of_data/60; # 60s/min
    string_character_count = 5; # 5 characters in string of interest
    characters_per_minute = string_character_count/minutes_of_data;

    # print the number of characters expected per minute given data
    print(f'For subject {subject}:','%.2f' % characters_per_minute, 'characters/minute\n');

            

    
    
