'''

Data Pipeline Script

This script loads wine data from the UCI Machine Learning repository, 
performs data visualisation and preprocessing operations, 
and then trains a deep neural network model to classify whether a wine is red or white.


'''

# Module Importations (A - Z)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Routine plots histograms of alcohol content for red and white wine
def plot_alcohol_histogram(red, white):

    # Initialise figure
    fig, ax = plt.subplots(1, 2)

    # Plot the data as histograms
    ax[0].hist(red.alcohol, 10, facecolor = 'red', alpha = 0.5, label = 'Red Wine')
    ax[1].hist(white.alcohol, 10, facecolor = 'white', ec = 'black', lw = 0.5, alpha = 0.5, label = 'White Wine')

    # Adjust figure geometry
    ax[0].set_ylim([0, 1000])

    # Add labels
    ax[0].set_xlabel('Alcohol: % Vol')
    ax[0].set_ylabel('Frequency')
    ax[1].set_xlabel('Alcohol: % Vol')

    fig.suptitle('Distribution of Alcohol % Vol')

    plt.show()

# Routine plots scatter graphs of sulphates and quality for red and white wines
def plot_sulphates_scatter(red, white):

    # Initialise figure
    fig, ax = plt.subplots(1, 2, figsize = (8, 4))

    # Plot the data as scatter charts
    ax[0].scatter(red['quality'], red['sulphates'], color = 'red')
    ax[1].scatter(white['quality'], white['sulphates'], color = 'white', ec = 'black', lw = 0.5)

    # Adjust figure geometry
    ax[0].set_xlim([0, 10])
    ax[0].set_ylim([0, 2.5])
    ax[1].set_xlim([0, 10])
    ax[1].set_ylim([0, 2.5])

    # Add labels
    ax[0].set_xlabel('Quality (Rating) - Red')
    ax[0].set_ylabel('Sulphates')
    ax[1].set_xlabel('Quality (Rating) - White')

    fig.suptitle('Sulphates vs. Quality for Red & White Wines')

    plt.show()

# Routine plots scatter graphs of acidic volatility, alcohol and quality for red and white wines
def plot_volatility_alcohol_quality_scatter(red, white):

    # Create colour legend using random numbers
    np.random.seed(570)
    red_colors = np.random.rand(6, 4)   # Array is shaped based on red.quality length
    white_colors = np.append(red_colors, np.random.rand(1, 4), axis = 0)    # Array is shaped based on white.quality length

    # Identify unique quality values for each wine
    red_labels = np.unique(red['quality'])
    white_labels = np.unique(white['quality'])

    # Initialise figure
    fig, ax = plt.subplots(1, 2, figsize = (8, 4))

    # Plot the data as scatter charts, by iterating over quality values

    for i in range(len(red_colors)):
        red_y = red['alcohol'][red.quality == red_labels[i]]
        red_x = red['volatile acidity'][red.quality == red_labels[i]]
        ax[0].scatter(red_x, red_y, c = red_colors[i])

    for i in range(len(white_colors)):
        white_y = white['alcohol'][white.quality == white_labels[i]]
        white_x = white['volatile acidity'][white.quality == white_labels[i]]
        ax[1].scatter(white_x, white_y, c = white_colors[i])

    # Adjust figure geometry
    ax[0].set_xlim([0, 1.7])
    ax[0].set_ylim([5, 15.5])
    ax[1].set_xlim([0, 1.7])
    ax[1].set_ylim([5, 15.5])

    # Add labels
    ax[0].set_title('Red Wine')
    ax[1].set_title('White Wine')
    ax[0].set_xlabel('Volatile Acidity')
    ax[0].set_ylabel('Sulphates')
    ax[1].set_xlabel('Volatile Acidity')

    # Add legend
    ax[0].legend(red_labels, loc = 'best', bbox_to_anchor = (1, 1))
    ax[1].legend(white_labels, loc = 'best', bbox_to_anchor = (1, 1))

    plt.show()

'''
Main body of Data Pipeline
'''

# Retrieve data from ML repository
white = pd.read_csv("http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv", sep = ';')

red = pd.read_csv("http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv", sep = ';')

# Peek at the retrieved data

print(red.info())
print(white.info())

# Inpsect data in more detail
print(red.head())
print(white.tail())
print(red.sample(5))

# Look at summary statisitics for each dataset
print(red.describe())
print(white.describe())

# Check for null values
print(red.isnull())
print(white.isnull())

# Conduct Exploratory Data Analysis on data
#plot_alcohol_histogram(red, white)
#plot_sulphates_scatter(red, white)
#plot_volatility_alcohol_quality_scatter(red, white)

# Tag & merge the datasets to preprocess
red['type'] = 1
white['type'] = 0

wines = red.append(white, ignore_index = True)