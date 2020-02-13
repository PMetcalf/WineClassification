'''

Data Pipeline Script

This script loads wine data from the UCI Machine Learning repository, 
performs data visualisation and preprocessing operations, 
and then trains a deep neural network model to classify whether a wine is red or white.


'''

# Module Importations (A - Z)
import matplotlib.pyplot as plt
import pandas as pd

# Routine plots histograms of alcohol content for red and white wine
def plot_alcohol_histogram(red, white):

    # Initialise the figure
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

    # Initialise the figure
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

# Visualise some of the input data
plot_alcohol_histogram(red, white)
plot_sulphates_scatter(red, white)