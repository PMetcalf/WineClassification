'''

Data Pipeline Script

This script loads wine data from the UCI Machine Learning repository, 
performs data visualisation and preprocessing operations, 
and then trains a deep neural network model to classify whether a wine is red or white.


'''

# Module Importations (A - Z)
import matplotlib.pyplot as plt
import pandas as pd

# Support routines
def plot_alcohol(red, white):

    # Initialise the figure
    fig, ax = plt.subplots(1, 2)

    # Plot the data as histograms
    ax[0].hist(red.alcohol, 10, facecolor = 'red', alpha = 0.5, label = 'Red Wine')
    ax[1].hist(white.alcohol, 10, facecolor = 'white', ec = 'black', lw = 0.5, alpha = 0.5, label = 'White Wine')

    # Make adjustments to the figure
    fig.subplots_adjust(left = 0, right = 1, bottom = 0, top = 0.5, hspace = 0.05, wspace = 1)
    ax[0].set_ylim([0, 1000])

    # Add labels

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
plot_alcohol(red, white)