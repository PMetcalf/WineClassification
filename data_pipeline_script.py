'''

Data Pipeline Script

This script loads wine data from the UCI Machine Learning repository, 
performs data visualisation and preprocessing operations, 
and then trains a deep neural network model to classify whether a wine is red or white.


'''

# Module Importations (A - Z)
import pandas as pd

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