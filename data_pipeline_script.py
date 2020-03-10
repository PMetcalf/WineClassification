'''

Data Pipeline Script

This script loads wine data from the UCI Machine Learning repository, 
performs data visualisation and preprocessing operations, 
and then trains a deep neural network model to classify whether a wine is red or white.

The neural network used is a multi-layer perceptron, using the relu function for activation. 
It uses TensorFlow via the Keras API.


'''

# Module Importations (A - Z)
import keras as krs
from keras.layers import Dense
from keras.models import Sequential
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, cohen_kappa_score
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import StandardScaler

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

'''
Create a Model to Predict Wine Type
'''

# Tag and merge the datasets to preprocess
red['type'] = 1
white['type'] = 0

wines = red.append(white, ignore_index = True)

# Create and display correlation matrix
corr = wines.corr()
sns.heatmap(corr, xticklabels = corr.columns.values, yticklabels = corr.columns.values)
#plt.show()

# Specify the model input data
X = wines.iloc[:, 0:11]

# Specify and flatten the model output data
y = np.ravel(wines.type)
print(y)

'''
# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 42)

# Use a library scaler to standardise disparate data points
scaler = StandardScaler().fit(X_train)

# Scale the training set
X_train = scaler.transform(X_train)

# Scale the test set
X_test = scaler.transform(X_test)

# Build the architecture of the model

# Clear any existing models
krs.backend.clear_session()

# Initialise the model
model = Sequential()

# Add the input layer
model.add(Dense(12, activation = 'relu', input_shape = (11,)))

# Add one hidden layer
model.add(Dense(8, activation = 'relu'))

# Add the output layer
model.add(Dense(1, activation = 'sigmoid'))

# View data about intialised NN model
#print(model.output_shape)
#print(model.summary())
#print(model.get_config())
#print(model.get_weights())

# Compile and fit the model
model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])

model.fit(X_train, y_train, epochs = 5, batch_size = 1, verbose = 1)

# Make some initial predictions with the model
y_pred = model.predict(X_test)
y_pred = [np.round(x) for x in y_pred]
print(y_test[:5])
print(y_pred[:5])

# Evaluate the performance of the model
score = model.evaluate(X_test, y_test, verbose = 1)
print("Score: " + str(score))

# Confusion matrix
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Model Precision
print("Precision:")
print(precision_score(y_test, y_pred))

# Model Recall
print("Recall:")
print(recall_score(y_test, y_pred))

# Model F1 Score
print("F1:")
print(f1_score(y_test, y_pred))

# Model Cohen's Kappa
print("Cohen Kappa:")
print(cohen_kappa_score(y_test, y_pred))
'''

'''
Create a Model to Predict Wine Quality
'''

# Isolate target labels
y = wines.quality
print(y)

# Isolate data
X = wines.drop('quality', axis = 1)
print(X)

# Re-apply data scaling to the isolated data.
X = StandardScaler().fit_transform(X)

# Build a regression neural network model.

# Clear any existing models
krs.backend.clear_session()

# Initialise the model.
model = Sequential()

# Add the input layer.
model.add(Dense(64, input_dim = 12, activation = 'relu'))

# Add the output layer.
model.add(Dense(1))