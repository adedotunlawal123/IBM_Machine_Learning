#!/usr/bin/env python
# coding: utf-8

#...

# Pandas is a software library written for the Python programming language for data manipulation and analysis.
import pandas as pd
# NumPy is a library for the Python programming language, adding support for large, multi-dimensional arrays and matrices, along with a large collection of high-level mathematical functions to operate on these arrays
import numpy as np
# Matplotlib is a plotting library for python and pyplot gives us a MatLab like plotting framework. We will use this in our plotter function to plot data.
import matplotlib.pyplot as plt
#Seaborn is a Python data visualization library based on matplotlib. It provides a high-level interface for drawing attractive and informative statistical graphics
import seaborn as sns
# Preprocessing allows us to standarsize our data
from sklearn import preprocessing
# Allows us to split our data into training and testing data
from sklearn.model_selection import train_test_split
# Allows us to test parameters of classification algorithms and find the best one
from sklearn.model_selection import GridSearchCV
# Logistic Regression classification algorithm
from sklearn.linear_model import LogisticRegression
# Support Vector Machine classification algorithm
from sklearn.svm import SVC
# Decision Tree classification algorithm
from sklearn.tree import DecisionTreeClassifier
# K Nearest Neighbors classification algorithm
from sklearn.neighbors import KNeighborsClassifier
# Bayesian Ridge Regression Algorithm
from sklearn.linear_model import BayesianRidge
# Allows to compute confusion matrix for accuracy of model
from sklearn.metrics import confusion_matrix
# A library to take arguments on the commandline
import sys

if len(sys.argv) != 2:
    print("Only one argument is needed")
    exit()
    
file_name = sys.argv[1]

with open (file_name,"r") as myfile:
    A = myfile.read().splitlines()

#Change string of tuples to tuples   
models = [eval(a) for a in A]

def plot_confusion_matrix(y_test,y_predict):
    """
    Return a plot of a confusion matrix
    
    Parameters
    ----------
    
    y_test : numpy.ndarray
        Y test for the train_test_split scikit learn method
    y_predict: numpy.ndarray
    	predicted values of Y
        
    Returns
    -------
    
    matplotlib object
    A matplotlib plot of the confusion matrix
    """

    cm = confusion_matrix(y, y_predict)
    ax= plt.subplot()
    sns.heatmap(cm, annot=True, ax = ax); #annot=True to annotate cells
    ax.set_xlabel('Predicted labels')
    ax.set_ylabel('True labels')
    ax.set_title('Confusion Matrix'); 
    ax.xaxis.set_ticklabels(['did not land', 'land']); ax.yaxis.set_ticklabels(['did not land', 'landed'])


data = pd.read_csv('https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DS0701EN-SkillsNetwork/api/dataset_part_2.csv')

X = pd.read_csv('https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DS0701EN-SkillsNetwork/api/dataset_part_3.csv')

Y = data['Class'].to_numpy()

scaler = preprocessing.StandardScaler()

scaler.fit(X)

X = scaler.transform(X)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 2)

# loop through the models and get accuracy
def get_best_model(models):
    """
    Get the Best Parameters, Accuracies of a model of a ML Algorithm
    
    Parameters
    ----------
    
    models : List
        A List of tuples. Tuple contain model name, instantiation of model and supplied parameters
        
    Returns
    -------
    
    str
    Best Parameters, Accuracies of model
    """
    for model, instance, parameter in models:
        i = instance
        i_cv = GridSearchCV(i, parameter, cv = 10)
        i_cv.fit(X_train, Y_train)
        score = i_cv.score(X_test, Y_test)
        print("Best Parameters,Accuracy for {}:".format(model))
        print("\n")
        print("Tuned hyperparameters :(best parameters) ",i_cv.best_params_)
        print("\n")
        print("Accuracy_1 :",i_cv.best_score_)
        print("\n")
        print("Accuracy_2", score)
        print("\n\n\n\n\n")


get_best_model(models)
