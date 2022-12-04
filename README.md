#MODULE 12 CHALLENGE 

For this challenge I used various techniques to train and evaluate models with imbalanced classes. I started this assignment by using historical lending activity to build a model that can identify the creditworthiness of borrowers. I used a logistic regression model to compare two versions of the dataset, the original, and a oversampled version of the dataset. For both versions of the dataset, I counted the target cases, trained a logistic regression classifier, calculated the balanced accuracy score, generated a confusion matrix, and generated a classification report.

###Technologies

This project utilizes following packages:

Pandas
Scikit-Learn
Numpy

###Installation Guide

Install the following dependencies and libraries prior to using the application

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import confusion_matrix
from imblearn.metrics import classification_report_imbalanced

import warnings
warnings.filterwarnings('ignore')

