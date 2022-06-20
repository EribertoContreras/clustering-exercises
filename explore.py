
# # credentials file to access the data
#import env
# Imports functions necessary to run visuals and hides unnecessary code
import wrangle
# coding 
import math
import numpy as np
import seaborn as sns
import scipy.stats as stats
import pandas as pd
import matplotlib.pyplot as plt
from pydataset import data
import scipy.stats
import scipy
import os
# needed for modeling
import sklearn.preprocessing
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, QuantileTransformer
from sklearn.metrics import explained_variance_score
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression, LassoLars, TweedieRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.cluster import KMeans
#import acquire

def split_clusters(df):
    # get train to expolore 
    train, validate, test = wrangle.split(df)
    # seeing what the train split dataset
    train.info()
    return train, validate, test

def explore_files(train):
    # graphing each colum seperately
    for col in train.columns:
        #graph size
        plt.figure(figsize=(4,2))
        #histogram graph
        plt.hist(train[col])
        #title of column
        plt.title(col)
        # show graph
        plt.show()
