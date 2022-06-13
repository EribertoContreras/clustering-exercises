import math
import numpy as np
import seaborn as sns
import scipy.stats as stats
import pandas as pd
import matplotlib.pyplot as plt
import env
from pydataset import data
import scipy.stats
import scipy
import os
import sklearn.preprocessing
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, QuantileTransformer
from sklearn.metrics import explained_variance_score
import statsmodels.api as sm
# needed for modeling
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression, LassoLars, TweedieRegressor
from sklearn.preprocessing import PolynomialFeatures
import acquire

def summarize(df):
    "combines these functions to summarize the data set given to us"
    a = print('--- Shape: {}'.format(df.shape))
    #returns the information from the data
    bb = print('--- Info:')
    b = print((df.info()))
     # describes the data
    c = print('--- Descriptions:', (df.describe()))
    # returns the sum of null values in columns
    d = print('--- Nulls by Column:', (df.isnull().sum()))
    # returns nulls by row
    e = print('nulls by row:', (pd.concat([df.isna().sum(axis=1).rename('n_missing'),df.isna().mean(axis=1).rename('percent_missing'),], axis=1).value_counts().sort_index()))
    print(a, bb,b, c, d, e)


def nulls_by_columns(df):
    # gives us a count and a percent of missing information.
    return pd.concat([
        df.isna().sum().rename('count'),
        df.isna().mean().rename('percent')
    ], axis=1)


def handle_missing_value(df, prop_required_column, prop_required_row):
    #this piece of code allows us to handle the missing data and get rid of it, both in the columns and in the rows(so that we can analize better).
    n_required_column = round(df.shape[0] * prop_required_column)
    n_required_row = round(df.shape[1] * prop_required_row)
    df = df.dropna(axis=0, thresh=n_required_row)
    df = df.dropna(axis=1, thresh=n_required_column)
    return df





