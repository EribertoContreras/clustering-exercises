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

##############################################################################################

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

##############################################################################################

def nulls_by_columns(df):
    # gives us a count and a percent of missing information.
    return pd.concat([
        df.isna().sum().rename('count'),
        df.isna().mean().rename('percent')
    ], axis=1)

##############################################################################################

def handle_missing_value(df, prop_required_column, prop_required_row):
    #this piece of code allows us to handle the missing data and get rid of it, both in the columns and in the rows(so that we can analize better).
    n_required_column = round(df.shape[0] * prop_required_column)
    n_required_row = round(df.shape[1] * prop_required_row)
    df = df.dropna(axis=0, thresh=n_required_row)
    df = df.dropna(axis=1, thresh=n_required_column)
    return df

##############################################################################################

def get_exploration_data(df):
    print ('Before dropping nulls, %d rows, %d cols' % df.shape)
    df = handle_missing_values(df, prop_required_column=.5, prop_required_row=.5)
    # changing na values to 0 
    df['heatingorsystemtypeid'] = df['heatingorsystemtypeid'].replace(np.nan, 0)
    # replacing na values with none to have the data matching on top
    df['heatingorsystemdesc'] = df['heatingorsystemdesc'].replace(np.nan, 'none')
    # filling in nozoned areas for data exploration purposes
    df['propertyzoningdesc'] = df['propertyzoningdesc'].replace(np.nan, 'non-zoned')
    # filling in missing value for unidentified biolding id
    df['buildingqualitytypeid'] = df['buildingqualitytypeid'].replace(np.nan, 0)
    #  replacing unit cnt values that are nulls 
    df['unitcnt'] = df['unitcnt'].replace(np.nan, 000)
    #dropping remaning null values that are not of major significance, under 2 thousand rows 
    df = df.dropna()
    # get rid of any crazy outliers such as large homes 
    print('After dropping nulls. %d rows. %d cols' % df.shape)
    train, validate, test = split(df)
    return train

##############################################################################################

def get_modeling_data(scale_data=False):
    df = acquire()
    print('Before dropping nulls, %d rows, %d cols' % df.shape)
    df = handle_missing_values(df, prop_required_column=.5, prop_required_row=.5)
    print('After dropping nulls, sd rows, %d cols' % df.shape)
    
    print()
    
    print('Before removing outliers, %d rows, %d cols' % df. shape)
    handle_outliers(df, ['age','spending_score','annual_income'], 1.5)#"make sure to input the columns u want to handle"
    print('after dropping nulls, %d rows, %d cols' % df.shape)
    print()
    
    df = one_hot_encode(df)
    
    train, validate, test = split(df)
    if scale_data:
        return scale(train, validate, test)
    else:
        train, validate, test
##############################################################################################
def split(df):
    train_and_validate, test = train_test_split(df, random_state=13, test_size=.15)
    train, validate = train_test_split(train and validate, random_state=13, test_size=.2)
    print('Train: %d rows, %d cols' % train.shape)
    print ('Validate: %d rows, %d cols' % validate. shape)
    print ('Test: %d rows, %d cols' % test.shape)
    
    return train, validate, test

##############################################################################################

def scale(train, validate, test):
    columns_to_scale = ['age','spending_score', 'annual_income']
    train_scaled = train.copy()
    validate_scaled = validate.copy()
    test_scaled = test.copy()
    scaler = MinMaxScaler()
    scaler.fit(train[columns_to_scale])
    train_scaled[columns_to_scale] = scaler.transform(train[columns_to_scale])
    validate_scaled[columns_to_scale] = scaler.transform(validate[columns_to_scale])
    test_scaled[columns_to_scale] = scaler.transform(test[columns_to_scale])
    return scaler, train_scaled, validate_scaled, test_scaled

##############################################################################################
def one_hot_encode(df):
    df['is_female'] = df.gender == 'Female'
    df = df.drop(columns='gender')
    return df
##############################################################################################

def handle_missing_values(df, prop_required_column, prop_required_row):
    n_required_column = round(df.shape[0] * prop_required_column)
    n_required_row = round(df.shape[1] * prop_required_row)
    df = df.dropna(axis=0, thresh=n_required_row)
    df = df.dropna(axis=1, thresh=n_required_column)
    return df

##############################################################################################

def handle_outliers(df, cols, k):
    # Create placeholder dictionary for each columns bounds
    bounds_dict = {}

    # get a list of all columns that are not object type
    non_object_cols = df.dtypes[df.dtypes != 'object'].index


    for col in non_object_cols:
        # get necessary iqr values
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        upper_bound =  q3 + k * iqr
        lower_bound =  q1 - k * iqr

        #store values in a dictionary referencable by the column name
        #and specific bound
        bounds_dict[col] = {}
        bounds_dict[col]['upper_bound'] = upper_bound
        bounds_dict[col]['lower_bound'] = lower_bound

    for col in non_object_cols:
        #retrieve bounds
        col_upper_bound = bounds_dict[col]['upper_bound']
        col_lower_bound = bounds_dict[col]['lower_bound']

        #remove rows with an outlier in that column
        df = df[(df[col] < col_upper_bound) & (df[col] > col_lower_bound)]
        
    return df

##############################################################################################

def split(df):
    train_and_validate, test = train_test_split(df, random_state=13, test_size=.15)
    train, validate = train_test_split(train_and_validate, random_state=13, test_size=.2)

    print('Train: %d rows, %d cols' % train.shape)
    print('Validate: %d rows, %d cols' % validate.shape)
    print('Test: %d rows, %d cols' % test.shape)

    return train, validate, test
#////////////////////////////////////////////////////////////////////////////////

def nulls_by_rows(df):
    print('nulls by row:', (pd.concat([df.isna().sum(axis=1).rename('n_missing'),
    df.isna().mean(axis=1).rename('percent_missing'),], axis=1).value_counts().sort_index()))

##############################################################################################
def null_to_zero(df):
    # changing na values to 0 
    df['heatingorsystemtypeid'] = df['heatingorsystemtypeid'].replace(np.nan, 0)
    # replacing na values with none to have the data matching on top
    df['heatingorsystemdesc'] = df['heatingorsystemdesc'].replace(np.nan, 'none')
    # filling in nozoned areas for data exploration purposes
    df['propertyzoningdesc'] = df['propertyzoningdesc'].replace(np.nan, 'non-zoned')
    # filling in missing value for unidentified biolding id
    df['buildingqualitytypeid'] = df['buildingqualitytypeid'].replace(np.nan, 0)
    # 
    df['unitcnt'] = df['unitcnt'].replace(np.nan, 000)
    return df
##############################################################################################
def null_row_removal(df):
    "after i cleaned the data based on percentages, i have taken a deeper look into the nulls that exist in each collumn"
    "ive decided that anything under 5 percent of null data missing should =  in a deletion of that row to make a better predictor"
    df = df.dropna()
    return df
