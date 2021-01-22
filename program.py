#!/usr/bin/env python
# coding: utf-8


import sqlite3
import csv
import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
sns.set(color_codes = True)
import sys
import xgboost
from sklearn import datasets, linear_model, preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from pandas import set_option
from pandas.plotting import scatter_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RepeatedKFold
from sqlalchemy import create_engine
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix 
from sklearn.metrics import classification_report 
from sklearn.preprocessing import MinMaxScaler
from xgboost import plot_importance
from numpy import absolute
from numpy import mean
from numpy import std

def XGB(X_train, X_validation, Y_train, Y_validation):
    xgb= XGBClassifier(learning_rate = 0.02) 
    xgb.fit(X_train, Y_train) 
    predictions = xgb.predict(X_validation) 
    print('XGBoost Scaled:',accuracy_score(Y_validation, predictions)) 
    print(confusion_matrix(Y_validation, predictions)) 
    print(classification_report(Y_validation, predictions))
    plot_importance(xgb,max_num_features =5)
    print('Top 5 features= f24:num_videos, f14:data_channel_world, f38:kw_avg_avg, f37: kw_min_min,    f28; n_comments')
    plt.show()
     
def LASSO(x_train, y_train):
    model =Lasso(alpha=0.99)
    kfold = KFold(n_splits=10, random_state=7)
    scores = cross_val_score(model, x_train, y_train, scoring='neg_mean_absolute_error', cv=kfold, n_jobs=-1)
    scores = absolute(scores)
    print('Mean MAE: %.3f (%.3f)' % (mean(scores), std(scores)))
    
    
def main():
    #Read input dataset
    # engine = create_engine('sqlite:///Users/benja/Documents/Datascience/AISG/ASIG2021_Test/Benjamin_Tan_24199/news_popularity.db', echo=True)
    # dat = sqlite3.connect('news_popularity.db')
    #data/news_popularity.db
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    db_path = os.path.join(BASE_DIR, "news_popularity.db")
    engine = create_engine(f"sqlite:///{db_path}", echo=True)
    dat = sqlite3.connect(db_path)

    #table articles
    query = dat.execute("SELECT * From articles")
    cols = [column[0] for column in query.description]
    art= pd.DataFrame.from_records(data = query.fetchall(), columns = cols)
    # table description
    query = dat.execute("SELECT * From description")
    cols = [column[0] for column in query.description]
    des= pd.DataFrame.from_records(data = query.fetchall(), columns = cols)
    #table keywords
    query = dat.execute("SELECT * From keywords")
    cols = [column[0] for column in query.description]
    key= pd.DataFrame.from_records(data = query.fetchall(), columns = cols)
    #Merge tables
    df=pd.merge(art,des,how='inner',on='ID') 
    df1=pd.merge(df,key,how='inner',on='ID')

    #drop index columns columns ID, URL 
    df2=df1.drop(['ID','url'], axis=1)
    # drop missing values <3% 
    df2 = df2[pd.notnull(df2['num_hrefs'])]
    df2['num_imgs'].fillna(0, inplace=True)
    df2['num_videos'].fillna(0, inplace=True)
    #transform from NAN to UKN(unknown) for data_channel 
    df2 = df2.replace(np.nan, 'UKN', regex=True)
    #group numerical colums
    dfnum=df2[['timedelta','n_tokens_title', 'n_tokens_content',
       'n_unique_tokens', 'n_non_stop_words', 'n_non_stop_unique_tokens',
       'num_hrefs', 'num_self_hrefs', 'num_imgs', 'num_videos', 'n_comments',
       'average_token_length', 'self_reference_min_shares',
       'self_reference_max_shares', 'self_reference_avg_shares',
       'num_keywords', 'kw_min_min', 'kw_max_min', 'kw_avg_min', 'kw_min_max',
       'kw_max_max', 'kw_avg_max', 'kw_min_avg', 'kw_max_avg', 'kw_avg_avg','shares']]
    #Categorical columns
    colcat= ['weekday','data_channel']
    dfrefcat=df2[colcat]
    dfrefcat=pd.get_dummies(dfrefcat[colcat])
    dfref= pd.concat([dfrefcat,dfnum],axis =1)
    dfm1=dfref.copy()

    # 1: unpopular    , 2 : popular 
    dfref.loc[dfref.shares<1400,'shares']=1
    dfref.loc[dfref.shares>=1400,'shares']=2
    df=dfref.copy()

    # Split Data for model 1 linear regression train and validation
    # Split-out validation dataset
    array = dfm1.values
    x = array[:,0:39]
    y = array[:,39]
    validation_size = 0.20
    seed = 7
    x_train, x_validation, y_train, y_validation = train_test_split(x, y, test_size=validation_size, random_state=seed)
    scaler = StandardScaler().fit(x_train)
    rescaledX = scaler.transform(x_train)

    #Split data for model 2 classification train and validation    
    dataset = df.values
    X = dataset[:,:-1]
    Y = dataset[:,-1]
    # encode string class values as integers
    label_encoder = LabelEncoder()
    label_encoder = label_encoder.fit(Y)
    label_encoded_y = label_encoder.transform(Y) # multiclass target encode 
    #Define X features, Y target
    X = X
    Y = label_encoded_y

    scaler = MinMaxScaler(feature_range = (0,1))
    scaler.fit(X)
    X = scaler.transform(X)

    validation_size = 0.20 
    seed = 7 
    X_train, X_validation, Y_train, Y_validation = train_test_split(X, Y, test_size=validation_size, random_state=seed)

    ## Call function 
    if (int(sys.argv[1]) == 1):
        print("Applying LASSO regression for prediction.")
        LASSO(x_train, y_train)
    elif (int(sys.argv[1]) == 2): 
        print("Applying XGBoost classification for prediction.")
        XGB(X_train, X_validation, Y_train, Y_validation)
if __name__ == '__main__':
    main()





