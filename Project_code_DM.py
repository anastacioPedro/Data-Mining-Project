#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 23 19:14:21 2018

@author: fruechtnicht
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Oct 19 19:24:28 2018

@author: dominika.leszko
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

df=pd.read_csv(r'C:\Users\dominika.leszko\Desktop\NOVA IMS\Data Mining\DM Project\A2Z Insurance.csv')

#from sklearn.preprocessing import Imputer
df = pd.DataFrame(df)
pd.set_option('display.max_columns', None)


df.info()
df.describe()
#replacing empty strings with nan
df = df.replace({'': np.nan})

#Renaming columns for easier analysis
df.columns.values

coldict={'Customer Identity':'CustId', 'First Policy´s Year':'1stPolYear', 'Brithday Year':'BirthYear',
       'Educational Degree':'EduDegree', 'Gross Monthly Salary':'GrossMthSalary',
       'Geographic Living Area':'GeoLivArea', 'Has Children (Y=1)':'HasChild',
       'Customer Monetary Value':'CustMonetVal', 'Claims Rate':'ClaimRate', 'Premiums in LOB: Motor':'PremLOBMotor',
       'Premiums in LOB: Household':'PremLOBHousehold', 'Premiums in LOB: Health':'PremLOBHealth',
       'Premiums in LOB:  Life':'PremLOBLife', 'Premiums in LOB: Work Compensations':'PremLOBWorkCompensation'}

df.rename(columns=coldict, inplace=True)

##############################Handling Outliers##############################################################

df.shape#10296 rows, 14 columns

df['1stPolYear'].describe()
#Drop values >2016, as the database comes from 2016
df = df.drop(df[df['1stPolYear']>2016].index)
sns.kdeplot(df['1stPolYear'])


df['BirthYear'].describe()
#Drop values <1900
df=df.drop(df[df['BirthYear']<1900].index)
df['BirthYear'].hist(bins=50)

df['GrossMthSalary'].describe()
sns.boxplot(x=df['GrossMthSalary'])
#Drop Salary>30000
df=df.drop(df[df['GrossMthSalary']>30000].index)
df['GrossMthSalary'].hist(bins=50)


sns.boxplot(x=df['PremLOBMotor'])
#Drop PremLOBMotor>2000
df=df.drop(df[df['PremLOBMotor']>2000].index)
sns.kdeplot(df['PremLOBMotor'])


df['PremLOBHousehold'].hist(bins=100)# SKEWED!!!!!!!!!
plt.xlim(0,4000)
sns.boxplot(x=df['PremLOBHousehold'])
plt.xlim(0,2000)
df['PremLOBHousehold'].describe()
#LOG OF VALUE 0? 
df['PremLOBHousehold'].min()#min is -75
df['PremLOBHousehold']=np.log(df['PremLOBHousehold'] + 1 - min(df['PremLOBHousehold']))
df['PremLOBHousehold'].min()#min is 0
sns.kdeplot(df['PremLOBHousehold'])
df['PremLOBHousehold'].describe()
#DELETE OUTLIERS NOTHING DROPPED?
test1=[x for x in df['PremLOBHousehold'] if (x < (df['PremLOBHousehold'].mean() + 2.5*df['PremLOBHousehold'].std()) or x > (df['PremLOBHousehold'].mean() - 2.5*df['PremLOBHousehold'].std()))]


sns.boxplot(x=df['PremLOBHealth'])
#Drop PremLOBHealth>5000
df=df.drop(df[df['PremLOBHealth']>5000].index)
sns.kdeplot(df['PremLOBHealth'])


plt.figure(figsize=(8,6))
df['PremLOBLife'].hist()
sns.boxplot(x=df['PremLOBLife'])#SKEWED!!!!!!!!
sns.kdeplot(df['PremLOBLife'])
df['PremLOBLife'].hist()
#LOG
df['PremLOBLife'].min()#min is -7
df['PremLOBLife']=np.log(df['PremLOBLife'] + 1 - min(df['PremLOBLife']))
df['PremLOBLife'].min()#min is 0
#DELETE OUTLIERS
test2=[x for x in df['PremLOBLife'] if (x < (df['PremLOBLife'].mean() + 3*df['PremLOBLife'].std()) or x > (df['PremLOBLife'].mean() - 3*df['PremLOBLife'].std()))]
len(test2)


sns.boxplot(x=df['PremLOBWorkCompensation'])
#drop >5000
df=df.drop(df[df['PremLOBWorkCompensation']>5000].index)
sns.kdeplot(df['PremLOBWorkCompensation'])#SKEWED!!!! Plot starts below 0 in kde??????? why such small numbers before we use log?????
#LOG
df['PremLOBWorkCompensation'].min()#min is -12
df['PremLOBWorkCompensation']=np.log(df['PremLOBWorkCompensation'] + 1 - min(df['PremLOBWorkCompensation']))
df['PremLOBWorkCompensation'].min()#min is 0
#DELETE OUTLIERS
test3=[x for x in df['PremLOBWorkCompensation'] if (x < (df['PremLOBWorkCompensation'].mean() + 4*df['PremLOBWorkCompensation'].std()))]
len(test3)#10138; 146 dropped


#############################Handling null values################################################################

df.isna().any()
df.isnull().sum(axis=0)

#### Replacing missing data with Regression ###################################

# We need to do same the other way round: for Salary based on regression on Bday, is it correct? 

y_train = df['BirthYear']
y_test = y_train.loc[y_train.index.isin(list(y_train.index[(y_train >= 0)== False]))]
X_train = pd.DataFrame(df['GrossMthSalary'].loc[y_train.index.isin(list(y_train.index[(y_train >= 0)== True ]))])
X_test  = pd.DataFrame(df['GrossMthSalary'].loc[y_train.index.isin(list(y_train.index[(y_train >= 0)== False]))])
y_train = y_train.dropna()

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)
y_pred= regressor.predict(X_test)

i=0
for index in y_test.index:
    df['BirthYear'][index] = y_pred[i]
    i+=1

#####Replacing missing data with K-Nearest Neighbors #####################################################

X = df.drop(columns=['HasChild'])
y = df['HasChild']

y_train = y
y_test = y_train.loc[y_train.index.isin(list(y_train.index[(y_train >= -1)== False]))]
X_train = pd.DataFrame(X.loc[y_train.index.isin(list(y_train.index[(y_train >= -1)== True]))])#DOMINIKA:Shouldnt be >-1?
X_test = pd.DataFrame(X.loc[y_train.index.isin(list(y_train.index[(y_train >= -1)== False]))])
y_train = y_train.dropna()

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

i=0
for index in y_test.index:
    df['HasChild'][index] = y_pred[i]
    i+=1
#### KNN replacement  v.2 ##########################################################
#pred_df = df[df['HasChild'].null()]
#train_df = df[df['HasChild'].notnull()]

#
#X = train_df.drop(['HasChild'], axis =1)
#y = train_df['HasChild']
#X_test = pred_df.drop(['HasChild'], axis =1)
#y_test = pred_df['HasChild']
#
#
#from sklearn.neighbors import KNeighborsClassifier
#knn = KNeighborsClassifier(n_neighbors = 3, metric = 'minkowski', p = 2)
#knn.fit(X, y)
## Predicting a new result
#y_pred = knn.predict(X_test)
#
#i=0
#for index in y_test.index:
#    df['HasChild'][index] = y_pred[i]
#
#    i+=1
##########################################       
####Replacing missing data with Decision Tree   ##########################################################
    
X = df.drop(columns=['EduDegree'])
y = df['EduDegree']

y_train = y
y_test = y.loc[y.isin(list(y[y.isna()== True]))]
X_train = pd.DataFrame(X.loc[y.isin(list(y[y.isna()== False]))])
X_test  = pd.DataFrame(X.loc[y.isin(list(y[y.isna()== True ]))])
y_train = y.dropna()

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelEncoder1 = LabelEncoder()
y_train = labelEncoder1.fit_transform(y_train)

# Fitting Decision Tree Regression to the dataset
from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state = 0)
regressor.fit(X_train, y_train)

# Predicting a new result
y_pred = regressor.predict(X_test)

i=0
for index in y_test.index:
    df['EduDegree'][index] = y_pred[i]
    i+=1       
###############################################################################


#1st Year-KNN (30 nulls - mean/median?)
plt.figure(figsize=(8,6))
df['1stPolYear'].hist()

#Education-NN (17 nulls -mean/median?)
plt.figure(figsize=(8,6))
df['EduDegree'].hist()

#Geogrpahy-NN (1 null- missing a lot of other columns, DROP!)
df=df.dropna(subset=['GeoLivArea'])

#Has children-NN (21 nulls, KNN/replace with 1?) after doing knn if 80% has 1 then its good!
df['HasChild']=df['HasChild'].fillna(1)

#Customer Monet-NO NULLS
#Claims Ratio - NO NULLS

#Replace Premium LOB values with 0 --> null means not insured, didn't pay
df['PremLOBWorkCompensation']=df['PremLOBWorkCompensation'].fillna(0)
df['PremLOBLife']=df['PremLOBLife'].fillna(0)
df['PremLOBHealth']=df['PremLOBHealth'].fillna(0)
df['PremLOBMotor']=df['PremLOBMotor'].fillna(0)



##################################Exploratory Data Analysis#####################################################################
#to verify: drop null before visualizing?
sns.set(rc={'figure.figsize':(20,20)})
sns.heatmap(df.corr(), annot=True)

sns.pairplot(df)

#DO MORE VISUALIZATIONS, search for relations!

################################FEATURE ENGINEERING AND SELECTION############################################################

#Drop CustId
df.drop(['CustId'], axis=1, inplace=True)

#EduDegree is an object. Convert to ordinal.
df['EduDegree'].head(10)
df.info()

ord_edu=df['EduDegree'].str.split(' - ', 1, expand=True)#Is it ok to use ordinal here? How is distance going to be measured then? PS+HS != Phd.
#do we include categorical variables in clustering where we calculate distance?
#How label encoder would code these? alphabetic?
ord_edu=ord_edu[0].astype(int)
df['ord_edu']=ord_edu
df.drop('EduDegree', inplace=True, axis=1)

df.info()
df.head()

#Feature Transformation_log#should we do log of all numeric variables? or is Standard Scaler going to do the job (normalize)
numeric_subset = df.select_dtypes('number')
#remove ord_edu from this!!!
for col in numeric_subset.columns:
        numeric_subset['log_' + col] = np.log(numeric_subset[col])
df2=pd.concat([df['EduDegree'], numeric_subset], axis=1)
df2.shape
df2.describe()


#DO DATA SCALING AND MAKE MEAN=1,STD=1 (Z SCORE). We need it scale cause its clustering

#
###############Z Score#same as min max and standardscaler?
#
#from scipy import stats
#z = np.abs(stats.zscore(df))
#print(z)
#
#threshold = 3
#print(np.where(z > 3))#returns 1 array of rows and 2 array of columns of outliers
#
###########Interquartile range IQR
#a = np.array(df['PremLOBHousehold'])
#a = np.sort(a)
#Q1 = df['PremLOBHousehold'].quantile(0.002)
#Q3 = df.quantile(0.75)
#IQR = Q3 - Q1
#print(IQR)
#
#print(df < (Q1 - 1.5 * IQR)) |(df > (Q3 + 1.5 * IQR))
##returns trues and falses
#
##Filtering out outliers detected by Z Score
#
#df_oZ = df[(z < 3).all(axis=1)]
#df.shape
#df_oZ.shape#Z score filters out 1101 outliers.its 10% of dataset
#
##Filtering out outliers detected by IQR
#
#df_oIQR = df[~((df < (Q1 - 1.5 * IQR)) |(df > (Q3 + 1.5 * IQR))).any(axis=1)]
#df_oIQR.shape#IQR filters out 2816 outliers. its 27%
#
#df.head()
#
###
#
##from sklearn.preprocessing import Imputer#DONE AT THE TOP
##df = df.where(df!='')  # Replacing empty values with NaN
#df=df.values
#imputer = Imputer(missing_values=np.nan, strategy = 'median', axis = 0)
#imputer = imputer.fit(df[:,-2:-1])
#
#df[:,-2:-1] = imputer.transform(df[:,-2:-1])


#############################


