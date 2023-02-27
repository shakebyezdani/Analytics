#!/usr/bin/env python
# coding: utf-8

# ### Problem and Objective
# ##### You are a cab rental start-up company. You have successfully run the pilot project and now want to launch your cab service across the country. You have collected the historical data from your pilot project and now have a requirement to apply analytics for fare prediction. You need to design a system that predicts the fare amount for a cab ride in the city.

# In[288]:


# import required libraries
import pandas as pd                  # for performing EDA
import numpy as np                   # for Linear Algebric operations
import matplotlib.pyplot as plt      # for Data Visualization
import seaborn as sns                # for Data Visualization
import os                            # getting access to input files
from collections import Counter
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor 
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score


# In[3]:


# working directory
os.chdir("E:/Analytics/Project 1")


# In[289]:


# loading data
train  = pd.read_csv("train_cab.csv", sep=",")
test   = pd.read_csv("test_cab.csv", sep=",")


# In[290]:


train.head()


# In[291]:


test.head()


# In[292]:


train.dtypes


# In[293]:


test.dtypes


# In[294]:


# checking size of train data
train.shape


# In[295]:


# checking size of test data
test.shape


# In[296]:


train.describe()


# In[297]:


test.describe()


# ### Data cleaning, missing value and outlier analysis

# In[298]:


# changing datatype of pickup_datetime variable from object to datetime
train['pickup_datetime'] =  pd.to_datetime(train['pickup_datetime'], format='%Y-%m-%d %H:%M:%S UTC', errors='coerce')


# In[299]:


print(train['pickup_datetime'].isnull().sum())


# In[300]:


# one value is null in pickup_datetime variable, so drop it.
train = train.drop(train[train['pickup_datetime'].isnull()].index, axis=0)


# In[301]:


train.shape


# In[302]:


# separate the pickup_datetime column into separate fields like year, month,day, day of the week, hour etc.
train['year'] = train['pickup_datetime'].dt.year
train['Month'] = train['pickup_datetime'].dt.month
train['Date'] = train['pickup_datetime'].dt.day
train['Day'] = train['pickup_datetime'].dt.dayofweek
train['Hour'] = train['pickup_datetime'].dt.hour
train['Minute'] = train['pickup_datetime'].dt.minute


# In[303]:


train.dtypes


# In[304]:


# checking null values
print(train['pickup_datetime'].isnull().sum())
print(train['year'].isnull().sum())
print(train['Month'].isnull().sum())
print(train['Date'].isnull().sum())
print(train['Day'].isnull().sum())
print(train['Hour'].isnull().sum())
print(train['Minute'].isnull().sum())


# In[305]:


# for test data
# changing datatype of pickup_datetime variable from object to datetime
test['pickup_datetime'] =  pd.to_datetime(test['pickup_datetime'], format='%Y-%m-%d %H:%M:%S UTC', errors='coerce')


# In[306]:


# separate the pickup_datetime column into separate fields like year, month,day, day of the week, hour etc.
test['year'] = test['pickup_datetime'].dt.year
test['Month'] = test['pickup_datetime'].dt.month
test['Date'] = test['pickup_datetime'].dt.day
test['Day'] = test['pickup_datetime'].dt.dayofweek
test['Hour'] = test['pickup_datetime'].dt.hour
test['Minute'] = test['pickup_datetime'].dt.minute


# In[307]:


test.dtypes


# In[308]:


# checking null values
print(test['pickup_datetime'].isnull().sum())
print(test['year'].isnull().sum())
print(test['Month'].isnull().sum())
print(test['Date'].isnull().sum())
print(test['Day'].isnull().sum())
print(test['Hour'].isnull().sum())
print(test['Minute'].isnull().sum())


# In[309]:


# Checking the fare_amount variable
#Converting fare_amount variable from object to numeric
train["fare_amount"] = pd.to_numeric(train["fare_amount"], errors = "coerce")


# In[310]:


train.dtypes


# In[311]:


train["fare_amount"].describe()


# In[312]:


# sort fare_amount in decending order to check outliers
train["fare_amount"].sort_values(ascending=False)


# In[313]:


# from above, we can see that there is huge difference in first three values of fare_amount
# first two values seems to be outlier in fare_amount, so drop them
train = train.drop(train[train["fare_amount"]> 454].index, axis=0)


# In[314]:


train.shape


# In[315]:


Counter(train["fare_amount"]<=0)


# In[316]:


# in fare_amount variable there are 4 values are present where fare is negative or zero. so drop them
train = train.drop(train[train["fare_amount"]<=0].index, axis=0)
train.shape


# In[317]:


print(train['fare_amount'].isnull().sum())


# In[318]:


# there are 25 rows which includes null fare_amount value. so drop those 25 rows
train = train.drop(train[train['fare_amount'].isnull()].index, axis=0)
train.shape


# In[319]:


# Checking the passenger_count variable
train["passenger_count"].describe()


# In[320]:


# any cab can not have more than 6 passengers, so we are dropping rows which includes more than 6 passengers
train = train.drop(train[train["passenger_count"]> 6].index, axis=0)


# In[321]:


train.shape


# In[322]:


Counter(train["passenger_count"]==0)


# In[323]:


#there are 57 rows in which passenger_count is 0. the cab should have atleast 1 passenger. so drop rows having 0 passenger.
train = train.drop(train[train["passenger_count"] == 0].index, axis=0)


# In[324]:


train.shape


# In[325]:


# sort passenger_count in ascending order
train["passenger_count"].sort_values(ascending=True)


# In[326]:


# 1 row includes passenger_count is 0.12 which is not possible. so drop that row
train = train.drop(train[train["passenger_count"] == 0.12].index, axis=0)


# In[327]:


print(train['passenger_count'].isnull().sum())


# In[328]:


# there are 55 rows which includes null passenger_count value. so drop those 55 rows
train = train.drop(train[train['passenger_count'].isnull()].index, axis=0)
train.shape


# In[329]:


test["passenger_count"].describe()


# In[330]:


print(test['passenger_count'].isnull().sum())


# In[331]:


train.describe()


# In[332]:


# from above, it is clear that max. value of pickup_latitude is 401.0833
# As we know that Lattitude ranges from (-90 to 90) and Longitude ranges from (-180 to 180)
# So, drop the rows which includes values outside the Lattitude and Longitude ranges


# In[333]:


# dropping one value of >90
train = train.drop((train[train['pickup_latitude']< -90]).index, axis=0)
train = train.drop((train[train['pickup_latitude']> 90]).index, axis=0)


# In[334]:


train.shape


# In[335]:


test.describe()


# ##### we have given pickup/drop latitude and longitude, so we need to calculate the distance using haversine formula

# In[336]:


# function for calculating the distance using haversine formula.
from math import radians, cos, sin, asin, sqrt

def haversine(a):
    lon1=a[0]
    lat1=a[1]
    lon2=a[2]
    lat2=a[3]
    """
    Calculate the great circle distance between two points 
    on the earth (specified in decimal degrees)
    """
    # convert decimal degrees to radians 
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    
    # haversine formula 
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c =  2 * asin(sqrt(a))
    # Radius of earth in kilometers is 6371
    km = 6371* c
    return km


# In[337]:


train['distance'] = train[['pickup_longitude','pickup_latitude','dropoff_longitude','dropoff_latitude']].apply(haversine,axis=1)


# In[338]:


test['distance'] = test[['pickup_longitude','pickup_latitude','dropoff_longitude','dropoff_latitude']].apply(haversine,axis=1)


# In[339]:


train.head()


# In[340]:


test.head()


# In[341]:


train['distance'].describe()


# In[342]:


Counter(train['distance'] == 0)


# In[343]:


# distance can not be 0 km, so drop the rows which includes distance 0 km
train = train.drop(train[train['distance']== 0].index, axis=0)
train.shape


# In[344]:


# arrange decending order of distance to check outlier
train['distance'].sort_values(ascending=False)[0:40]


# In[345]:


# from above, it is clear that first 23 distance values are outliers as in first 23 values, distance is in thousands.
# And after first 23 values, distance goes down to 129 km
# remove the rows whose distance values is very high which is more than 129 km
train = train.drop(train[train['distance'] > 130].index, axis=0)
train.shape


# In[346]:


test['distance'].describe()


# In[347]:


Counter(test['distance'] == 0)


# In[348]:


# distance can not be 0 km, so drop the rows which includes distance 0 km
test = test.drop(test[test['distance']== 0].index, axis=0)
test.shape


# In[349]:


test['distance'].sort_values(ascending=False)


# ##### we have splitted the pickup date time variable into different variables like month, year, day etc and we have created distance using pickup and drop longitudes and latitudes so we will drop pickup date time, pickup and drop longitudes and latitudes variables.

# In[350]:


train_col_drop = ['pickup_datetime', 'pickup_longitude', 'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude', 'Minute']
train = train.drop(train_col_drop, axis = 1)


# In[351]:


train.head()


# In[352]:


train.dtypes


# In[353]:


train['passenger_count'] = train['passenger_count'].astype('int64')
train.dtypes


# In[354]:


train.isnull().sum()


# In[355]:


train.shape


# In[356]:


test_col_drop = ['pickup_datetime', 'pickup_longitude', 'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude', 'Minute']
test = test.drop(test_col_drop, axis = 1)


# In[357]:


test.head()


# In[358]:


test.dtypes


# In[359]:


test.isnull().sum()


# In[360]:


test.shape


# In[361]:


#taking copy of the data

train_data_df1 = train.copy()
test_data_df1 = test.copy()

#train = train_data_df1.copy()
#test = test_data_df1.copy()


# #### boxplot and scatter plot analysis for outlier detection

# In[362]:


# checking boxplot of continous variables

get_ipython().run_line_magic('matplotlib', 'inline')
plt.boxplot(train['fare_amount'])


# In[363]:


# checking boxplot of continous variables
plt.boxplot(train['distance'])


# In[364]:


# scatter plot on continous variables

plt.scatter(x=train.fare_amount, y=train.index)
plt.ylabel('Index')
plt.xlabel('fare_amount')
plt.show()


# In[365]:


# scatter plot on continous variables

plt.scatter(x=train.distance, y=train.index)
plt.ylabel('Index')
plt.xlabel('distance')
plt.show()


# ##### from above scatter plots, it is clear that fare greater than 80 is outlier and distance greater than 30 km is outlier. so, drop the rows which includes fare greater than 80 and distance greater than 30 km.

# In[366]:


train = train.drop(train[train['fare_amount'] > 80].index, axis=0)
train.shape


# In[367]:


train = train.drop(train[train['distance'] > 30].index, axis=0)
train.shape


# In[368]:


# scatter plot on continous variables after removing outlier

plt.scatter(x=train.fare_amount, y=train.index)
plt.ylabel('Index')
plt.xlabel('fare_amount')
plt.show()


# In[369]:


# scatter plot on continous variables after removing outlier

plt.scatter(x=train.distance, y=train.index)
plt.ylabel('Index')
plt.xlabel('distance')
plt.show()


# In[370]:


#Plot for fare_amount variation across distance

plt.scatter(y=train['distance'], x=train['fare_amount'])
plt.xlabel('fare')
plt.ylabel('distance')
plt.show()


# In[371]:


# Count plot on passenger count

sns.countplot(train['passenger_count'])


# In[372]:


# check relationship between fare and passengers

plt.scatter(x=train['passenger_count'], y=train['fare_amount'], s=10)
plt.xlabel('No. of Passengers')
plt.ylabel('Fare')
plt.show()


# In[373]:


# check relationship between fare and date

plt.scatter(x=train['Date'], y=train['fare_amount'], s=10)
plt.xlabel('Dates')
plt.ylabel('Fare')
plt.show()


# In[374]:


# check relationship between fare and day

plt.scatter(x=train['Day'], y=train['fare_amount'], s=10)
plt.xlabel('Days')
plt.ylabel('Fare')
plt.show()


# In[375]:


# check relationship between fare and hour

plt.scatter(x=train['Hour'], y=train['fare_amount'], s=10)
plt.xlabel('Hours')
plt.ylabel('Fare')
plt.show()


# ### Feature Selection

# In[376]:


# Correlation Analysis
# generating heatmap

cnames = ['fare_amount', 'distance']
df_corr = train.loc[:,cnames]
f, ax = plt.subplots(figsize=(7, 5))

# correlation matrix
corr = df_corr.corr()

sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool), cmap=sns.diverging_palette(220, 10, as_cmap=True), square=True, ax=ax)


# In[377]:


# checking VIF for multicolinerity

from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant

VIF_df = add_constant(train.iloc[:,1:8])
pd.Series([variance_inflation_factor(VIF_df.values, i) 
               for i in range(VIF_df.shape[1])], 
              index=VIF_df.columns)


# In[378]:


#taking copy of the data

train_data_df2 = train.copy()
test_data_df2 = test.copy()

#train = train_data_df2.copy()
#test = test_data_df2.copy()


# ##### from above VIF values are less than 10 for each variable, so there is no multicolinerity exists.

# ### Feature Scaling

# In[379]:


# check histogram of fare_amount variable

plt.hist(train['fare_amount'], bins='auto')


# In[380]:


# check histogram of distance variable

plt.hist(train['distance'], bins='auto')


# In[381]:


# performing normalization

cnames = ['fare_amount', 'distance']
for i in cnames:
    print(i)
    train[i] = (train[i] - train[i].min())/(train[i].max() - train[i].min())


# In[382]:


train.head()


# ## Models Development

# In[383]:


# split train data into train and test
X_train, X_test, y_train, y_test = train_test_split(train.iloc[:,1:8], train.iloc[:,0], test_size = 0.2, random_state = 1)


# In[384]:


print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)


# ### Linear Regression Model

# In[385]:


# Build model on train data
LR = LinearRegression().fit(X_train , y_train)


# In[386]:


# predict on train data
pred_train_LR = LR.predict(X_train)


# In[387]:


# predict on test data
pred_test_LR = LR.predict(X_test)


# In[388]:


# Model Evaluation

# calculate RMSE on train data
RMSE_train_LR= np.sqrt(mean_squared_error(y_train, pred_train_LR))

# calculate RMSE on test data
RMSE_test_LR = np.sqrt(mean_squared_error(y_test, pred_test_LR))


# In[389]:


print("RMSE on training data = "+str(RMSE_train_LR))
print("RMSE on test data = "+str(RMSE_test_LR))


# In[390]:


# calculate R^2 on train data
r2_train_LR = r2_score(y_train, pred_train_LR)

# calculate R^2 on test data
r2_test_LR = r2_score(y_test, pred_test_LR)


# In[391]:


print("r2 on training data = "+str(r2_train_LR))
print("r2 on test data = "+str(r2_test_LR))


# ### Decision Tree Model

# In[392]:


# Build model on train data
DT = DecisionTreeRegressor(max_depth = 2).fit(X_train, y_train)


# In[393]:


# predict on train data
pred_train_DT = DT.predict(X_train)

# predict on test data
pred_test_DT = DT.predict(X_test)


# In[394]:


# Model Evaluation

# calculate RMSE on train data
RMSE_train_DT = np.sqrt(mean_squared_error(y_train, pred_train_DT))

# calculate RMSE on test data
RMSE_test_DT = np.sqrt(mean_squared_error(y_test, pred_test_DT))


# In[395]:


print("RMSE on training data = "+str(RMSE_train_DT))
print("RMSE on test data = "+str(RMSE_test_DT))


# In[396]:


# calculate R^2 on train data
r2_train_DT = r2_score(y_train, pred_train_DT)

# calculate R^2 on test data
r2_test_DT = r2_score(y_test, pred_test_DT)


# In[397]:


print("r2 on training data = "+str(r2_train_DT))
print("r2 on test data = "+str(r2_test_DT))


# ### Random Forest Model

# In[398]:


# Build model on train data
RF = RandomForestRegressor(n_estimators = 300).fit(X_train, y_train)


# In[399]:


# predict on train data
pred_train_RF = RF.predict(X_train)

# predict on test data
pred_test_RF = RF.predict(X_test)


# In[400]:


# Model Evaluation

# calculate RMSE on train data
RMSE_train_RF = np.sqrt(mean_squared_error(y_train, pred_train_RF))

# calculate RMSE on test data
RMSE_test_RF = np.sqrt(mean_squared_error(y_test, pred_test_RF))


# In[401]:


print("RMSE on training data = "+str(RMSE_train_RF))
print("RMSE on test data = "+str(RMSE_test_RF))


# In[402]:


# calculate R^2 on train data
r2_train_RF = r2_score(y_train, pred_train_RF)

# calculate R^2 on test data
r2_test_RF = r2_score(y_test, pred_test_RF)


# In[403]:


print("r2 on training data = "+str(r2_train_RF))
print("r2 on test data = "+str(r2_test_RF))


# ### Gradient Boosting

# In[404]:


# Build model on train data
GB = GradientBoostingRegressor().fit(X_train, y_train)


# In[405]:


# predict on train data
pred_train_GB = GB.predict(X_train)

# predict on test data
pred_test_GB = GB.predict(X_test)


# In[406]:


# Model Evaluation

# calculate RMSE on train data
RMSE_train_GB = np.sqrt(mean_squared_error(y_train, pred_train_GB))

# calculate RMSE on test data
RMSE_test_GB = np.sqrt(mean_squared_error(y_test, pred_test_GB))


# In[407]:


print("RMSE on training data = "+str(RMSE_train_GB))
print("RMSE on test data = "+str(RMSE_test_GB))


# In[408]:


# calculate R^2 on train data
r2_train_GB = r2_score(y_train, pred_train_GB)

# calculate R^2 on test data
r2_test_GB = r2_score(y_test, pred_test_GB)


# In[409]:


print("r2 on training data = "+str(r2_train_GB))
print("r2 on test data = "+str(r2_test_GB))


# ## Applying Hyper-parameter Tuning for optimizing the results

# ##### there are two ways to apply hyper-parameter tuning
# ##### 1. RandomizedSearchCV
# ##### 2. GridSearchCV

# In[410]:


# 1. RandomizedSearchCV

from sklearn.model_selection import RandomizedSearchCV

# RandomizedSearchCV on Random Forest Model

RFR = RandomForestRegressor(random_state = 0)
n_estimator = list(range(1,20,2))
depth = list(range(1,100,2))

# Create the random grid
rand_grid = {'n_estimators': n_estimator, 'max_depth': depth}

randomcv_rf = RandomizedSearchCV(RFR, param_distributions = rand_grid, n_iter = 5, cv = 5, random_state=0)
randomcv_rf = randomcv_rf.fit(X_train, y_train)
predictions_RFR = randomcv_rf.predict(X_test)

best_params_RFR = randomcv_rf.best_params_

best_estimator_RFR = randomcv_rf.best_estimator_

predictions_RFR = best_estimator_RFR.predict(X_test)

# calculate R^2
RFR_r2 = r2_score(y_test, predictions_RFR)

# calculate RMSE
RFR_rmse = np.sqrt(mean_squared_error(y_test, predictions_RFR))

print('RandomizedSearchCV - Random Forest Regressor Model Performance:')
print('Best Parameters = ',best_params_RFR)
print('R-squared = {:0.2}.'.format(RFR_r2))
print('RMSE = ',RFR_rmse)


# In[411]:


# RandomizedSearchCV on gradient boosting model

GBR = GradientBoostingRegressor(random_state = 0)
n_estimator = list(range(1,20,2))
depth = list(range(1,100,2))

# Create the random grid
rand_grid = {'n_estimators': n_estimator, 'max_depth': depth}

randomcv_gb = RandomizedSearchCV(GBR, param_distributions = rand_grid, n_iter = 5, cv = 5, random_state=0)
randomcv_gb = randomcv_gb.fit(X_train, y_train)
predictions_gb = randomcv_gb.predict(X_test)

best_params_gb = randomcv_gb.best_params_

best_estimator_gb = randomcv_gb.best_estimator_

predictions_gb = best_estimator_gb.predict(X_test)

# calculate R^2
gb_r2 = r2_score(y_test, predictions_gb)

# calculate RMSE
gb_rmse = np.sqrt(mean_squared_error(y_test, predictions_gb))

print('RandomizedSearchCV - Gradient Boosting Model Performance:')
print('Best Parameters = ',best_params_gb)
print('R-squared = {:0.2}.'.format(gb_r2))
print('RMSE = ', gb_rmse)


# In[412]:


# 2. GridSearchCV

from sklearn.model_selection import GridSearchCV

# GridSearchCV on Random Forest Model

rfr_gs = RandomForestRegressor(random_state = 0)
n_estimator = list(range(11,20,1))
depth = list(range(5,15,2))

# Create the grid
grid_search = {'n_estimators': n_estimator, 'max_depth': depth}

## Grid Search Cross-Validation with 5 fold CV
gridcv_rf = GridSearchCV(rfr_gs, param_grid = grid_search, cv = 5)
gridcv_rf = gridcv_rf.fit(X_train,y_train)

best_params_GRF = gridcv_rf.best_params_
best_estimator_GRF = gridcv_rf.best_estimator_

#Apply model on test data
predictions_GRF = best_estimator_GRF.predict(X_test)

# calculate R^2
GRF_r2 = r2_score(y_test, predictions_GRF)

# calculate RMSE
GRF_rmse = np.sqrt(mean_squared_error(y_test, predictions_GRF))

print('GridSearchCV - Random Forest Regressor Model Performance:')
print('Best Parameters = ',best_params_GRF)
print('R-squared = {:0.2}.'.format(GRF_r2))
print('RMSE = ',(GRF_rmse))


# In[413]:


# GridSearchCV on gradient boosting model

gbr_gs = GradientBoostingRegressor(random_state = 0)
n_estimator = list(range(11,20,1))
depth = list(range(5,15,2))

# Create the grid
grid_search = {'n_estimators': n_estimator, 'max_depth': depth}

# Grid Search Cross-Validation with 5 fold CV
gridcv_gb = GridSearchCV(gbr_gs, param_grid = grid_search, cv = 5)
gridcv_gb = gridcv_gb.fit(X_train,y_train)

best_params_Ggb = gridcv_gb.best_params_
best_estimator_Ggb = gridcv_gb.best_estimator_

#Apply model on test data
predictions_Ggb = best_estimator_Ggb.predict(X_test)

# calculate R^2
Ggb_r2 = r2_score(y_test, predictions_Ggb)

# calculate RMSE
Ggb_rmse = np.sqrt(mean_squared_error(y_test, predictions_Ggb))

print('Grid Search CV Gradient Boosting regression Model Performance:')
print('Best Parameters = ',best_params_Ggb)
print('R-squared = {:0.2}.'.format(Ggb_r2))
print('RMSE = ',(Ggb_rmse))


# ##### from above models, it is clear that GridSearchCV on Random Forest Model is providing best results having R-squared = 0.8 and RMSE =  0.05243

# ## Fare prediction on the Test data

# ##### we have already cleaned the test dataset, so we are applying GridSearchCV on Random Forest Model on Test data.
# ##### Let's create standalone model on entire training dataset.

# In[414]:


train = train_data_df2.copy()


# In[415]:


X = train.drop('fare_amount', axis=1).values
y = train['fare_amount'].values


# In[416]:


print(X.shape)
print(y.shape)


# In[417]:


# GridSearchCV for random Forest model - test data fare prediction

rfr_test = RandomForestRegressor(random_state = 0)
n_estimator = list(range(11,20,1))
depth = list(range(5,15,2))

# Create the grid
grid_search = {'n_estimators': n_estimator, 'max_depth': depth}

## Grid Search Cross-Validation with 5 fold CV
gridcv_rf_test = GridSearchCV(rfr_test, param_grid = grid_search, cv = 5)
gridcv_rf_test = gridcv_rf_test.fit(X, y)

best_params_GRF_test = gridcv_rf_test.best_params_
best_estimator_GRF_test = gridcv_rf_test.best_estimator_

# Apply model on test data
predictions_GRF_test = best_estimator_GRF_test.predict(test)

print('Best Parameters = ',best_params_GRF_test)


# In[418]:


predictions_GRF_test


# In[419]:


test['Predicted_fare'] = predictions_GRF_test


# In[420]:


test.head()


# In[421]:


test.to_csv('test_predicted.csv')


# In[ ]:




