# -*- coding: utf-8 -*-
"""
Created on Sat Aug  3 21:45:48 2019

@author: dell
"""

# importing essential packages

import pandas as pd, numpy as np, matplotlib.pyplot as plt, pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import LinearSVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
from date_year import date_to_year

# loading the dataset

car_data = pd.read_csv('data.csv')

# Data Preprocessing, Data Exploration and Data Modelling

# Getting first five rows of the dataset
print(car_data.head())

# gettind info of the dataset
print(car_data.info())

# checking null values in each column
print(car_data.isnull().any(axis = 0)) # no null values present in any column

# to get unique values of any column
def uniq(col):
    return col.unique()

car_data['year'] = date_to_year(car_data)

# to get unique values of only object type columns
for column in car_data.columns:
    if car_data[column].dtypes == 'O':
        print(column, ':', uniq(car_data[column]))

# Relation b/w car_price and engine_power
engine_power = car_data['engine_power']
car_price = car_data['car_price']
plt.plot(engine_power, car_price, 'o', markersize = 3, alpha = 0.3)
plt.xlabel('Engine Power')
plt.ylabel('Car Price')
plt.show() # this visual is column centric also acc. to this visual one engine_power has value 0.

# therefore cheching the minimum 5 values
print(engine_power.sort_values()[:5]) # minimum entry is 0 at index 1298.

# dropping row with 1298 index
car_data = car_data.drop(1298)

# Applying some jittering to make more appropriate visual
engine_power = engine_power + np.random.normal(0, 2, size=len(engine_power))
car_price = car_price + np.random.normal(0, 1.5, size=len(car_price))
plt.plot(engine_power, car_price, 'o', markersize = 1, alpha = 0.2)
plt.xlabel('Engine Power')
plt.ylabel('Car Price')
# focusing on engine_power 100 to 200 and price 0 to 50000
plt.axis([100, 200, 0, 50000])
plt.show() # according to scatter plot car_price is increasing as engine_power is increasing.

# Relation b/w car_price and mileage
mileage = car_data['mileage']
plt.plot(mileage, car_data['car_price'], 'o', markersize=3, alpha=0.2)
plt.xlabel('Mileage')
plt.ylabel('Car price')
plt.show()  # car_price is decreasing as mileage is increasing

# getting the minimum 5 values of mileage
print(mileage.sort_values()[:5]) #-64 is present so we have to drop it becz no car has mileage <= 0.
car_data = car_data.drop(2931) # dropping the row which has -64 as mileage

# handling outliers
boxplot = car_data.boxplot(return_type = 'dict')
plt.xticks(rotation='vertical')
plt.show()
outliers = [flier.get_ydata() for flier in boxplot['fliers']]
car_data = car_data[car_data['mileage'].apply(lambda m: m not in outliers[0])]
car_data = car_data[car_data['car_price'].apply(lambda cp: cp not in outliers[-2])]

# Extracting features and labels
# Clearly, car_company does not affect the price therefore it will not be considered in features
features_set1 = car_data.drop(['car_company', 'model_ID', 'registration_date', 'sold_date', 'fuel', 'car_type', 'car_paint_color', 'car_price'], axis = 1)
# these below columns has to be vectorized
labels = np.array(car_data['car_price'])

# vectorizing modelID
cv_modelID = CountVectorizer()
model_ID = cv_modelID.fit_transform(car_data['model_ID']).toarray()

# vectorizing car_paint_color
cv_cpc = CountVectorizer()
car_paint_color = cv_cpc.fit_transform(car_data['car_paint_color']).toarray()

print(features_set1.info())

# label encoding fuel column
le_fuel = LabelEncoder()
fuel = le_fuel.fit_transform(car_data['fuel'])

# label encoding car_type column
le_car_type = LabelEncoder()
car_type = le_car_type.fit_transform(car_data['car_type'])

# Applying OneHotEncoder on fuel and car_type column
ohe_fuel = OneHotEncoder()
fuel = ohe_fuel.fit_transform(np.array(fuel).reshape(-1, 1)).toarray()[:, 1:]

ohe_car_type = OneHotEncoder()
car_type = ohe_car_type.fit_transform(np.array(car_type).reshape(-1, 1)).toarray()[:, 1:]

# Applying label encoding on feature1 - feature8
le_f18 = LabelEncoder()
features_set1['feature1'] = le_f18.fit_transform(features_set1['feature1'])

for i in range(2, 9):
    features_set1[f'feature{i}'] = le_f18.transform(features_set1[f'feature{i}'])

# Features_set1, ModelID, CarPaintColor, Fuel, CarType are features therefore combining them.
features = np.concatenate((np.array(features_set1), model_ID, car_paint_color, fuel, car_type), axis = 1)

# now scaling of features is required
sc = StandardScaler()
features = sc.fit_transform(features)

# splitting training and testing data
features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.02, random_state = 0)

# Applying different models

# LinearRegressor model
lr = LinearRegression()
lr.fit(features_train, labels_train)
print(lr.score(features_train, labels_train))    # 0.7949933486551278
print(lr.score(features_test, labels_test))      # 0.8119678725321358

# Linear Support Vector Regressor
lsvr = LinearSVR()
lsvr.fit(features_train, labels_train)
print(lsvr.score(features_train, labels_train))  # -4.115532120789137
print(lsvr.score(features_test, labels_test))    # -3.6439529728140565

# K-Neighbors Regressor
knr = KNeighborsRegressor(n_neighbors=5, p=2)
knr.fit(features_train, labels_train)
print(knr.score(features_train, labels_train))   # 0.756401841732305
print(knr.score(features_test, labels_test))     # 0.7403518498259278

# Decision Tree Regressor
dtr = DecisionTreeRegressor()
dtr.fit(features_train, labels_train)
print(dtr.score(features_train, labels_train))   # 1.0
print(dtr.score(features_test, labels_test))     # 0.7653735804027293

# Random Forest Regressor
rfr = RandomForestRegressor(n_estimators = 20, random_state = 0)
rfr.fit(features_train, labels_train)
print(rfr.score(features_train, labels_train))   # 0.9643463357794885
print(rfr.score(features_test, labels_test))     # 0.8357725138873492

# Predicting labels for test_data
labels_pred = rfr.predict(features_test)

# mean_squared_error for RandomForestRegressor
print(np.sqrt(mean_squared_error(labels_test, labels_pred))) # 2296.944235844798

# For comparison making a dataframe with actual and predicted values
predictions = pd.DataFrame(list(zip(labels_test, labels_pred)), columns = ['Actual', 'Predicted'])
print(predictions)

# to deploy model and vectorizers
def deploy(name, model):
    with open(f'pickle_files/{name}.pkl', 'wb') as f:
        pickle.dump(model, f)
        
models = {'cv_modelID':cv_modelID, 'cv_cpc':cv_cpc, 'le_fuel':le_fuel, 'le_car_type':le_car_type, 'ohe_fuel':ohe_fuel, 'ohe_car_type':ohe_car_type, 'le_18':le_f18, 'sc':sc, 'RFR':rfr}
for name, model in models.items():
    deploy(name, model)
    
# -------------------------- Model training and Testing Completed -----------------------------------