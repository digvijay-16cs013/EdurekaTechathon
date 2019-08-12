# -*- coding: utf-8 -*-
"""
Created on Sun Aug  4 19:25:47 2019

@author: dell
"""

# To get predictions on the given data

import pickle, pandas as pd, numpy as np
from date_year import date_to_year

data = pd.read_csv('************File Name (with extension) goes here********************')

# Vectorizer, Encoder, Model Loader from pickle file
def pkl_file_loader(model_name):
    with open(f'pickle_files/{model_name}.pkl', 'rb') as f:
        model = pickle.load(f) 
    return model

# date_to_year_converter
data['year'] = date_to_year(data)

# loading pickle files
cv_modelID_t = pkl_file_loader('cv_modelID')
cv_cpc_t = pkl_file_loader('cv_cpc')
le_fuel_t = pkl_file_loader('le_fuel')
le_car_type_t = pkl_file_loader('le_car_type')
ohe_fuel_t = pkl_file_loader('ohe_fuel')
ohe_car_type_t = pkl_file_loader('ohe_car_type')
le_f18_t = pkl_file_loader('le_18')
sc_t = pkl_file_loader('sc')

# loading RandomForestRegressor model
RFR_model = pkl_file_loader('RFR')

# partial features, If the dataset already has car_price column then add column name i.e. car_price to the below list
partial_features = data.drop(['car_company', 'model_ID', 'registration_date', 'sold_date', 'fuel', 'car_type', 'car_paint_color'], axis = 1)

# for vectorization and encoding
def data_preprocessing(df, vec_or_en, col_name, ohe_cv): # dataframe, vectorizer_or_encoder, column_name
    if ohe_cv:
        return vec_or_en.transform(df[col_name]).toarray()
    return vec_or_en.transform(df[col_name])

# For one hot encoding
def one_hot_encoding(col_value, ohe):
    return ohe.transform(np.array(col_value).reshape(-1, 1)).toarray()[:, 1:]

# making other features ready.......
modID = data_preprocessing(data, cv_modelID_t, 'model_ID', True)
cpc = data_preprocessing(data, cv_cpc_t, 'car_paint_color', True)
fl = data_preprocessing(data, le_fuel_t, 'fuel', False)
c_type = data_preprocessing(data, le_car_type_t, 'car_type', False)
fl = one_hot_encoding(fl, ohe_fuel_t)
c_type = one_hot_encoding(c_type, ohe_car_type_t)

for i in range(1, 9):
    partial_features[f'feature{i}'] = le_f18_t.transform(partial_features[f'feature{i}'])

# combining all the features
f_test = np.concatenate((np.array(partial_features), modID, cpc, fl, c_type), axis = 1)

# standard scaling of features
f_test = sc_t.transform(f_test)

# getting test_results
test_results = pd.DataFrame(RFR_model.predict(f_test))
# storing results to csv file
test_results.to_csv('test_results.csv', index=False)

