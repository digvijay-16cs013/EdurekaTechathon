# -*- coding: utf-8 -*-
"""
Created on Sun Aug  4 20:33:03 2019

@author: dell
"""

def date_to_year(df):
    # splitting date, month and year(this is needed to convert the registration date into year)
    df['registration_date'] = df['registration_date'].str.split('-')
    df['sold_date'] = df['sold_date'].str.split('-')
    
    # converting date into year
    df['registration_date'] = df['registration_date'].apply(lambda ddmmyy: round(int(ddmmyy[-1]) + (int(ddmmyy[-2])/12) + (int(ddmmyy[-3])/365), 2))
    df['sold_date'] = df['sold_date'].apply(lambda ddmmyy: round(int(ddmmyy[-1]) + (int(ddmmyy[-2])/12) + (int(ddmmyy[-3])/365), 2))
    
    #difference b/w sold_date and registration date gives us the duration in years
    df['year'] = df['sold_date'] - df['registration_date']
    return df['year']