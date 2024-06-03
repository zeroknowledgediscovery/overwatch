#!/usr/bin/env python

import pandas as pd

df=pd.read_csv('./data_2019-01-01_to_2024-03-28.csv')
df=df[['dispatch_date_time','text_general_code','lat', 'lng']]
df['date']=pd.to_datetime(df.dispatch_date_time).dt.tz_localize(None)
df=df.drop('dispatch_date_time',axis=1).set_index('date')


allcodes=df.text_general_code.value_counts().index.values.astype(str)

code_categories=['violent','property','drug-alcohol-related']


violent=['Other Assaults',
 'Aggravated Assault No Firearm',
 'Aggravated Assault Firearm',
 'Robbery No Firearm',
 'Robbery Firearm',
 'Rape',
 'Homicide - Criminal',
 'Offenses Against Family and Children',
 'Homicide - Justifiable',
 'Homicide - Gross Negligence']
property=['Thefts',
 'Vandalism/Criminal Mischief',
 'Theft from Vehicle',
 'All Other Offenses',
 'Motor Vehicle Theft',
 'Fraud',
 'Burglary Residential',
 'Burglary Non-Residential',
 'Receiving Stolen Property',
 'Arson',
 'Forgery and Counterfeiting',
 'Prostitution and Commercialized Vice']
drug-alcohol-related=['Narcotic / Drug Law Violations']

df.to_csv('data.csv')





