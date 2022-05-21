import os
import pandas as pd

pd.set_option('display.max_columns', 15)

df = pd.read_csv("advertising_pre.csv")

print(df.describe(include='all'))

time = df['Timestamp']

df['ymd'] = list(map(lambda x: x.split()[0], df['Timestamp']))
df['hms'] = list(map(lambda x: x.split()[1], df['Timestamp']))

df['month'] = list(map(lambda x: x.split('-')[1], df['ymd']))
df['day'] = list(map(lambda x: x.split('-')[2], df['ymd']))

df['hour'] = list(map(lambda x: x.split(':')[0], df['hms']))


# print(len(df['Ad Topic Line'].unique()))  # 1000
# print(len(df['City'].unique()))  # 969
# print(len(df['Country'].unique()))  # 237


# df.to_csv("advertising_pre.csv", index=False)