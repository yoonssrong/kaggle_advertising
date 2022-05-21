from sklearn.preprocessing import StandardScaler
import pandas as pd

pd.set_option('display.max_columns', 15)

data = pd.read_csv('./data/advertising_pre3.csv')

data_for_st = data[['Daily Time Spent on Site', 'Age', 'Area Income', 'Daily Internet Usage']]
data_rem = data[['Male', 'Country', 'region', 'region_incomeLevel', '']]

scaler = StandardScaler()
scaler = scaler.fit_transform(data_for_st)


print(data)