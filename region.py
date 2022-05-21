import pandas as pd
import world_bank_data as wb

pd.set_option('display.max_columns', 15)
# pd.set_option('display.max_rows', 300)
countries = wb.get_countries()


# print(countries[['name', 'region']])
# print(countries['name'])

ref = list(countries['name'])

df = pd.read_csv("C:/Users/YSS_Home/Desktop/과제/advertising.csv")

not_in = []
for i in df['Country']:
    if i not in ref:
        not_in.append(i)

# print(not_in)
# print(len(not_in))

# countries[['name', 'region', 'incomeLevel']].to_csv('ref.csv')
# pre = pd.DataFrame(not_in)

target = df[df['Country'] == 'Congo']

print(target)