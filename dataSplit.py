import os
import pandas as pd


df = pd.read_csv('./data/advertising_pre3.csv')

output = pd.DataFrame()

# Partitioning
Train = df.sample(frac=0.8)
Test = df.drop(Train.index)

Train.to_csv("train.csv", index=False, header=True)
Test.to_csv("test.csv", index=False, header=True)

print(len(Train))
print(len(Test))