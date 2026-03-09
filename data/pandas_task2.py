import pandas as pd

df = pd.read_csv('/home/alexey/Документы/Projects/ml-foundation/datasets/people.csv')

print(df["age"].mean(), df["salary"].mean(),df["city"][df["city"].idxmax()])