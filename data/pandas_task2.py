import pandas as pd

df = pd.read_csv('/home/alexey/Документы/Projects/ml-foundation/datasets/people.csv')

m_city = df["city"].value_counts().index[0]
mean_salary = df["salary"].mean()

print(df["age"].mean(),
      mean_salary,
      m_city,
      df.groupby("city")["salary"].mean(),"\n",
      df.loc[df["salary"] >= mean_salary,"name"])
