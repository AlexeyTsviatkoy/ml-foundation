import pandas as pd

dataframe = {
    "Name" : ["Alex","John","Anna","Mike","Sara"],
    "Age" : [25, 30, 22, 35, 28],
    "Salary" : [3000, 4000, 3500, 5000, 4200]
}

df = pd.DataFrame(dataframe)

# print(df['Age'].mean(), df['Salary'].mean())

# print(df["Name"][df['Salary'].idxmax() ], df["Name"][df['Salary'].idxmin()])

df['Salary_after_tax'] = df['Salary'] * 0.87

# print(df_sorted := df.sort_values(by = "Salary", ascending = False))

print("\nИмена сотрудников зарплата которых выше 4000:\n",df.loc[:,["Name","Salary"]][df["Salary"] > 4000],
      "\nИмена сотрудников возраст которых ниже 30:\n", df.loc[:,["Name","Salary"]][df["Age"] < 30],
      "\nИмена сотрудников возраст которых ниже 30:\n", df.loc[:,["Name","Salary"]][df["Salary_after_tax"] > 3500])
