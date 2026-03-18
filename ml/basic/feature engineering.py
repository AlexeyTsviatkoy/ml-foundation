import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

df = pd.read_csv("../../datasets/housing.csv")

df["price_per_room"] = df["price"] / df["rooms"]
df["size_per_room"] = df["size"] / df["rooms"]

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
# print(df.corr()) Новые данные влияют на price

print(df.corr()["price"].sort_values())
X = df[["size","rooms","age"]]
y = df["price"]
X_train1, X_test1, y_train1, y_test1 = train_test_split(X, y, test_size=0.2, random_state=42)

reg = LinearRegression()
reg.fit(X_train1, y_train1)
pred = reg.predict(X_test1)
print(r2_score(y_test1, pred))

X_new = df[["size","rooms","age","price_per_room","size_per_room"]]
y_new = df["price"]
X_train, X_test, y_train, y_test = train_test_split(X_new, y_new, test_size=0.2, random_state=42)
reg = LinearRegression()
reg.fit(X_train, y_train)
predNew = reg.predict(X_test)
print(r2_score(y_test, predNew))
