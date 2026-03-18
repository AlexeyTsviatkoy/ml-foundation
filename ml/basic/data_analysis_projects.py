import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

df = pd.read_csv("../../datasets/housing.csv")

print(df.head())
print(df.describe())

print(df["price"].mean(), df["size"][df["size"].idxmax()], df["age"][df["age"].idxmin()])

print(df.corr()) #Размер больше всего влияет на цену

X = df[["size","rooms","age"]]
y = df["price"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_new = pd.DataFrame([[95,3,5]],columns=["size","rooms","age"])

reg = LinearRegression()
reg.fit(X_train, y_train)
pred = reg.predict(X_new)

y_pred = reg.predict(X_test)
print(r2_score(y_test, y_pred))

print(reg.coef_)
print(reg.intercept_)

plt.scatter(df["size"], df["price"], color="red")
plt.scatter(df["rooms"], df["price"] ,color="blue")
plt.scatter(df["age"], df["price"] ,color="green")
plt.scatter(X_new["size"], pred ,color="yellow", s = 100)
plt.scatter(X_new["rooms"], pred ,color="pink", s = 100)
plt.scatter(X_new["age"], pred ,color="black",s = 100)
plt.xlabel("Size(red),Rooms(blue),Age(green)")
plt.ylabel("Price")
plt.show()
