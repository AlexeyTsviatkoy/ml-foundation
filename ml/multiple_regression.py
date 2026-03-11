from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import r2_score


df = pd.read_csv("../datasets/employees.csv")

X = df[["age","experience"]]
y = df["salary"]

X_new = pd.DataFrame([[27,3],[41,9]], columns=["age","experience"])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

reg = LinearRegression()
reg.fit(X_train, y_train)

pred = reg.predict(X_test)
predNew = reg.predict(X_new)

print(reg.coef_, reg.intercept_)
print(predNew)

y_pred = reg.predict(X_test)
print("R2:", r2_score(y_test, y_pred))

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.scatter3D(df["age"],df["experience"], df["salary"], color="blue", s = 50)
ax.scatter3D(X_new["age"],X_new["experience"], predNew, color="red", s = 100)

ax.set_xlabel('age')
ax.set_ylabel('experience')
ax.set_zlabel('salary')

plt.show()
