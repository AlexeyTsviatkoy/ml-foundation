import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pandas as pd

df = pd.read_csv('/home/alexey/Документы/Projects/ml-foundation/datasets/people.csv')

X = df[["age"]]
y = df["salary"]
X_new = pd.DataFrame([[26],[33],[45]], columns=["age"])

X_train,X_test,y_train,y_test = train_test_split(X,y ,test_size=0.2,random_state=42)

reg = LinearRegression()
reg.fit(X_train,y_train)

pred = reg.predict(X_test)

preg_new = reg.predict(X_new)

print(preg_new)

plt.scatter(X,y,color="red")
plt.plot(X,reg.predict(X),color="green")
plt.xlabel("Age")
plt.ylabel("Salary")
plt.show()