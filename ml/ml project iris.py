from sklearn.datasets import load_iris
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

iris = load_iris()

X = iris.data
y = iris.target

df = pd.DataFrame(X, columns=iris.feature_names)
df['species'] = iris.target_names[y]

# print(df.head(), df.describe(),df['species'].value_counts())

plt.scatter(df['petal length (cm)'], df['petal width (cm)'])
plt.xlabel('petal length (cm)')
plt.ylabel('petal width (cm)')

X_train1, X_test1, y_train1, y_test1 = train_test_split(X, y, test_size=0.2, random_state=42)
reg = LogisticRegression()
reg.fit(X_train1, y_train1)
pred = reg.predict(X_test1)

acc = accuracy_score(y_test1, pred)
print(acc)

X_new = pd.DataFrame([[5.1,3.5,1.4,0.2]],columns = ['sepal length (cm)','sepal width (cm)','petal length (cm)', 'petal width (cm)'])

predNew = reg.predict(X_new.values)

new_point = X_new.iloc[0]
plt.scatter(new_point['petal length (cm)'], new_point['petal width (cm)'], color='red', s=100, label='Новый цветок')
plt.show()

print(iris.target_names[predNew[0]])



