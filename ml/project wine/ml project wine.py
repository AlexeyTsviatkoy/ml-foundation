from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_wine
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

wine = load_wine()

X = wine.data
y = wine.target

X_new = [[13.2, 1.8, 2.4, 15.6, 100, 2.5, 2.7, 0.3, 1.8, 5.2, 1.04, 3.1, 1050]]
feature_names = wine.feature_names

df = pd.DataFrame(X, columns=wine.feature_names)
df['target'] = wine.target_names[y]

#print(df.head())
#print(df.describe())
#print(df["target"].value_counts())
#print(df.corr(numeric_only=True))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = DecisionTreeClassifier()
model.fit(X_train, y_train)
pred = model.predict(X_test)

accuracy = accuracy_score(y_test, pred)
print(accuracy)
print(confusion_matrix(y_test, pred))
predNew = model.predict(X_new)
print(wine.target_names[predNew[0]])

plt.figure(figsize=[8,6])
plt.bar(feature_names, model.feature_importances_)
plt.xticks(rotation = 90)
plt.tight_layout()

plt.show()



