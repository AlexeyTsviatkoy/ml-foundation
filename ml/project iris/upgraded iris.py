from pyexpat import features
import numpy as np
from sklearn.datasets import load_iris
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

iris = load_iris()

X_new = pd.DataFrame([[5.1,3.5,1.4,0.2]],columns = ['sepal length (cm)','sepal width (cm)','petal length (cm)', 'petal width (cm)'])
X = iris.data
y = iris.target

df = pd.DataFrame(X, columns= iris.feature_names)
df['target'] = iris.target_names[y]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
scaled_X_train = scaler.fit_transform(X_train)
scaled_X_test = scaler.transform(X_test)

logreg = LogisticRegression()
logreg.fit(scaled_X_train,y_train)
pred = logreg.predict(scaled_X_test)
pred_names = iris.target_names[pred]

acc1 = accuracy_score(y_test, pred)

df_coef = pd.DataFrame(logreg.coef_,columns=iris.feature_names, index=iris.target_names)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
print(df_coef)

cm = confusion_matrix(y_test,pred)
print(cm)

df['petal_area'] = df['petal length (cm)'] * df['petal width (cm)']
df['sepal_area'] = df['sepal length (cm)'] * df['sepal width (cm)']

X = df[['petal length (cm)', 'petal width (cm)', 'sepal length (cm)', 'sepal width (cm)','petal_area','sepal_area']]
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler2 = StandardScaler()
X_trained_scaled = scaler2.fit_transform(X_train)
X_test_scaled = scaler2.transform(X_test)
logreg2 = LogisticRegression(max_iter= 500)
logreg2.fit(X_trained_scaled,y_train)
pred = logreg2.predict(X_test_scaled)

acc2 = accuracy_score(y_test, pred)
print(acc1,acc2)

predX_new = logreg.predict(scaler.transform(X_new))
print(iris.target_names[predX_new[0]])

coef_abs = np.abs(logreg.coef_).mean(axis=0)
features = iris.feature_names

plt.figure(figsize = (8,5))

plt.bar(features,coef_abs)

plt.xlabel('Features')
plt.ylabel('Coefficients')

plt.xticks(rotation = 45)

plt.tight_layout()
plt.show()



