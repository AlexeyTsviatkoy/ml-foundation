from sklearn.datasets import load_wine
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

wine = load_wine()

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

X = wine.data
y = wine.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

df = pd.DataFrame(X, columns=wine.feature_names)
df['target'] = wine.target_names[y]

scaler = StandardScaler()
scaled_X_train = scaler.fit_transform(X_train)
scaled_X_test = scaler.transform(X_test)

modelLR = LogisticRegression(max_iter=500)
modelLR.fit(scaled_X_train, y_train)
predLR = modelLR.predict(scaled_X_test)
print(accuracy_logreg := accuracy_score(y_test, predLR))

modelDTC = DecisionTreeClassifier()
modelDTC.fit(scaled_X_train, y_train)
predDTC = modelDTC.predict(scaled_X_test)
print(accuracy_tree := accuracy_score(y_test, predDTC))

modelRFC = RandomForestClassifier()
modelRFC.fit(scaled_X_train, y_train)
predRFC = modelRFC.predict(scaled_X_test)
print(accuracy_forest := accuracy_score(y_test, predRFC))

table = pd.DataFrame([[accuracy_logreg,accuracy_tree,accuracy_forest]], columns = ['Logistic Regression','Decision Tree','Random Forest'])
print(table)

plt.figure(figsize = (8,5))
plt.bar(table.columns, table.values[0])
plt.show()

predRFCF = modelRFC.feature_importances_
plt.bar(wine.feature_names, predRFCF)
plt.xticks(range(len(wine.feature_names)), wine.feature_names, rotation=45, ha='right')
plt.tight_layout()
plt.show()





