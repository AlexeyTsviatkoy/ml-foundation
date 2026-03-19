from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score

_california_housing = fetch_california_housing()

X = _california_housing.data
y = _california_housing.target

df = pd.DataFrame(X, columns=_california_housing.feature_names)
df['target'] = _california_housing.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
scaled_X_train = scaler.fit_transform(X_train)
scaled_X_test = scaler.transform(X_test)

modelLR = LinearRegression()
modelLR.fit(scaled_X_train, y_train)
predLR = modelLR.predict(scaled_X_test)

modelRFC = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
modelRFC.fit(X_train, y_train)
predRFC = modelRFC.predict(X_test)

r2_scoreLR = r2_score(y_test, predLR)
r2_scoreRFC = r2_score(y_test, predRFC)

print(r2_scoreLR)
print(r2_scoreRFC)
scores = cross_val_score(modelRFC, X, y, cv=5, scoring='r2')
print(scores.mean())

plt.figure(figsize=[15,15])
plt.scatter(y_test, predLR, color='red', s= 10)
plt.scatter(y_test, predRFC, color='blue', s= 10)
plt.plot([y_test.min(), y_test.max()],
         [y_test.min(), y_test.max()], color='black')
plt.xlabel('Фактические значения', fontsize=12)
plt.ylabel('Предсказанные значения', fontsize=12)
plt.show()

predRFRF = modelRFC.feature_importances_
plt.bar(_california_housing.feature_names, predRFRF)
plt.xticks(range(len(_california_housing.feature_names)), _california_housing.feature_names, rotation=45, ha='right')
plt.tight_layout()
plt.show()



