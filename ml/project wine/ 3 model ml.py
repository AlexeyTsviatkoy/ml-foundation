from sklearn.datasets import load_wine
import matplotlib.pyplot as plt
import pandas as pd
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

X_new, X_test, y_new, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

df = pd.DataFrame(X, columns=wine.feature_names)
df['target'] = wine.target_names[y]

