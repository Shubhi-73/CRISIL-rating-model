import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import sklearn as sk
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.datasets import make_classification
from sklearn.feature_selection import RFECV
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split


import pandas as pd

# Takes the file's folder
filepath = r"Downloads\corporate_rating.csv"

# read the CSV file
df = pd.read_csv(filepath)

# selecting all coulmns from currentRatio Onwards
x = df.iloc[:2025, 6:31]
y = df.iloc[:2025, 0]
x_test = df.iloc[2030:, 6:31]
# print the first five rows
# x.head()

print(y.head())

print(x.head())


# selector = SelectKBest(chi2, k=10)
# x = selector.fit_transform(x, y)
# x_test = selector.transform(x_test)


# scale with feature names

x, y, x_test, y_test = train_test_split(x, y, test_size=0.25, random_state=0)


scaler = StandardScaler()
scaler.fit(x)
x = pd.DataFrame(scaler.transform(x), columns=x.columns)
x_test = pd.DataFrame(scaler.transform(x_test), columns=x_test.columns)
print("These are x values")

print(x)

# pca = PCA(n_components=12)
# x = pca.fit_transform(x)
# x_test = pca.transform(x_test)


estimator = DecisionTreeClassifier()

# Create an RFECV object
selector = RFECV(estimator, cv=2)

# Fit the RFECV object to the data
selector.fit(x, y)

# Print the selected features
selected_features = selector.transform(x)

clf = LogisticRegression(multi_class="multinomial", solver="lbfgs", max_iter=1000)
clf.fit(selected_features, y)


# scores = cross_val_score(clf, x, y, cv=3)
# print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

selected_features_test = selector.transform(x_test)

y_pred = clf.predict(selected_features_test)
print(y_pred)
