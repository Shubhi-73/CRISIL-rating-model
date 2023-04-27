import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
import pandas as pd
import sklearn as sk
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import RFECV
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score, recall_score, f1_score

filepath = r"Downloads\corporate_rating.csv"

# read the CSV file
df = pd.read_csv(filepath)


# dropping less valis categories
df.drop(df.index[(df["Rating"] == "AAA")], axis=0, inplace=True)
df.drop(df.index[(df["Rating"] == "CC")], axis=0, inplace=True)
df.drop(df.index[(df["Rating"] == "C")], axis=0, inplace=True)
df.drop(df.index[(df["Rating"] == "CCC")], axis=0, inplace=True)
df.drop(df.index[(df["Rating"] == "D")], axis=0, inplace=True)


# selecting all coulmns from currentRatio Onwards
X = df.iloc[:, 6:31].values
y = df.iloc[:, 0].values

scaler = StandardScaler()
X = scaler.fit_transform(X)


model = LogisticRegression(
    penalty="l2",
    solver="newton-cg",
    multi_class="multinomial",
    random_state=0,
    max_iter=1000,
    C=0.1,
)


# Recursive Feature Selection with cross-validation
rfecv = RFECV(estimator=model, step=1, cv=StratifiedKFold(5), scoring="accuracy")
rfecv.fit(X, y)

# shows the features selected with - true
print("support", rfecv.support_)

X = X[:, rfecv.support_]


# Print the optimal number of features
print("Optimal number of features : %d" % rfecv.n_features_)


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=0
)


# training the model
model.fit(X_train, y_train)

y_pred = model.predict(X_test)


print(y_pred)

precision = precision_score(y_test, y_pred, average="macro")
recall = recall_score(y_test, y_pred, average="macro")
f1 = f1_score(y_test, y_pred, average="macro")

# print the precision, recall, and F1-score
print("Precision:", precision * 100)
print("Recall:", recall * 100)
print("F1-score:", f1 * 100)

print("Accuracy: ", accuracy_score(y_test, y_pred) * 100)
