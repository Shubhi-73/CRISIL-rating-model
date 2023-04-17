import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import sklearn as sk
from sklearn.linear_model import LogisticRegression


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

# scale with feature names


scaler = StandardScaler()
scaler.fit(x)
x = pd.DataFrame(scaler.transform(x), columns=x.columns)
x_test = pd.DataFrame(scaler.transform(x_test), columns=x_test.columns)
print("These are x values")

print(x)

pca = PCA(n_components=12)
x = pca.fit_transform(x)
x_test = pca.transform(x_test)

explained_variance = pca.explained_variance_ratio_
print(explained_variance)


model = LogisticRegression(
    solver="newton-cg", multi_class="multinomial", random_state=0, max_iter=1000
)
model.fit(x, y)

y_pred = model.predict(x_test)
print("THIS IS X TEST")
print(x_test)
print(y_pred)
