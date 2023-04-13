import matplotlib.pyplot as plt
import numpy as np

import sklearn as sk
from sklearn.linear_model import LogisticRegression


import pandas as pd

# Takes the file's folder
filepath = r"Downloads\corporate_rating.csv"

# read the CSV file
df = pd.read_csv(filepath)

# selecting all coulmns from currentRatio Onwards
x = df.iloc[:, 6:31]
y = df["Rating"]

# print the first five rows
print(x.head())
print(y.head())

# model = LogisticRegression(solver='lbfgs', multi_class='multinomial',random_state=0)
# model.fit(x, y)
