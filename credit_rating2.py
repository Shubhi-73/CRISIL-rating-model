#kernel SVM

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
dataset = pd.read_csv('corporate_rating.csv')
X = dataset.iloc[:, 6:31].values
y = dataset.iloc[:, 0].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)                                    

#applying PCA
from sklearn.decomposition import PCA
pca = PCA(n_components = 22)
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

from sklearn.svm import SVC
classifier = SVC(kernel = 'rbf', random_state = 0)
classifier.fit(X_train, y_train)

print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))

from sklearn.metrics import accuracy_score
print(accuracy_score(y_test, y_pred)*100)
