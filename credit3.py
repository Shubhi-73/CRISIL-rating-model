import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('corporate_rating.csv')
X_df = dataset.iloc[:, 5:31]
y_df = dataset.iloc[:, 0]

#dropping less valis categories
dataset.drop(dataset.index[(dataset["Rating"] == "AAA")],axis=0,inplace=True)
dataset.drop(dataset.index[(dataset["Rating"] == "CC")],axis=0,inplace=True)
dataset.drop(dataset.index[(dataset["Rating"] == "C")],axis=0,inplace=True)
#dataset.drop(dataset.index[(dataset["Rating"] == "CCC")],axis=0,inplace=True)
dataset.drop(dataset.index[(dataset["Rating"] == "D")],axis=0,inplace=True)

X_df = dataset.iloc[:, 5:31]
y_df = dataset.iloc[:, 0]

#encoding y and Sector
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y_df)
X_df.Sector = le.fit_transform(X_df.Sector)

X=X_df.to_numpy()  

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

#Applying PCA

from sklearn.decomposition import PCA
pca = PCA(n_components = 23)
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))


#Calculating accuracy
from sklearn.metrics import accuracy_score
print(accuracy_score(y_test, y_pred)*100)
