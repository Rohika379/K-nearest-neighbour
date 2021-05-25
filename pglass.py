# -*- coding: utf-8 -*-
"""
Created on Sun Mar 14 19:35:53 2021

@author: rohika
"""

#import dependencies
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import seaborn as sns
from sklearn.metrics import classification_report, accuracy_score

#Importing the dataset
glass =pd.read_csv("C:\\Users\\rohika\\OneDrive\\Desktop\\360digiTMG assignment\\KNN\\glass.csv")

glass.head()

sns.scatterplot(glass['RI'],glass['Na'],hue=glass['Type'])
scaler = StandardScaler()
scaler.fit(glass.drop('Type',axis=1))
StandardScaler(copy=True, with_mean=True, with_std=True)

#perform transformation
scaled_features = scaler.transform(glass.drop('Type',axis=1))
scaled_features

df_feat = pd.DataFrame(scaled_features,columns=glass.columns[:-1])
df_feat.head()

dff = df_feat.drop(['Ca','K'],axis=1) #Removing features - Ca and K 
X_train,X_test,y_train,y_test  = train_test_split(dff,glass['Type'],test_size=0.3,random_state=45)
knn = KNeighborsClassifier(n_neighbors=4,metric='manhattan')

knn.fit(X_train,y_train)

KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='manhattan',
                     metric_params=None, n_jobs=None, n_neighbors=4, p=2,
                     weights='uniform')

y_pred = knn.predict(X_test)

print(classification_report(y_test,y_pred))


accuracy_score(y_test,y_pred)
from sklearn.model_selection import cross_val_score

k_range = range(1,25)
k_scores = []
error_rate =[]
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    #kscores - accuracy
    scores = cross_val_score(knn,dff,glass['Type'],cv=5,scoring='accuracy')
    k_scores.append(scores.mean())
    
    #error rate
    knn.fit(X_train,y_train)
    y_pred = knn.predict(X_test)
    error_rate.append(np.mean(y_pred!=y_test))

#plot k vs accuracy
plt.plot(k_range,k_scores)
plt.xlabel('value of k - knn algorithm')
plt.ylabel('Cross validated accuracy score')
plt.show()

#plot k vs error rate
plt.plot(k_range,error_rate)
plt.xlabel('value of k - knn algorithm')
plt.ylabel('Error rate')
plt.show()


















