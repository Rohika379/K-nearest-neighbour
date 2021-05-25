# -*- coding: utf-8 -*-
"""
Created on Sun Mar 14 21:29:55 2021

@author: rohika
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

zoo=pd.read_csv("C:\\Users\\rohika\\OneDrive\\Desktop\\360digiTMG assignment\\KNN\\Zoo.csv")

zoo.drop("animal name",axis=1,inplace=True)

def norm_func(i):
    x=(i-i.min()/i.max()-i/min())
    return(x)

zoo_n=norm_func(zoo.iloc[:,0:16])
zoo_n.describe()

x=np.array(zoo.iloc[:,:])
y=np.array(zoo['type'])

from sklearn.model_selection import train_test_split

x_train, x_test ,y_train ,y_test =train_test_split(x,y,test_size =0.2)

from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier(n_neighbors=10)
knn.fit(x_train,y_train)
pred=knn.predict(x_test)

from sklearn.matrics import accuracy_score
print(accuracy_score(y_test,pred))
pred

pd.crosstab(y_test,pred,rownames=['actual'],colnames=['predictions'])
pred_train=knn.predict(x_train)

pd.crosstab(y_train,pred_train,rownames=['actual'],colnames=['predictions'])

acc=[]

for i in range(3,50,2):
    neigh = KNeighborsClassifier(n_neighbors=i)
    neigh.fit(x_train, y_train)
    train_acc = np.mean(neigh.predict(x_train) == y_train)
    test_acc = np.mean(neigh.predict(x_test) == y_test)
    acc.append([train_acc, test_acc])


plt.plot(np.arange(3,50,2),[i[0] for i in acc],"bo-")

plt.plot(np.arange(3,50,2),[i[1] for i in acc],"bo-")
