# -*- coding: utf-8 -*-
"""
Created on Tue Oct 24 15:50:52 2023

@author: Acer
"""


import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 

veriler=pd.read_csv("Wine.csv")
print(veriler)




X=veriler.iloc[:,3:13].values
Y=veriler.iloc[:,13].values



#egitim test bolunmesş
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=33,random_state=0)

from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_train=sc.fit_transform(x_train)
X_test=sc.fit_transform(x_test)
#pca
from sklearn.decomposition import PCA

pca=PCA(n_components=2)

X_train2=pca.fit_transform(X_train)#hem eğitim hem çalıştırma
X_test2=pca.transform(X_test)
#pca dönüşümden  önce gelen lr
from sklearn.linear_model import LogisticRegression
classifer=LogisticRegression(random_state=0)
classifer.fit(X_train2,y_train)
#pca dönüşümden  sonra gelen lr
classifer2=LogisticRegression(random_state=0)
classifer2.fit(X_train2,y_train)
#tahminler
y_pred=classifer.predict(X_test)
y_pred2=classifer2.predict(X_test2)



from sklearn.metrics import confusion_matrix
#actual / PCA olmadan çıkan sonuç
print('gercek / PCAsiz')
cm = confusion_matrix(y_test,y_pred)
print(cm)

#actual / PCA sonrası çıkan sonuç
print("gercek / pca ile")
cm2 = confusion_matrix(y_test,y_pred2)
print(cm2)

#PCA sonrası / PCA öncesi
print('pcasiz ve pcali')
cm3 = confusion_matrix(y_pred,y_pred2)
print(cm3)



from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA


lda=LDA(n_components=2)

X_train_lda=lda.fit_transform(X_train,y_train)
X_test_lda=lda.transform(X_test)


#lda dönüşümden sonra
classifer_lda=LogisticRegression(random_state=0)
classifer_lda.fit(X_train_lda,y_train)

#tahminler
y_pred_lda=classifer_lda.predict(X_test_lda)
#ldA sonrası / PCA öncesi
print('lda  ve orjinal')
cm4 = confusion_matrix(y_pred,y_pred_lda)
print(cm4)



