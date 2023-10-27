import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 


veriler=pd.read_csv("veriler.csv")
print(veriler)


x=veriler.iloc[:,2:4].values#bağimsiz degisken
y=veriler.iloc[:,4:].values#bagımli degisken
print(y)


#verilerin egitim ve test bolunmesi
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.33,random_state=0)

#verilerin olceklenmesi
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()

X_train=sc.fit_transform(x_train)
X_test=sc.transform(x_test)



from sklearn.linear_model import LogisticRegression
lo_rg=LogisticRegression(random_state=0)
lo_rg.fit(X_train,y_train)

y_pred=lo_rg.predict(X_test)
print(y_pred)
print(y_test)



from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)
print(cm)

from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier(n_neighbors=1,metric="minkowski")
knn.fit(X_train,y_train)
y_pred=knn.predict(X_test)
cm=confusion_matrix(y_test,y_pred)
print(cm)

from sklearn.svm import SVC
svc=SVC(kernel="poly")
svc.fit(X_train,y_train)
y_pred=svc.predict(X_test)
cm=confusion_matrix(y_test,y_pred)
print("SVC")
print(cm)



