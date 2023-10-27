import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
veriler=pd.read_excel("iris.xlsx")
print(veriler)

x=veriler.iloc[:,1:4].values
y=veriler.iloc[:,4:].values
print(y)


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.33,random_state=0)


sc=StandardScaler()


X_train=sc.fit_transform(x_train)
X_test=sc.transform(x_test)

#logictic regressin
from sklearn.linear_model import LogisticRegression
log=LogisticRegression(random_state=0)
log.fit(X_train,y_train)#egitim
y_pred=log.predict(X_test)#tahmin

print("linear")


#karmaşıklık matrisi
cm=confusion_matrix(y_test,y_pred)
print(cm)
#knn 
from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier(n_neighbors=1,metric="minkowski")
knn.fit(X_train,y_train)
cm=confusion_matrix(y_test,y_pred)
print(cm)

# e svc svm classs
from sklearn.svm import SVC
svc=SVC(kernel="poly")
svc.fit(X_train,y_train)
y_pred=svc.predict(X_test)

cm=confusion_matrix(y_test,y_pred)
print("SVC")
print(cm)
# 4 naivbeayes
from sklearn.naive_bayes import GaussianNB
gnb=GaussianNB()
gnb.fit(X_train,y_train)
y_pred=gnb.predict(X_test)

cm=confusion_matrix(y_test,y_pred)
print("QNB")
print(cm)

# decision tree
from sklearn.tree import DecisionTreeClassifier
dtc=DecisionTreeClassifier(criterion="entropy")


dtc.fit(X_train,y_train)
y_pred=dtc.predict(X_test)

cm=confusion_matrix(y_test,y_pred)
print("DTC")
print(cm)

# random forest
from sklearn.ensemble import RandomForestClassifier
rfc=RandomForestClassifier(n_estimators=10,criterion="entropy")

rfc.fit(X_train,y_train)
y_pred=rfc.predict(X_test)

cm=confusion_matrix(y_test,y_pred)
print("RFC")
print(cm)




#roc,tpr,fpr degerleri
y_proba=rfc.predict_proba(X_test)
print(y_proba[:,0])
print(y_test)
from sklearn import metrics
fpr,tpr,thold=metrics.roc_curve(y_test,y_proba[:,0],pos_label="e") 
print(fpr)
print(tpr)