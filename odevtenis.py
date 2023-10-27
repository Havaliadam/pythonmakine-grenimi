import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 

veri=pd.read_csv("odev_tenis.csv")
print(veri)

from sklearn.preprocessing import LabelEncoder
veri2=veri.apply(LabelEncoder().fit_transform)


c=veri2.iloc[:,:1]
from sklearn.preprocessing import OneHotEncoder
one=OneHotEncoder(categrorical_features="all")
c=one.fit_transform(c).toarray()
print(c)

havadurumu=pd.DataFrame(data=c,index=range(14),columns=["o","r","s"])
sonveri=pd.concat([havadurumu,veri.iloc[:,1:3]],axis=1)
sonveri=pd.concat([havadurumu,veri2.iloc[:,-2:],sonveri],axis=1)
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(sonveri.iloc[:,:-1],sonveri.iloc[:,-1:],test_size=0.33,random_state=0)

from sklearn.linear_model import LinearRegression
reg=LinearRegression()
reg.fit(x_train,y_train)

y_pred=reg.predict(x_test)

print(y_pred)

import statsmodels.api as sm

X=np.append(arr=np.ones((14,1)).astype(int),values=sonveri.iloc[:,:-1],axis=1)
X_1=sonveri.iloc[:,[0,1,2,3,4,5]].values
r_ols=sm.OLS(endog=sonveri.iloc[:,:-1],exog=X_1)
r=r_ols.fit()
print(r.summary())



sonveri=sonveri.iloc[:,1:]

import statsmodels.formula.api as sm 
X=np.append(arr=np.ones((14,1)).astype(int),values=sonveri.iloc[:,:-1],axis=1)
X_1=sonveri.iloc[:,[0,1,2,3,4,5]].values
r_ols=sm.OLS(endog=sonveri.iloc[:,:-1],exog=X_1)
r=r_ols.fit()
print(r.summary())
x_train=x_train.iloc[:,1:]
x_test=x_test.iloc[:,1:]
reg.fit(x_train,y_train)
y_pred=reg.predict(x_test)

