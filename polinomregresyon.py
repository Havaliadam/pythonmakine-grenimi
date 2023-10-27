import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 


veriler=pd.read_csv("maaslar.csv")

#print(veriler)

x=veriler.iloc[:,1:2]
y=veriler.iloc[:,2:]
X=x.values
Y=y.values
from sklearn.linear_model import LinearRegression
lin_reg=LinearRegression()
lin_reg.fit(x.values,y.values)

plt.scatter(x.values,y.values,color="red")
plt.plot(x,lin_reg.predict(x.values),color="blue")

#polynormal reggression
from sklearn.preprocessing import PolynomialFeatures
pe=PolynomialFeatures(degree=2)

x_poly=pe.fit_transform(X)
print(x_poly)

lin_reg2=LinearRegression()
lin_reg2.fit(x_poly,y)

plt.scatter(x,y,color="red")
plt.plot(x,lin_reg2.predict(pe.fit_transform(X)),color="blue")


#tahminler
print(lin_reg.predict([[11]]))
print(lin_reg.predict([[6.6]]))
print(lin_reg2.predict(pe.fit_transform([[6.6]])))
print(lin_reg2.predict(pe.fit_transform([[11]])))

plt.show()