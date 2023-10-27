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



from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_olcek1=sc.fit_transform(X)
sc2=StandardScaler()
Y_olcek2=sc2.fit_transform(Y)



from sklearn.svm import SVR
sv_reg=SVR(kernel="rbf")
sv_reg.fit(X_olcek1,Y_olcek2)

plt.scatter(X_olcek1,Y_olcek2,color="red")
plt.plot(X_olcek1,sv_reg.predict(X_olcek1),color="blue")
plt.show()
print(sv_reg.predict([[11]]))
print(sv_reg.predict([[6.6]]))


from sklearn.tree import DecisionTreeRegressor
r_dt=DecisionTreeRegressor(random_state=0)
r_dt.fit(X,Y)
Z=X*0.5
K=X-0.5
plt.scatter(X,Y,color="black")
plt.plot(X,r_dt.predict(X),color="pink")
plt.plot(X,r_dt.predict(Z),color="yellow")
plt.plot(X,r_dt.predict(K),color="orange")
plt.show()
print(r_dt.predict([[11]]))
print(r_dt.predict([[6.6]]))


from sklearn.ensemble import RandomForestRegressor
rf=RandomForestRegressor(n_estimators=10,random_state=0)
rf.fit(X,Y.ravel())
print(rf.predict([[6.6]]))

plt.scatter(X,Y,color="red")
plt.plot(X,rf.predict(X),color="blue")

plt.plot(X,rf.predict(Z),color="green")

plt.plot(X,rf.predict(K),color="orange")
plt.show()
