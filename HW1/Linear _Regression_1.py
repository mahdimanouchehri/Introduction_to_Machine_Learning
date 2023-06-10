import matplotlib.pyplot as plt
import numpy as np
import random

X= np.random.choice(100, (90,1) , replace=False)
X_test=[]
for i in range(0,100):

    if i in X:
        continue
    else:
        X_test.append(i)

X_test = np.array(X_test).reshape(10,1)
y_test=3 + 0.4*X_test + np.random.randint(-10, 10, 10  ).reshape(10,1)
y=3 + 0.4*X + np.random.randint(-10, 10, 90  ).reshape(90,1)
plt.plot(X,y,'b.')
plt.xlabel('$x_1$',fontsize= 18)
plt.ylabel('$y$', rotation=0 , fontsize=18)
plt.axis([-10,105,-20,50])
plt.show()



X_b = np.c_[np.ones((90,1)),X]

theta_best= np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)
print(theta_best)




X_new=np.array([[0],[100]])
X_new_b=np.c_[np.ones((2,1)),X_new]
y_predict=X_new_b.dot(theta_best)


plt.plot(X_new,y_predict,"r-",linewidth=3,label='predictions')
plt.plot(X,y,'b.')
plt.xlabel('$x_1$',fontsize=18)
plt.ylabel('$y$',rotation=0,fontsize=18)
plt.legend(loc='upper left',fontsize=14)
plt.axis([0,105,-20,50])
plt.show()





eta=0.1
n_iteration=100
m=100

theta=np.random.randn(2,1)

for iteration in range(n_iteration):
    gradiant=2/m* X_b.T.dot(X_b.dot(theta)-y)
    theta=theta-eta*gradiant
print('theta',theta)



def cost_function(X, y, theta):

    m = len(y)

    J = np.sum((X.dot(theta) - y) ** 2) / 2 / m

    return J


theta_best = theta_best.reshape(1, 2)
e_test = cost_function(X_test, y_test, theta_best)
print(e_test)

