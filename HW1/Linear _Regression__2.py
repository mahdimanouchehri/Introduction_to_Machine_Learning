
import matplotlib.pyplot as plt
import numpy as np
import random


def cost_function(X, y, theta):

    ## number of training examples
    m = len(y)

    ## Calculate the cost with the given parameters
    J = np.sum((X.dot(theta) - y) ** 2) / 2 / m

    return J






X= np.random.choice(100, (100,1) , replace=False)
e_avr_train=0
m=20
theta_avr=0
e_avr=0
landa = 0.001
for i in range (0 , 5) :
    print('for',i+1 ,':')
    
    X_test=X[m*i:m*i+20 , 0 ].reshape(20,1)
    y_test = 3 + 0.4 * X_test + np.random.randint(-10, 10, 20).reshape(20, 1)
    X_train1=X[0:m*i ,0]
    X_train2=X[m*(i+1):100,0]
    X_train = np.concatenate((X_train1, X_train2), axis=0).reshape(80,1)
    y_train= 3 + 0.4 * X_train + np.random.randint(-10, 10, 80).reshape(80, 1)


    a = np.array([[0, 0], [0, 1]])
    X_b = np.c_[np.ones((80, 1)), X_train]
    theta_best = np.linalg.inv(X_b.T.dot(X_b) + landa * a).dot(X_b.T).dot(y_train)
    theta_avr = theta_avr + theta_best
    print('theta best:',theta_best)
    theta_best = theta_best.reshape(1, 2)
    e_test = cost_function(X_test, y_test, theta_best)
    e_train = cost_function(X_train, y_train, theta_best)
    print('E test:',e_test)
    print('E train:', e_train)
    e_avr= e_avr+e_test
    e_avr_train=e_avr_train+e_train
theta_avr=theta_avr/5
e_avr_train=e_avr_train/5
print('theta_avr:',theta_avr)

e_avr=e_avr/5
print('e_avr:',e_avr)
print('e_avr_train',e_avr_train)


