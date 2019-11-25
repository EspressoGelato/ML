import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys
sys.path.append('C:/Users/Yuchen/PycharmProjects/ML')
import Linear_regression.EX_1.Pre as pre
from Regularization_and_GP.EX_1.Ridge_regression import L2_r


tmp=np.load('./../data_ex05/sheet5-linreg1.npz')
X=tmp['X']
Y=tmp['Y']

'''
print('X=',X)
print('Y=',Y)

print(np.shape(X))
print(np.shape(Y))
'''

p,n=X.shape

cal_beta=pre.fit_beta(X,Y)
print(cal_beta)

num_1=cal_beta[1]
num_2=cal_beta[2]



def calc_ssq(beta_1,beta_2,X,Y):
    beta = np.vstack([beta_1,beta_2])
    yhat=beta.T.dot(X)
    residuals=Y-yhat
    ssq=residuals**2
    SSQ=np.sum(ssq,axis=1)
    return SSQ


def SSQ(X,Y):
    x = np.arange(-1, 3, 4 / n)
    y = np.arange(-1, 3, 4 / n)
    beta_1, beta_2 = np.meshgrid(x, y)
    ssq=calc_ssq(x,y,X,Y)
    plt.contourf(beta_1, beta_2, ssq)

'''
x = np.arange(-1, 3, 4 / n)
y = np.arange(-1, 3, 4 / n)
ssq=calc_ssq(x,y,X,Y)
print(ssq)
print(np.shape(ssq))
'''

SSQ(X,Y)
L2_r()
plt.scatter(num_1,num_2)
plt.show()

