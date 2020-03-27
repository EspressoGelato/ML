import numpy as np
import matplotlib.pyplot as plt
from random import sample
X = np.load('knn2d.npy')
Y = np.load('knn2dlabels.npy')
print(X.shape)#(2,400)
print(Y.shape)

def KNN(TestData, K, X, Y):
    #The shape of TestData is (2,N)
    dist = np.sum((X[:,:,None] - TestData[:,None,:])**2, axis = 0)#shape(400,N)
    index = np.argsort(dist, axis = 0)[0:K, :]#shape(K,N)
    label = np.array([np.mean(Y[index[:, i]]) for i in range(index.shape[1])])
    return (label >= (np.zeros(index.shape[1])+0.5))+0

plt.scatter(X[0, Y==0], X[1,Y==0], color = 'darkgreen', alpha=0.4)
plt.scatter(X[0, Y==1], X[1,Y==1], color = 'blue',alpha = 0.4)

x = np.arange(-4,4,0.1)
y = np.arange(-4,4,0.1)
x, y  =np.meshgrid(x, y)
xx = x.reshape(1,-1)
yy = y.reshape(1,-1)

print(xx.shape, yy.shape)

Test = np.vstack([xx,yy])
print(Test.shape)#(2,16)
label = KNN(Test, 5, X, Y)
print(label)

plt.scatter(Test[0,label ==1],Test[1, label ==1],color = 'red',alpha = 0.2)
plt.scatter(Test[0,label ==0],Test[1, label ==0],color = 'darkgreen',alpha = 0.2)
plt.show()



