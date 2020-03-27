import numpy as np
import matplotlib.pyplot as plt

import seaborn as sns
plt.rcParams['figure.figsize'] = (20,20)
N = 200
data = np.load('kmeans2d.npy')
print(data.shape)
plt.scatter(data[0,:],data[1,:])
#plt.show()

K = 3
colors = ['darkblue','darkgreen','orange','darked','deeppink']
C_column = np.random.randint(600, size = K)
#print(C_column)
#print(C_column.shape)
C = data[:,C_column]
#print(C)
T = 10
loss = np.zeros(T)
#print(loss)
#print(C[:,:,None].shape)
#print(data[:,None].shape)
#print(C[:,:,None]- data)

for i in range (T):
    dist = np.sum((C[:,:,None]-data[:,None,:])**2, axis = 0)
    M = dist.argmin(axis = 0)
    loss[i] = np.sum(dist.min(axis=0))
    # Get the matrix to show the dist pair of each point and the centers.
    # print(dist.shape) #(3,600). Sum add according the axis_0, make it vanish.
    #An index array: each point gets the closest center index.
    #print(M.shape)#(600,)
    #print(M)
    #print(M==1)

    for k in range(3):
        plt.scatter(data[0, M==k], data[1, M==k], color = colors[k])
        plt.scatter(C[0,k], C[1,k], s = 200, color = 'red', marker = 'X')
        plt.title(f'Iteration{i}',fontsize = 20)
    plt.show()
    #Update the centers
    for k in range(K):
        C[:,k] = np.mean(data[:, M==k],axis=1)


plt.plot(loss)
plt.xlabel('literation',fontsize = 20)
plt.ylabel('loss', fontsize = 20)
plt.show()