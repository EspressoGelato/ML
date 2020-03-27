#Compute unsupervised clustering of data into fixed number of clusters.

import numpy as np
import matplotlib.pyplot as plt
from random import sample
from numpy import linalg as LA
#cluster a data set X into K clusters

X = np.load('kmeans2d.npy')
N = X.shape[1]
plt.scatter(X[0,:],X[1,:])
print(X, N)

class KMeans_class:
	def __init__(self):
		pass

	def initialize(self, K):
		a = sample(range(N), K)
		print('#######', a)
		M = np.zeros((K, N))
		C_0 = X[:,a]
		#print(C_0)
		cmap = plt.cm.jet
		# extract all colors from the .jet map
		self.cmaplist = [cmap(i) for i in range(K)]
		print(len(self.cmaplist))
		return C_0, M

	def update(self, K, C_0, M):
		C = C_0
		for n in range(N):
			x_n = X[:,n]
			x_n = x_n.reshape(2,1)
			D = np.sum((x_n - C)**2, axis=0)
			k = np.argmin(D)
			M[k,n] = 1
		#print('#######',M)

		# create the new map
		for k in range(K):
			Index = M[k, :] == 1
			#print('#', Index)
			#print(np.sum(Index))
			if (Index).all():
				break
			else:
				X_k = X[:,Index]
				x, y = X_k[0,:], X_k[1,:]
				plt.scatter(x, y,cmap=self.cmaplist[k])
				#print(np.sum(Index))
				c_k = np.mean(X[:, Index], axis=1)
				#c_k = np.mean(X[:, [M[k,n] == 1 for n in range(N)]], axis = 1)
				C[:,k] = c_k
				x, y = C[0,:], C[1,:]

			#plt.scatter(x,y,cmap=self.cmaplist[k])
		return C,M


KM = KMeans_class()

C_0, M_0 = KM.initialize(3)
print(C_0, M_0)
max_step = 50
Trace = [C_0, M_0]
for t in range(max_step):
	C_t, M_t = KM.update(3, Trace[-2], Trace[-1])
	if (C_t != Trace[-2]).any():
		#Trace.append(C_t, M_t)
		Trace += [C_t, M_t]
	else:
		break
plt.show()






