import numpy as np
import matplotlib.pyplot as plt
from ML.PCA.PCA import PCA_class
from copy import deepcopy

PCA = PCA_class(200)
X = np.load('data_ex01/digits.npy')


X_original = deepcopy(X)

Y = X.reshape(X.shape[0], 784)
print(Y.shape)
X = Y.T
M = np.mean(X, axis = 1, keepdims= True)

P,R = PCA.do_operation(X)
print(R.shape)

Y=np.dot(P.T,R)
Y=(Y+M).T
print(Y.shape)

plt.figure(figsize=(8,8))

for i in range(10):
    plt.subplot(2,10,i+1)
    plt.imshow(X_original[i,:,:])
    plt.subplot(2, 10, i+11)
    plt.imshow(Y[i,:].reshape(28,28))

plt.tight_layout()
plt.subplots_adjust(hspace=0,wspace =0)
plt.show()
