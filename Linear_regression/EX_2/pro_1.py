import numpy as np
import matplotlib.pyplot as plt

tmp=np.load('linear-unc1.npz')
X=tmp['X']
Y=tmp['Y']
print(X.shape,Y.shape)
plt.scatter(X[0],X[1],alpha=0.1)


def get_sample(X):
    column_num=X.shape[1]
    num=10
    generate=range(1,column_num)
    list=np.random.sample(generate,10)
    index=np.random.permutation(list)
    for i in index:
        Columnget=X[:,index]
    return Columnget

print(get_sample(X))



