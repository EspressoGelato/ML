#Iteratively Reweighted Least Squares approach

import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as plt

def sigma(x):
    f=1/1+np.exp(-x)
    return f

def IRLS(X,Y,beta):
    miu=sigma(beta.T.dot(X))
    g=X.dot(miu-Y.T)
    n, p = miu.shape
    I = np.zeros(n) + 1
    M=np.diagal(miu.T.dot(I-miu))
    H = X.dot(M).dot(X.T)
    beta_new=beta-inv(H).dot(g)

    return beta_new


def Gradient_descent(X,Y,beta):
    beta_new=IRLS(X,Y,beta)
    while beta!=beta:
        beta=beta_new
        beta_new=IRLS(X,Y,beta)
    return beta_new













