import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from numpy.linalg import inv



def fit_beta(X,y,add_bias=True):
    if len(X.shape) == 1:
        X=X[None]
    #np.newaXis is None, X[np.newaXis]:
    # made it into a matriX,we want shape is (1,n) not (n,)
    if len(y.shape)==1:
        y=y[None]

    p,n=X.shape

    if add_bias:
        X=np.vstack([np.ones(n),X])
        beta=inv(X.dot(X.T)).dot(X).dot(y.T)
    return beta
#This is a column vector with shape(p,1)

def predict(beta,X):
    if len(X.shape)==1:
        X=X[None]
    return beta[0]+beta[1:].T.dot(X)
#A number add a vectpr, python will made the
#number beta[0] into a vector with the same shape,
# beta[1:] is a column vector.

def calc_rss(beta,X,y):
    if len(X.shape)==1:
        X=X[None]
    if len(y.shape) == 1:
        y = y[None]

    yhat=predict(beta,X)
    residuals=(y-yhat)
    ssq=np.sum(residuals**2)
    return ssq


