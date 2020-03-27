#Principle Component Analysis
#Find 'best' subspace to summarize high dimensional data

#Data visualization

import numpy as np
import matplotlib.pyplot as plt

#import seaborn as sns

class PCA_class:

    def __init__(self, dim):
        self.dim = dim

    def Cov(self, X):
        m = X.shape[1]
        M = np.mean(X, axis = 1, keepdims= True)
        Y = X - M
        Cov = np.dot(Y, Y.T) / m
        return Cov

    def CorChange(self, a, W):

        b = a[0:self.dim]
        P=W[:,b]
        return P.T

    def do_operation(self, X):
        Cov = self.Cov(X)
        eigenvalues, w = np.linalg.eig(Cov)
        a = np.argsort(eigenvalues)[::-1]  # from small to large

        P=np.real(self.CorChange(a,w))
        print("P shape {}".format(P.shape))
        R=np.dot(P,X)
        return P,R




if __name__ == '__main__':


    X=np.loadtxt('data_ex01/protein.txt')
    X=X.T
    print(X.shape)

    countries = ["Albania", "Austria", "Belgium", "Bulgaria", "Czechoslovakia", "Denmark",
            "E Germany", "Finland", "France", "Greece", "Hungary", "Ireland", "Italy",
            "Netherlands", "Norway", "Poland", "Portugal", "Romania", "Spain", "Sweden",
            "Switzerland", "UK", "USSR", "W Germany", "Yugoslavia"]

    p, N = X.shape
    if N !=len(countries):
        print('You missed some countries.')

    PCA = PCA_class(dim = 2)
    R = PCA.do_operation(X)

    x, y = R[0, :], R[1,:]
    plt.rcParams["figure.figsize"] = (20,10)
    plt.scatter(x,y)
    for i in range(N):
        plt.annotate(countries[i], (x[i], y[i]), fontsize=20)
    plt.show()



















'''
def PCA_func(X,dim=2):

    def Cov(X):
        m = X.shape[1]  # Get the column number of the matrix.(The number of the samples)
        # print(m)
        M = np.mean(X, axis=1, keepdims=True)  # .reshape(9,1)
        # Keepdims=True or Reshape(9,1) keep the M's dimension the same as X does.(X has axis 0 and 1)
        # Or M.shape is (9,1)
        # Otherwise M will be an array. Its shape is (9,)
        # print(M)
        #print(M.shape)
        # Y=np.repeat(m,9,axis=1)
        # print(Y)

        Y = X - M  # M automatically change to the same dimension as X.
        #print(Y.shape)
        Cov = np.dot(Y, Y.T) / m
        print(Cov.shape)
        return Cov

    Cov=Cov(X)

    eigenvalues,w =np.linalg.eig(Cov)
    #print(len(eigenvalues))
#print(eigenvalues)
    #print(w.shape)
#print(eigenvector)
    a=np.argsort(eigenvalues) #from small to large
    a=a[::-1]#Reverse the order.

    #print(a)

    def CorChange(a,W):

        b=a[0:dim]#b=a[0:2] the first two elements.

    # m = W.shape[0]
    # P=np.empty((0,m))
    #for i in b:
        P=W[:,b] # Get the corresponding column.
        #x=W[:,i].reshape(1,m) #W[:, I:(I+1)]
        #P=np.vstack((P,x))
    #P=np.delete(P,0,0)
        return P.T

    P=np.real(CorChange(a,w))#Neglect the imaginary part.

    #print(P.shape)
    #print('P=',P)
    #print('X=',X)
    #print(X.shape)

    R=np.dot(P,X)
    #print(R.shape)
    return R
'''


