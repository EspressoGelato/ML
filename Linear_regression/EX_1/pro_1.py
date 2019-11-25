#One diagnostic tool for verfication that the assumption
#of a linear model was resonably correct is the plotting
#of the residuals. Given our assumption we consider them
# to be spread around zero with a constant variation and
#no discernible pattern
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

tmp = np.load('linear-res.npz')
#print(list(tmp.keys()))
#print(tmp['X'])
#print(tmp['Y'])
#print(tmp['X'].shape,tmp['Y'].shape)


def Augmented_matrix_x (X,column_num):
    row = np.zeros([1, column_num]) + 1
    X_aug = np.row_stack((row,X))
    return X_aug


def Caculate_Y_hat(X,Y):
    B = np.dot(X,X.T)
    inverseB=np.linalg.inv(B)
    A = np.dot(inverseB,X)
    beta_hat = np.dot(A,Y.T)
    Y_hat = np.dot(beta_hat.T,X)
    return Y_hat

def Square_x(X):
    M = np.power(X,2)
    return M


if __name__=='__main__':

    X = Augmented_matrix_x(tmp['X'],20)
    Y_hat = Caculate_Y_hat(X,tmp['Y'])
    X_2= Augmented_matrix_x(Square_x(tmp['X']),20)
    Y_hat_2 = Caculate_Y_hat(X_2,tmp['Y'])

    plt.figure()
    plt.scatter(tmp['X'],tmp['Y'])
    plt.plot(tmp['X'],Y_hat)
    plt.show()

    plt.figure()
    plt.scatter(Square_x(tmp['X']),tmp['Y'])
    plt.plot(Square_x(tmp['X']),Y_hat_2)
    plt.show()
