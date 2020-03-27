#plot the Ridge regression regularization
import matplotlib
import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt


def L2_r(delta=0.01):
    x=np.arange(-1,3,delta)
    y=np.arange(-1,3,delta)
    X,Y=np.meshgrid(x,y)
    #print(x,y)
    #print(X,Y)
    Z =X**2+Y**2
    plt.contour(X, Y, Z)




