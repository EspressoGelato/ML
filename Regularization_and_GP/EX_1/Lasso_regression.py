#plot the Ridge regression regularization
import matplotlib
import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from Ridge_regression import L2_r

delta=0.01
x=np.arange(-1,3,delta)
y=np.arange(-1,3,delta)
X,Y=np.meshgrid(x,y)
Z=abs(X)+abs(Y)

L2_r()

#plt.figure(figsize=(10,6))
plt.contour(X,Y,Z)

plt.show()