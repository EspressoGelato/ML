import numpy as np
import matplotlib.pyplot as plt
import math
import mpl_toolkits.mplot3d

xsq = np.arange(-3,3,0.01)
ysq = np.arange(-3,3,0.01)
x, y = np.meshgrid(xsq,ysq)
print(xsq.shape)
print((x.shape))

def norm_3d(x,y,miu,sigma):
    z = 1/(sigma*np.sqrt(2*np.pi))*np.exp(-((x-miu)**2+(y-miu)**2)/2*sigma**2)
    return z
C_1 = plt.contour(x,y,norm_3d(x,y,-1,1),alpha =0.5,levels = 10)
C_2 = plt.contour(x,y,norm_3d(x,y,1,2), alpha = 0.5,levels = 10)
plt.clabel(C_1, inline =True, fontsize=10)
plt.clabel(C_2, inline =True, fontsize=10)
#print(norm_3d(x,y,-1,1))
print('1:',(norm_3d(x,y,-1,1)).shape)
print('2:',(norm_3d(x,y,1,2)).shape)
#print('#####', (norm_3d(x,y,-1,1)-norm_3d(x,y,1,2))<0.0001)
index = norm_3d(x,y,-1,1)
plt.scatter(x[norm_3d(x,y,-1,1)-norm_3d(x,y,1,2)<0.0001],y[norm_3d(x,y,-1,1)-norm_3d(x,y,1,2)<0.0001],c='red',alpha = 0.3)
plt.show()
