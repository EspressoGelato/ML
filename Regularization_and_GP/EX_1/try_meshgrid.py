import numpy as np

x=np.arange(-3,3,1)
y=np.arange(-3,3,2)
X,Y=np.meshgrid(x,y)
print(x,y)
print('X=',X)
print('Y=',Y)

z=X+Y
print('z=',z)
Z=X**2+Y**2
print('Z=',Z)
