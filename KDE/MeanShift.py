import numpy as np
import matplotlib.pyplot as plt

def Epanechnikov_kernel(x,miu,w):
    k = np.zeros(len(x))
    for i in range(len(x)):
        u = (x - miu)/w

        if abs(u[i]) < 1:
            k[i] = (1-u[i]**2)*3/(4*w)
        else:
            k[i] = 0
    return x, k

def plot_kernel(x):
    #x = np.arange(-10,10,0.1)
    x, k = Epanechnikov_kernel(x,0,1)
    plt.plot(x,k)
    plt.show()


class Update_mean:

    def dist(self,x,y):
        dist = np.linalg.norm(x,y)
        return dist

    def get_mean(self,X,m):
        for i in range(len(X)):
            sum = 0
            num = 0
            if self.dist(X[i],m) < 1:
                num = num + 1
                sum = sum + self.dist*X[i]
        m = sum/num
        return m


    def gradient_ascent(self,m,X,e):
        m_next = m
        m = self.get_mean(X, m)
        t = 0
        while self.dist(m_next, m) > e:
            m_next = self.get_mean(X, m)
            t = t + 1
        return m_next




