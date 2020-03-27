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

Data = np.load('data_ex02/meanshift1d.npy')
print(Data)
print(Data.shape)

from random import sample

def KDE(N,w,x,S):
    x_n = Data
    for k in range(S):
        sum = 0
        for j in range(N):
            miu = x_n[j]
            x, k = Epanechnikov_kernel(x,miu,w)
            sum = sum + k
        value = sum/N
        #print(value)

        plt.plot(x, value, color='blue', alpha=0.1)

x = np.arange(-5, 5, 0.01)
KDE(20,1,x,1)
#plt.show()

class Update_mean:

    def dist(self,x,y):

        dist = abs(x-y)
        return dist

    def get_mean(self,X,m):
        sum = 0
        num = 0
        for i in range(len(X)):
            d = self.dist(X[i], m)
            if d < 1:
                num = num + 1
                sum = sum + X[i]
        m = sum/num
        return m


    def gradient_ascent(self,m,X,e=0.5):
        m_next = m
        m = self.get_mean(X, m)
        t = 0
        while self.dist(m_next, m) > e:
            m = m_next
            m_next = self.get_mean(X, m)
            t = t + 1
            plt.scatter(m_next,0)

        return m_next




KM = Update_mean()

def mean_shift(X):
    L= X.tolist()
    m_0 = sample(L, 1)
    print(m_0)
    KM.gradient_ascent(m_0, X)


mean_shift(Data)
plt.show()