import numpy as np
import matplotlib.pyplot as plt
from random import sample


class MeanShift_class:
    def kernel(self, x, m, w):
        u = (x - m)/w
        supp = abs(u) < 1
        k = (1-u**2)*3/(4*w)
        return k*supp

    def weight_center(self, m, X, w=1):
        #res = []
        sum = 0
        num = 0
        for i in range(len(X)):

            d = abs(X[i]-m)
            #print('dist=', X[i]-m)
            if d < 1:
                num += 1#self.kernel(X[i], m, w)
                sum += X[i]#self.kernel(X[i],m, w) * X[i]

        m = sum / num
        #res.append(m)
        return m#np.array(res)
    def gradient_ascent(self,m,X,e=0.00000001):
        m_next = m
        m = self.weight_center(m,X)
        t = 0
        while abs(m_next - m) > e:
            m = m_next
            print(m)

            m_next = self.weight_center(m,X)
            print(m_next)
            t = t + 1
            plt.scatter(m_next, 0)
        return m_next

Data = np.load('data_ex02/meanshift1d.npy')
print(Data)
print(Data.shape)
L= Data.tolist()
m_0 = sample(L, 1)[0]
print(m_0)

MS = MeanShift_class()
MS.gradient_ascent(m_0,Data)
plt.show()