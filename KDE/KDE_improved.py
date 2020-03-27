import numpy as np
import matplotlib.pyplot as plt
from random import sample

x = np.arange(-10, 20, 0.01)  # x = np.linspace(a,b,num = n)
N = 50
Data = np.load('samples.npy')
print(Data)
print(Data.shape)

def Qkernel(x, m, w):
    u = (x - m)/w
    supp = abs(u) < 1
    k = ((1-u**2)**2)*15/(16*w)
    return k*supp

'''
def QuarticKernel(w,miu):

    k = np.zeros(len(x))
    for i in range(len(x)):
        if miu - w < x[i] < miu + w:
            k[i] = 15 / (16 * w) * (1 - ((x[i] - miu) / w) ** 2) ** 2
        else:
            k[i] = 0
    return x, k

def plot_single_kernel():
    for w in [0.5,1,2,3]:
        x, y = QuarticKernel(w,0)
        plt.plot(x, y)
    plt.show()

def separate_KDE(N,w):
    x_n = Data[:N]
    for i in range(len(x_n)):
        miu = x_n[i]
        x, k = QuarticKernel(w,miu)
        plt.plot(x,k)

'''
def KDE(N,w,x,S):
    #x_n = Data[:N]
    L = Data.tolist()
    for k in range(S):
        x_n = sample(L, N)
        sum = 0
        for j in range(N):
            miu = x_n[j]
            x, k = Qkernel(w,miu)
            sum = sum + k
        value = sum/N
        #print(value)

        plt.plot(x, value, color='blue', alpha=0.1)




'''
for w in [0.5,1,2,3]:
    KDE(50,w,x)
plt.show()
'''
if __name__ =='__main__':

    KDE(50,1,x,10)
    plt.show()








