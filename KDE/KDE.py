#Kernel Density Estimation
import numpy as np
import matplotlib.pyplot as plt

class KDE_class:
    def __init__(self, N, x_n):
        self.data_num = N
        self.x_n = x_n

    def K_x(self, a, b, miu, w, dis):
        x = np.arange(a, b, dis)
        k = np.square((1 - np.square((x - miu) / w))) * 15 / (16 * w)
        return x, k

    def Kernel(self, x, miu, w):
        if miu-1 < x < miu+1:
            kernel = np.square((1 - np.square((x - miu) / w))) * 15 / (16 * w)
        else:
            kernel = 0
        return kernel

    def QuarticKernel(self, w, a, b, dis=1):

        #print(x.shape)
        #print(x)
        #res = np.zeros(len(x))

        for i in range(N):
            miu = x_n[i]
            # print(miu)
            # x= np.arange(miu-1, miu+1, dis)
            x, y = self.K_x(miu - 1, miu + 1, miu, w, 0.01)

            plt.plot(x, y)
        

        x = np.arange(-10, 20, 0.1)
        result = np.zeros(len(x))
        for j in range(len(x)):
            for i in range(N):
                q = self.Kernel(x[j], x_n[i], 1)

            result[j] = np.mean(q)
            plt.plot(x,result)
        plt.show()



if __name__ == '__main__':

    N = 50
    X = np.load('samples.npy')
    # print(X)
    print(X.shape)
    x_n = X[:N]
    #print(x_n)
    KDE = KDE_class(N, x_n)
    a = -10
    b = 20

    w = 5
    #KDE.QuarticKernel(w, a, b)




'''

def biweight(x,m,w):
    u = (x - m)/w
    supp = abs(u) < 1
    res = 15/(16*w) * (1 - u**2)**2
    return res * supp
import seaborn as sns

xseq = np.arange(-5,5,0.01)
plt.plot(xseq, biweight(xseq, 0, 0.5), label=r"$w=0.5$")
plt.plot(xseq, biweight(xseq, 0, 1), label=r"$w=1$")
plt.plot(xseq, biweight(xseq, 0, 2), label=r"$w=2$")
plt.plot(xseq, biweight(xseq, 0, 3), label=r"$w=3$")
plt.legend(fontsize=20)
plt.title("Biweight kernel", fontsize=20, fontweight="bold")
sns.despine()


'''