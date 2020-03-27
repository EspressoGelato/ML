import numpy as np
import matplotlib.pyplot as plt

def biweight(x,m,w):
    u = (x - m)/w
    supp = abs(u) < 1
    res = 15/(16*w) * (1 - u**2)**2
    return res * supp

xseq = np.arange(-5,5,0.01)
plt.plot(xseq, biweight(xseq, 0, 0.5), label=r"$w=0.5$")
plt.plot(xseq, biweight(xseq, 0, 1), label=r"$w=1$")
plt.plot(xseq, biweight(xseq, 0, 2), label=r"$w=2$")
plt.plot(xseq, biweight(xseq, 0, 3), label=r"$w=3$")
plt.legend(fontsize=20)
plt.title("Biweight kernel", fontsize=20, fontweight="bold")
plt.show()

def epanechnikov(x,m,w=1):
    u = (x - m)/w
    supp = abs(u) < 1
    res = 3/(4*w) * (1 - u**2)
    return res * supp

xseq = np.arange(-5,5,0.01)
plt.plot(xseq, epanechnikov(xseq, 0, 0.5), label=r"$w=0.5$")
plt.plot(xseq, epanechnikov(xseq, 0, 1), label=r"$w=1$")
plt.plot(xseq, epanechnikov(xseq, 0, 2), label=r"$w=2$")
plt.plot(xseq, epanechnikov(xseq, 0, 3), label=r"$w=3$")
plt.legend(fontsize=20)
plt.title("Epanechnikov kernel", fontsize=20, fontweight="bold")