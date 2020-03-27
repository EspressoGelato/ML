
import Linear_regression.EX_1.Pre

#import sys
#sys.path.append('C:/Users/Yuchen/PycharmProjects')
#import ML.Linear_regression.EX_1.Pre
import Pre

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams["figure.figsize"] = (20,10)

tmp=np.load('C:/Users/Yuchen/PycharmProjects/MLsheet4/Linear_regression/EX_1/linear-res.npz')


tmp=np.load('./linear-res.npz')

X=tmp['X']
Y=tmp['Y']
plt.scatter(X,Y)
sns.despine()


#fit with


beta= Linear_regression.EX_1.Pre.fit_beta(X, Y)

beta=Pre.fit_beta(X, Y)

Xseq = np.arange(-5,5,0.1)
Yhat=beta[0]+beta[1]*Xseq

Xsq=np.vstack([X,X**2])#square each element

betasq= Linear_regression.EX_1.Pre.fit_beta(Xsq, Y)

betasq=Pre.fit_beta(Xsq, Y)

Xseq=np.arange(-5,5,0.1)
Yhatsq=betasq[0]+betasq[1]*Xseq+betasq[2]*Xseq**2
plt.scatter(X,Y)
plt.plot(Xseq,Yhat,label='linear features')
plt.plot(Xseq,Yhatsq,label='squared features')
plt.legend(fontsize=20)
sns.despine()
plt.title('fit',fontsize=20)


#get residuals for each case
#plt.scatter(X,Y-Yhat,label='linear features')
plt.scatter(X, Y - (betasq[0] + betasq[1] * X + betasq[2] * X**2), label="squared features")
plt.legend(fontsize=20)
sns.despine()
plt.title('residuals',fontsize=20)
plt.show()


