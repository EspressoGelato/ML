'''
Linear Discriminant Analysis and Quadratic Discriminant Analysis are two classic classifiers,
with a linear and a quadratic decision surface, respectively.
Have closed form solutions and can be easily computed, inherently multiclass, have no hyperparameters to tune

LDA can be used to perform supervised dimensionality reduction by projecting the input data to a linear subspace
consisting of the directions which maximize the separation between classes.

'''
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
import numpy as np
from sklearn.datasets import load_digits

digits, labels = load_digits(return_X_y=True)
print(digits.shape,'###', labels.shape) #(1797, 64) ### (1797,)

permutate = np.random.permutation(digits.shape[0])
#print(permutate.shape)
perm_digits, perm_labels = digits[permutate,:], labels[permutate]
N = 200
test_data, test_labels= perm_digits[:N], perm_labels[:N]
vali_data, vali_labels = perm_digits[N:2*N], perm_labels[N:2*N]
train_data, train_labels = perm_digits[2*N:], perm_labels[2*N:]
#print(vali_data.shape, vali_labels.shape)
clf = QuadraticDiscriminantAnalysis()
clf.fit(train_data, train_labels)
#print(clf.predict(vali_data))

import matplotlib.pyplot as plt
import scipy.stats as stats
miu = 1
var = 1
x = np.arange(-7,7,0.05)
plt.plot(x,0.5*stats.norm.pdf(x,miu,var))
plt.show()