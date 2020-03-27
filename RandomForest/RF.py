from sklearn.datasets import load_digits
import numpy as np
from sklearn.ensemble import RandomForestClassifier

digits, labels = load_digits(return_X_y=True)
print(digits.shape,'###', labels.shape) #(1797, 64) ### (1797,)

permutate = np.random.permutation(digits.shape[0])
#print(permutate.shape)
perm_digits, perm_labels = digits[permutate,:], labels[permutate]
N = 200
test_data, test_labels= perm_digits[:N], perm_labels[:N]
vali_data, vali_labels = perm_digits[N:2*N], perm_labels[N:2*N]
train_data, train_labels = perm_digits[2*N:], perm_labels[2*N:]
print(vali_data.shape, vali_labels.shape)
for Nr in [5,10,20,100]:
    for Depth in [2,5,10,'pure']:
        d = None if Depth == "pure" else Depth
        clf = RandomForestClassifier(n_estimators=Nr, max_depth = d)
        clf.fit(train_data, train_labels)
        scores =clf.score(vali_data, vali_labels)
        print('Nr=', Nr, 'Depth=', Depth, scores)

Nr= 20
Depth= 10
#0.975

Nr= 100
Depth= 10
clf = RandomForestClassifier(n_estimators=Nr, max_depth=Depth)
clf.fit(np.concatenate([train_data, vali_data]), np.concatenate([train_labels, vali_labels]))
scores = clf.score(test_data, test_labels)
print(scores)
