import numpy as np
import matplotlib.pyplot as plt
data = np.load('knn2d.npy')
label = np.load('knn2dlabels.npy')
import seaborn as sns
# labels = np.concatenate([np.zeros(N),np.ones(N)])   ??

#Generate testdata
xseq = np.arange(np.min(data[0])-0.1, np.max(data[0])+0.1, 0.1)
yseq = np.arange(np.min(data[1])-0.1, np.max(data[1])+0.1, 0.1)
print(xseq.shape, yseq.shape)
xgrid, ygrid = np.meshgrid(xseq, yseq)
print(xgrid.shape, ygrid.shape)
testdata = np.vstack([xgrid.reshape(1,-1),ygrid.reshape(1,-1)])
print(testdata.shape)
#
#plt.show()
##############
dist = np.sum((data[:,:,None]-testdata[:,None])**2, 0)
'''
for K in [1,3,5,11,21,51]:
    pred = label[dist.argsort(0)[:K]].mean(0)

    plt.scatter(testdata[0, :], testdata[1, :], color='gray', marker='.', alpha=0.1)
    plt.scatter(data[0, label == 0], data[1, label == 0], alpha=0.5)
    plt.scatter(data[0, label == 1], data[1, label == 1], alpha=0.5)
    plt.xlim(np.min(data[0]) - 0.1, np.max(data[0]) + 0.1)  #
    plt.ylim(np.min(data[1]) - 0.1, np.max(data[1]) + 0.1)

    plt.scatter(testdata[0,pred<0.5],testdata[1,pred<0.5],marker = '.', color = sns.color_palette()[0],alpha = 0.2)
    plt.scatter(testdata[0, pred > 0.5], testdata[1, pred > 0.5], marker=".", color=sns.color_palette()[1],
                alpha=0.2)
    plt.xlim(np.min(data[0]) - 0.1, np.max(data[0]) + 0.1)
    plt.ylim(np.min(data[1]) - 0.1, np.max(data[1]) + 0.1)
    plt.title(f"K={K}", fontsize=20)
    plt.show()
'''
def run_cv(data, labels, folds = 5, K = 1, vis = False):
    N = data.shape[1]
    assert N % folds == 0, 'please come up a new splitting'
    size_fold = int(N / folds)
    permute = np.random.permutation(N)
    perm_data = data[:,permute]
    perm_label = label[permute]
    cvdata = [(perm_data[:, (size_fold*c):(size_fold*(c+1))], perm_label[(size_fold*c):(size_fold*(c+1))]) for c in range(folds)]
    avg_acc = np.zeros(folds)
    for c in range(folds):
        testdata, testlabel = cvdata[c]
        traindata = np.hstack([cvdata[i][0] for i in range(folds) if i !=c])
        trainlabel = np.concatenate([cvdata[i][1] for i in range(folds) if i !=c])
        dist = np.sum((traindata[:, :, None] - testdata[:, None]) ** 2, 0)
        pred = trainlabel[dist.argsort(0)[:K]].mean(0) > 0.5
        avg_acc[c] = np.mean(pred == testlabel)

        if vis:
            plt.scatter(traindata[0, trainlabel == 0], traindata[1, trainlabel == 0], color=sns.color_palette()[0])
            plt.scatter(traindata[0, trainlabel == 1], traindata[1, trainlabel == 1], color=sns.color_palette()[1])
            plt.show()
        return avg_acc


folds = 5
acc = run_cv(data, label, folds, K=0)
overall = [np.mean(acc),]
plt.scatter(np.zeros(len(acc)), acc, color="darkblue", alpha=0.1)
for K in np.arange(1, 201, 2):
    acc = run_cv(data, label, folds, K)
    plt.scatter(K*np.ones(len(acc)), acc, color="darkblue", alpha=0.1)
    overall.append(np.mean(acc))
plt.plot(np.hstack([np.zeros(1), np.arange(1, 201, 2)]), overall)
sns.despine()
plt.title("Cross Validation for K-NN", fontsize=20, fontweight="bold")
plt.ylabel("Accuracy", fontsize=20)
plt.xlabel("Nr of Neighbors", fontsize=20)
plt.show()
