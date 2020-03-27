import numpy as np
import matplotlib.pyplot as plt

X = np.load('knn2d.npy')
Y = np.load('knn2dlabels.npy')
print(X.shape)#(2,400)
#print(Y.shape)

#index = list(range(400))#np.linspace(0, 399, 400)#np.random.shuffle(np.linspace(0, 399, 400))
#np.random.shuffle(index)
#print(index)
class cross_validation:
    def __init__(self):
        pass
    def One_sample(self, index, st, end):
        TestIndex = index[st:end]
        TrainIndex = np.delete(index, np.arange(st,end,1))
        return TestIndex,TrainIndex

    def KNN(self, TestData, K, X, Y):
        # The shape of TestData is (2,N)
        dist = np.sum((X[:, :, None] - TestData[:, None, :]) ** 2, axis=0)  # shape(400,N)
        index = np.argsort(dist, axis=0)[0:K, :]  # shape(K,N)
        label = np.array([np.mean(Y[index[:, i]]) for i in range(index.shape[1])])
        return (label >= (np.zeros(index.shape[1]) + 0.5)) + 0

    def Text(self,K, X, Y,N = 5):
        rate = np.zeros(N)
        num_samples = len(Y)
        num_per = int(num_samples / N)
        index = np.random.permutation(X.shape[1])
        for i in range(N):
            TestIndex, TrainIndex = self.One_sample(index, num_per*i, num_per*(i+1))
            label = self.KNN(X[:,TestIndex], K, X[:, TrainIndex], Y[TrainIndex])
            rate[i] = np.sum(label == Y[TestIndex])/num_per
        return np.mean(rate)

CrossValidation = cross_validation()
Rate = np.array([CrossValidation.Text(K,X,Y) for K in range(1,21,2)])
print(Rate)
plt.plot(np.arange(1,21,2), Rate, c = 'darkblue', alpha= 0.2)
plt.show()





