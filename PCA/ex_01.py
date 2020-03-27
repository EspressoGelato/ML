import numpy as np
import matplotlib.pyplot as plt

#import seaborn as sns
from ML.PCA.PCA import PCA_class


if __name__ == '__main__':


    X=np.loadtxt('data_ex01/protein.txt')
    X=X.T
    print(X.shape)

    countries = ["Albania", "Austria", "Belgium", "Bulgaria", "Czechoslovakia", "Denmark",
            "E Germany", "Finland", "France", "Greece", "Hungary", "Ireland", "Italy",
            "Netherlands", "Norway", "Poland", "Portugal", "Romania", "Spain", "Sweden",
            "Switzerland", "UK", "USSR", "W Germany", "Yugoslavia"]

    p, N = X.shape
    if N !=len(countries):
        print('You missed some countries.')
    R = PCA_func(X)
    x, y = R[0, :], R[1,:]
    plt.rcParams["figure.figsize"] = (20,10)
    plt.scatter(x,y)
    for i in range(N):
        plt.annotate(countries[i], (x[i], y[i]), fontsize=20)
    plt.xlabel("PC1", fontsize=20)
    plt.ylabel("PC2", fontsize=20)
    plt.title("Protein Consumption in Europe", fontsize=20, fontweight='bold')
    #sns.despine()
    plt.show()

