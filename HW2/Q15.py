import numpy as np
import matplotlib.pyplot as plt
from utils import load_data


def PCA(x, W, x_bar):
    WWT = W @ W.T
    return WWT @ (x-x_bar) + x_bar


if __name__ == '__main__':
    X = load_data('./data/zip.train.txt')
    x_bar = np.mean(X, axis=0).reshape(-1, )

    S, V = np.linalg.eigh(np.dot(X.T, X))
    indx = np.argsort(S)[::-1]
    S = S[indx]
    V = V[:, indx]

    Ein = []
    d_ = range(1, 8)
    for i in d_:
        k = 2**i
        W = V[:, :k]  # get top d eigenvectors
        Err = 0
        for i in range(len(X)):
            Err += np.sum((PCA(X[i], W, x_bar) - X[i])**2)/256
        Err /= len(X)
        print("d_ = %d, Err = %.4f" % (k, Err))
        Ein.append(Err)

    print('Ein =', Ein)
    plt.style.use('ggplot')
    plt.plot(d_, Ein)
    plt.xlabel("$log_{2}\widetilde{d}$")
    plt.ylabel("$Ein(g)$")
    plt.title("PCA algorithm")
    plt.savefig("./15.pdf", format="pdf")
    plt.show()
