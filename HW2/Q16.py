import numpy as np
import matplotlib.pyplot as plt
from utils import load_data


def PCA(x, W, x_bar):
    WWT = W @ W.T
    return WWT @ (x-x_bar) + x_bar


if __name__ == '__main__':
    X = load_data('./data/zip.train.txt')
    test_data = load_data('./data/zip.test.txt')
    x_bar = np.mean(X, axis=0).reshape(-1, )

    S, V = np.linalg.eigh(np.dot(X.T, X))
    indx = np.argsort(S)[::-1]
    S = S[indx]
    V = V[:, indx]

    Eout = []
    d_ = range(1, 8)
    for i in d_:
        k = 2**i
        W = V[:,:k] # get top d eigenvectors
        Err = 0
        for i in range(len(test_data)):
            Err += np.sum((PCA(test_data[i], W, x_bar) - test_data[i])**2)/256
        Err /= len(test_data)
        print("d_ = %d, Err = %.4f" % (k, Err))
        Eout.append(Err)

    print('Eout =', Eout)
    plt.style.use('ggplot')
    plt.plot(d_, Eout)
    plt.xlabel("$log_{2}\widetilde{d}$")
    plt.ylabel("$Eout(g)$")
    plt.title("PCA algorithm")
    plt.savefig("./16.pdf", format="pdf")
    plt.show()
