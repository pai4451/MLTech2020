import numpy as np
import matplotlib.pyplot as plt
from SVM import SVM


if __name__ == '__main__':
    train_data_path = "./data/features.train.txt"
    test_data_path = "./data/features.test.txt"

    s = SVM(train_data_path, test_data_path, digit=0)
    LogC = [-5, -3, -1, 1, 3]
    W = []
    for i in LogC:
        C = 10 ** i
        model = s.train(kernel="linear", C=C)
        w = model.coef_
        W.append(np.linalg.norm(w))
    plt.style.use('ggplot')
    plt.plot(LogC, W)
    plt.xlabel("$log_{10}C$")
    plt.ylabel("$||\mathbf{w}||$")
    plt.title("0 versus not 0")
    plt.savefig('./11.pdf', format="pdf")
    plt.show()
