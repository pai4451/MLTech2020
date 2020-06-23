import numpy as np
import matplotlib.pyplot as plt
from SVM import SVM


if __name__ == '__main__':
    train_data_path = "./data/features.train.txt"
    test_data_path = "./data/features.test.txt"

    s = SVM(train_data_path, test_data_path, digit=8)

    LogC = [-5, -3, -1, 1, 3]
    Ein = []
    N_support = []
    for i in LogC:
        C = 10 ** i
        model = s.train(kernel='poly',
                        degree=2, coef0=1, gamma=1, C=C)
        err = s.calculate_error(error_type="Ein")
        n_support = len(model.support_)
        N_support.append(n_support)
        Ein.append(err)

    plt.style.use('ggplot')
    plt.plot(LogC, Ein)
    plt.xlabel("$log_{10}C$")
    plt.ylabel("$E_{in}$")
    plt.title("8 versus not 8")
    plt.savefig('./12.pdf', format="pdf")
    plt.show()
