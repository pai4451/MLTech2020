import numpy as np
import matplotlib.pyplot as plt
from SVM import SVM


if __name__ == '__main__':

    train_data_path = "./data/features.train.txt"
    test_data_path = "./data/features.test.txt"

    s = SVM(train_data_path, test_data_path, digit=0)

    LogGamma = range(5)
    Eout = []

    for i in LogGamma:
        gamma = 10**i
        model = s.train(kernel='rbf', gamma=gamma, C=0.1)  # shrinking=False
        err = s.calculate_error(error_type="Eout")
        Eout.append(err)

    plt.style.use('ggplot')
    plt.plot(LogGamma, Eout)
    plt.xlabel("$\log_{10}\gamma$")
    plt.ylabel("$E_{out}$")
    plt.title("0 versus not 0")
    plt.savefig('./15.pdf', format="pdf")
    plt.show()
