import numpy as np
import matplotlib.pyplot as plt
from SVM import SVM


if __name__ == '__main__':
    train_data_path = "./data/features.train.txt"
    test_data_path = "./data/features.test.txt"

    s = SVM(train_data_path, test_data_path, digit=0)

    N = 100
    Cnt = np.zeros(5)
    LogGamma = range(-1, 4)
    for _ in range(N):
        s.train_valid_split(validation_size=1000, digit=0)
        Eval = []
        for i in LogGamma:
            Gamma = 10 ** i
            model = s.train(kernel='rbf', gamma=Gamma, C=0.1)
            err = s.calculate_error(error_type="Eval")
            Eval.append(err)

        index = np.argmin(Eval)
        Cnt[index] += 1

    plt.style.use('ggplot')
    plt.bar(LogGamma, Cnt)
    plt.xlabel("$\log_{10}\gamma$")
    plt.ylabel("Frequency")
    plt.title("0 versus not 0")
    plt.savefig('./16.pdf', format="pdf")
    plt.show()
