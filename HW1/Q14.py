import numpy as np
import matplotlib.pyplot as plt
from SVM import SVM


if __name__ == '__main__':

    train_data_path = "./data/features.train.txt"
    test_data_path = "./data/features.test.txt"

    s = SVM(train_data_path, test_data_path, digit=0)

    Distance = []
    LogC = range(-3, 2)
    for i in LogC:
        C = 10**i
        model = s.train(kernel='rbf', gamma=80, C=C)  # , shrinking=False
        distance = s.calculate_distance()
        Distance.append(distance)

    plt.style.use('ggplot')
    plt.plot(LogC, Distance)
    plt.xlabel("$log_{10}C$")
    plt.ylabel("Distance")
    plt.title("0 versus not 0")
    plt.savefig('./14.pdf', format="pdf")
    plt.show()
