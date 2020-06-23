import numpy as np
from sklearn import svm


class SVM(object):
    def __init__(self, train_data_path, test_data_path, digit):
        self.x_train, self.y_train = self.load_data(train_data_path)
        self.x_test, self.y_test = self.load_data(test_data_path)
        self.data = np.c_[self.x_train, self.y_train]
        self.train_label_data = 2 * (self.y_train == digit) - 1
        self.test_label_data = 2 * (self.y_test == digit) - 1

    def load_data(self, path):
        data = np.genfromtxt(path)
        y, X = data[:, 0], data[:, 1:]
        return X, y

    def train(self, kernel='rbf', C=1.0, degree=3, gamma='scale', coef0=0.0):
        self.kernel = kernel
        self.C = C
        self.degree = degree
        self.gamma = gamma
        self.coef0 = coef0

        self.model = svm.SVC(kernel=self.kernel, C=self.C, degree = self.degree,
                             gamma=self.gamma, coef0=self.coef0)
        self.model.fit(self.x_train, self.train_label_data)
        return self.model

    def calculate_error(self, error_type):
        if error_type == "Ein":
            return np.mean(self.model.predict(self.x_train) != self.train_label_data)
        if error_type == "Eout":
            return np.mean(self.model.predict(self.x_test) != self.test_label_data)
        if error_type == "Eval":
            return np.mean(self.model.predict(self.x_val) != self.val_label_data)

    def calculate_distance(self):
        X = self.x_train[self.model.support_]
        XTX =  X.dot(X.T)
        diagX_1 = np.diagonal(XTX).reshape(-1,1).dot(np.ones((X.shape[0],1)).T)
        D = diagX_1 + diagX_1.T - 2*XTX
        K = np.exp(- self.gamma * D)
        ya = self.model.dual_coef_[0]
        w2 = ya.dot(K).dot(ya.T)
        distance = 1 / np.sqrt(w2)
        return distance

    def train_valid_split(self, validation_size, digit):
        idx = np.arange(len(self.data))
        np.random.shuffle(idx)
        train_data = self.data[idx[validation_size:]]
        val_data = self.data[idx[:validation_size]]
        self.x_train = train_data[:, :2]
        self.y_train = train_data[:, 2]
        self.train_label_data = 2 * (self.y_train == digit) - 1
        self.x_val = val_data[:, :2]
        self.y_val = val_data[:, 2]
        self.val_label_data = 2 * (self.y_val == digit) - 1
