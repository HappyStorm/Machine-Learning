import sys, csv, random, json
import numpy as np


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def read_data(datapath):
    data = np.genfromtxt(datapath, delimiter=',')  # (4001, ID + 57 + label)
    X, Y = [], []
    for row in data:
        #     X.append(row[1:-1])
        X.append(list(row[1:-1]) + map(lambda x: x ** 0.5, row[-10:-1]))
        #     X.append(list(row[1:-1]) + map(lambda x: x ** 0.5, row[-10:-1]) + map(lambda x: x ** 0.25, row[-10:-1]))
        Y.append(row[-1])
    X = np.array(X)
    Y = np.array(Y)
    return X, Y


def gen_mean(X, Y):
    X_0, X_1 = [], []
    for i in range(Y.shape[0]):
        X_0.append(X[i]) if Y[i] == 0 else X_1.append(X[i])
    X_0, X_1 = np.array(X_0).T, np.array(X_1).T
    u_0, u_1 = [], []
    col = X.shape[1]
    for i in range(col):
        u_0.append(np.mean(X_0[i][:]))
        u_1.append(np.mean(X_1[i][:]))
    return np.array(u_0), np.array(u_1), X_0.shape[1], X_1.shape[1]


def gen_cov(X):
    return np.cov(X, rowvar=False)


def gen_model(modelpath, u_0, u_1, cov, N_0, N_1):
    inv_cov = np.linalg.inv(cov)
    W = np.dot((u_0 - u_1).T, inv_cov).T
    b = -0.5 * np.dot(np.dot(u_0.T, inv_cov), u_0) + \
         0.5 * np.dot(np.dot(u_1.T, inv_cov), u_1) + \
         np.log(N_0 / N_1)
    with open(modelpath, 'wb') as file:
        json.dump({'b': b, 'W': list(W)}, file)


def read_model(modelpath):
    with open(modelpath, 'rb') as file:
        dic = json.load(file)
    return dic['W'], dic['b']


def read_test(testpath):
    test = np.genfromtxt(testpath, delimiter=',')  # (4320, 11)
    X_TEST = []
    for row in test:
        #     X_TEST.append(row[1:])
        X_TEST.append(list(row[1:]) + map(lambda x: x ** 0.5, row[-9:]))
    X_TEST = np.array(X_TEST)
    return X_TEST


def gen_ans(anspath, X_TEST, W, b):
    Y_TEST = []
    for x in X_TEST:
        pc0_x = sigmoid(np.dot(x, W) + b)
        Y_TEST.append(0) if np.greater(pc0_x, 0.5) else Y_TEST.append(1)
    with open(anspath, 'wb') as file:
        writer = csv.writer(file, delimiter=',')
        writer.writerow(('id', 'label'))
        for i in range(len(Y_TEST)): writer.writerow((i + 1, Y_TEST[i]))


if __name__ == '__main__':
    if len(sys.argv) == 3:  # train
        X, Y = read_data(sys.argv[1])
        u_0, u_1, N_0, N_1 = gen_mean(X, Y)
        cov = gen_cov(X)
        gen_model(sys.argv[2], u_0, u_1, cov, N_0, N_1)
    elif len(sys.argv) == 4:
        W, b = read_model(sys.argv[1])
        X = read_test(sys.argv[2])
        gen_ans(sys.argv[3], X, W, b)
    else:
        pass
