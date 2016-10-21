import sys, csv, random, json
import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def cross_entropy(fx_n, y_n):
    ans = []
    for i in range(fx_n.shape[0]):
        if np.isclose(fx_n[i], 0.): 
            ans.append(y_n[i] * np.log(fx_n[i]+1e-3) + (1-y_n[i]) * np.log(1-fx_n[i]))
        elif np.isclose(1-fx_n[i], 0.):
            ans.append(y_n[i] * np.log(fx_n[i]) + (1-y_n[i]) * np.log(1-fx_n[i]+1e-3))
        else:
            ans.append(y_n[i] * np.log(fx_n[i]) + (1-y_n[i]) * np.log(1-fx_n[i]))
    return -np.array(ans)
     
def read_data(datapath):
    data = np.genfromtxt(datapath, delimiter=',') # (4001, ID + 57 + label)
    X, Y = [], []
    for row in data:
    #     X.append(row[1:-1])
        X.append(list(row[1:-1]) + map(lambda x: x ** 0.5, row[-10:-1]))
    #     X.append(list(row[1:-1]) + map(lambda x: x ** 0.5, row[-10:-1]) + map(lambda x: x ** 0.25, row[-10:-1]))
        Y.append(row[-1])
    X = np.array(X)
    Y = np.array(Y)
    return X, Y

def logistic_regression(X_TRAIN, Y_TRAIN):
    W, b = np.zeros(X_TRAIN.shape[1]), 0
    SUM_SQDW, SUM_SQDB = np.zeros(X_TRAIN.shape[1])+1, 0
    norm, adag, adam = 0.00000001, 0.1, 0.01 # adam-default = 0.001
    beta1, beta2 = 0.9, 0.999
    Wmt, Wvt = 0, 0
    Bmt, Bvt = 0, 0
    epoch, Lambda, t, eps = 10000, 0, 0, 1e-8
    for i in range(epoch):
        fwb = sigmoid(np.dot(X_TRAIN, W) + b)
        ERR = Y_TRAIN - fwb
        DW = -1 * np.dot(X_TRAIN.T, ERR)
        DB = -1 * np.sum(ERR)

        # Compute Loss & Print
        if i % 500 == 0:
            Loss = np.sum(cross_entropy(fwb, Y_TRAIN))
            print "Iter %7s | Loss: %.7f" % (i, Loss)

        # Regularization
        DW += Lambda * 2 * W

        # Normal
        # W -= norm * DW # / X_TRAIN.shape[0]
        # b -= norm * DB # / X_TRAIN.shape[0]

        # Adagrad
        # SUM_SQDW += np.square(DW)
        # SUM_SQDB += np.square(DB)
        # W -= adag / np.sqrt(SUM_SQDW) * DW # / X_TRAIN.shape[0]
        # b -= adag / np.sqrt(SUM_SQDB) * DB # / X_TRAIN.shape[0]

        # Adamgrad
        t += 1
        Wmt = beta1 * Wmt + (1-beta1) * DW
        Wvt = beta2 * Wvt + (1-beta2) * np.square(DW)
        Wmthat = Wmt / (1-np.power(beta1, t))
        Wvthat = Wvt / (1-np.power(beta2, t))
        Bmt = beta1 * Bmt + (1-beta1) * DB
        Bvt = beta2 * Bvt + (1-beta2) * np.square(DB)
        Bmthat = Bmt / (1-np.power(beta1, t))
        Bvthat = Bvt / (1-np.power(beta2, t))
        W -= (adam*Wmthat) / (np.sqrt(Wvthat) + eps)
        b -= (adam*Bmthat) / (np.sqrt(Bvthat) + eps)
    return W, b

def gen_model(modelpath, W, b):
    with open(modelpath, 'wb') as file:
        json.dump({'b': b, 'W': list(W)}, file)

def read_model(modelpath):
    with open(modelpath, 'rb') as file:
        dic = json.load(file)
    return dic['W'], dic['b']

def read_test(testpath):
    test = np.genfromtxt(testpath, delimiter=',') # (4320, 11)
    X_TEST = []
    for row in test:
    #     X_TEST.append(row[1:])
        X_TEST.append(list(row[1:]) + map(lambda x: x ** 0.5, row[-9:]))
    X_TEST = np.array(X_TEST)
    return X_TEST

def gen_ans(anspath, X_TEST, W, b):
    fwb_TEST = sigmoid(np.dot(X_TEST, W) + b)
    Y_TEST = []
    for i in fwb_TEST:
        Y_TEST.append(0) if np.less_equal(i, 0.5) else Y_TEST.append(1)
    with open(anspath, 'wb') as file:
        writer = csv.writer(file, delimiter=',')
        writer.writerow(('id', 'label'))
        for i in range(len(Y_TEST)): writer.writerow((i+1, Y_TEST[i]))


if __name__ == '__main__':
    if len(sys.argv) == 3: # train
        X, Y = read_data(sys.argv[1])
        W, b = logistic_regression(X, Y)
        gen_model(sys.argv[2], W, b)
    elif len(sys.argv) == 4:
        W, b = read_model(sys.argv[1])
        X = read_test(sys.argv[2])
        gen_ans(sys.argv[3], X, W, b)
    else:
        pass