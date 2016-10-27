import sys, csv, random, json, copy
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
        # X.append(row[1:-1])
    #     X.append(list(row[1:-1]) + map(lambda x: x ** 0.5, row[-10:-1]))
    #     X.append(list(row[1:-1]) + map(lambda x: x ** 0.5, row[-10:-1]) + map(lambda x: x ** 0.25, row[-10:-1]))
        X.append(list(np.sqrt(row[1:-1])) + list(np.log(row[-4:-1])))
    #     X.append(list(np.sqrt(row[1:-1])) + list(np.square(row[-10:-4])) + list(np.log(row[-4:-1])))
    #     X.append(list(row[1:-1]) + list(np.square(row[-10:-4])) + list(np.log(row[-4:-1])))
        Y.append(row[-1])
    X = np.array(X)
    Y = np.array(Y)
    return X, Y

def multi_logistic(neuron, layer, X, Y):
    W_dic, b_dic = {}, {}
    X_NEXT, WT, bT = backward_prop(X, Y, neuron)
    W_dic, b_dic = update_neuron_info(W_dic, b_dic, WT, bT, neuron, 0)
    for i in range(1, layer):
        X_NEXT, WT, bT = backward_prop(X_NEXT, Y, neuron)
        W_dic, b_dic = update_neuron_info(W_dic, b_dic, WT, bT, neuron, i)
    W_dic, b_dic = classification(X_NEXT, Y, W_dic, b_dic, neuron, layer)
    return W_dic, b_dic

def logistic_regression(X_TRAIN, Y_TRAIN):
    W = np.random.uniform(low=-1, high=1, size=X_TRAIN.shape[1])
    b = np.random.uniform(low=-1, high=1, size=1)
    SUM_SQDW, SUM_SQDB = np.zeros(X_TRAIN.shape[1]), 0
    norm, adag, adam = 0.00000001, 1, 0.001 # adam-default = 0.001
    beta1, beta2 = 0.9, 0.999
    Wmt, Wvt = 0, 0
    Bmt, Bvt = 0, 0
    epoch, Lambda, t, eps = 7500, 0, 0, 1e-8
    for i in range(epoch):
        fwb = sigmoid(np.dot(X_TRAIN, W) + b)
        ERR = Y_TRAIN - fwb
        DW = -1 * np.dot(X_TRAIN.T, ERR)
        DB = -1 * np.sum(ERR)

        # Compute Loss & Print
        # if i % 500 == 0:
        #     Loss = np.sum(cross_entropy(fwb, Y_TRAIN))
        #     print "Iter %7s | Loss: %.7f" % (i, Loss)

        # Regularization
        DW += Lambda * 2 * W

        # Normal
        # W -= norm * DW # / X_TRAIN.shape[0]
        # b -= norm * DB # / X_TRAIN.shape[0]

        # Adagrad
        SUM_SQDW += np.square(DW)
        SUM_SQDB += np.square(DB)
        W -= adag / np.sqrt(SUM_SQDW) * DW # / X_TRAIN.shape[0]
        b -= adag / np.sqrt(SUM_SQDB) * DB # / X_TRAIN.shape[0]

        # Adamgrad
        # t += 1
        # Wmt = beta1 * Wmt + (1-beta1) * DW
        # Wvt = beta2 * Wvt + (1-beta2) * np.square(DW)
        # Wmthat = Wmt / (1-np.power(beta1, t))
        # Wvthat = Wvt / (1-np.power(beta2, t))
        # Bmt = beta1 * Bmt + (1-beta1) * DB
        # Bvt = beta2 * Bvt + (1-beta2) * np.square(DB)
        # Bmthat = Bmt / (1-np.power(beta1, t))
        # Bvthat = Bvt / (1-np.power(beta2, t))
        # W -= (adam*Wmthat) / (np.sqrt(Wvthat) + eps)
        # b -= (adam*Bmthat) / (np.sqrt(Bvthat) + eps)
    return W, b

def backward_prop(X_TRAIN, Y_TRAIN, neuron):
    WFT, bFT,  = [], []
    X_NEXT = [[] for i in range(neuron)]
    for i in range(neuron):
        tW, tb = logistic_regression(X_TRAIN, Y_TRAIN)
        WFT.append(tW)
        bFT.append(tb)
        fwb = sigmoid(np.dot(X_TRAIN, tW) + tb)
        for j in range(fwb.shape[0]):
            X_NEXT[i].append(fwb[j])  # shape=(neuron, 4001)
    X_NEXT = np.array(X_NEXT).T  # shape=(4001, neuron)
    return X_NEXT, WFT, bFT

def update_neuron_info(W_dic, b_dic, W, b, neuron, layer):
    wdic, bdic = {}, {}
    for i in range(neuron):
        wdic[i], bdic[i] = W[i], b[i]
    W_dic[layer], b_dic[layer] = wdic, bdic
    return W_dic, b_dic

def classification(X_TRAIN, Y_TRAIN, W_dic, b_dic, neuron, layer):
    X_TRANS = [[] for i in range(neuron)]
    for i in range(neuron):
        fwb = sigmoid(np.dot(X_TRAIN, W_dic[layer-1][i]) + b_dic[layer-1][i])
        for j in range(fwb.shape[0]):
            X_TRANS[i].append(fwb[j]) # shape=(neuron, 4001)
    X_TRANS = np.array(X_TRANS).T     # shape=(4001, neuron)
    WC, bC = logistic_regression(X_TRANS, Y_TRAIN) # shape=(neuron, 1)
    W_dic[layer], b_dic[layer] = WC, bC
    return W_dic, b_dic

def gen_model(modelpath, W_dic, b_dic, neuron, layer):
    with open(modelpath, 'wb') as file:
        dic = {}
        dic['W'], dic['b'] = {}, {}
        for i in range(layer):
            dic['W'][i], dic['b'][i] = {}, {}
            for j in range(neuron):
                dic['W'][i][j] = list(W_dic[i][j])
                dic['b'][i][j] = float(b_dic[i][j])
        dic['W'][layer] = list(W_dic[layer])
        dic['b'][layer] = float(b_dic[layer])
        json.dump(dic, file)

def read_model(modelpath):
    with open(modelpath, 'rb') as file:
        dic = json.load(file)
        W_dic, b_dic = dic['W'], dic['b']
    return W_dic, b_dic

def read_test(testpath):
    test = np.genfromtxt(testpath, delimiter=',') # (4320, 11)
    X_TEST = []
    for row in test:
        # X_TEST.append(row[1:])
    #     X_TEST.append(list(row[1:]) + map(lambda x: x ** 0.5, row[-9:]))
    #     X_TEST.append(list(np.sqrt(row[1:])) + list(np.log(row[-3:])))
    #     X_TEST.append(list(np.sqrt(row[1:])) + list(np.square(row[-9:-3])) + list(np.log(row[-3:])))
    #     X_TEST.append(list(row[1:]) + list(np.square(row[-9:-3])) + list(np.log(row[-3:])))
        X_TEST.append(list(np.sqrt(row[1:])) + list(np.log(row[-3:])))
    X_TEST = np.array(X_TEST)
    return X_TEST

def gen_ans(anspath, X_TEST, W_dic, b_dic, neuron, layer):
    X_NEXT = [[] for j in range(neuron)]
    for i in range(neuron):
        fwb = sigmoid(np.dot(X_TEST, np.array(W_dic[unicode(0)][unicode(i)])) + b_dic[unicode(0)][unicode(i)])
        for j in range(fwb.shape[0]):
            X_NEXT[i].append(fwb[j])
    X_NEXT = np.array(X_NEXT).T

    for i in range(1, layer):
        X_TEMP = [[] for k in range(neuron)]
        for j in range(neuron):
            fwb = sigmoid(np.dot(X_NEXT, np.array(W_dic[unicode(i)][unicode(j)])) + b_dic[unicode(i)][unicode(j)])
            for k in range(fwb.shape[0]):
                X_TEMP[j].append(fwb[k])
        X_NEXT = np.array(X_TEMP).T
    fwb_TEST = sigmoid(np.dot(X_NEXT, W_dic[unicode(layer)]) + b_dic[unicode(layer)])

    Y_TEST = []
    for i in fwb_TEST:
        Y_TEST.append(0) if np.less_equal(i, 0.5) else Y_TEST.append(1)
    with open(anspath, 'wb') as file:
        writer = csv.writer(file, delimiter=',')
        writer.writerow(('id', 'label'))
        for i in range(len(Y_TEST)): writer.writerow((i+1, Y_TEST[i]))


if __name__ == '__main__':
    layer = 5
    neuron = 4
    if len(sys.argv) == 3: # train
        X, Y = read_data(sys.argv[1])
        W_dic, b_dic = multi_logistic(neuron, layer, X, Y)
        gen_model(sys.argv[2], W_dic, b_dic, neuron, layer)
    elif len(sys.argv) == 4:
        W_dic, b_dic = read_model(sys.argv[1])
        X = read_test(sys.argv[2])
        gen_ans(sys.argv[3], X, W_dic, b_dic, neuron, layer)
    else:
        pass
    # X, Y = read_data('./spam_train.csv')
    # W_dic, b_dic = {}, {}
    # X_NEXT, WT, bT = backward_prop(X, Y, neuron)
    # W_dic, b_dic = update_neuron_info(W_dic, b_dic, WT, bT, neuron, 0)
    # for i in range(1, layer):
    #     X_NEXT, WT, bT = backward_prop(X_NEXT, Y, neuron)
    #     W_dic, b_dic = update_neuron_info(W_dic, b_dic, WT, bT, neuron, i)
    # W_dic, b_dic = classification(X_NEXT, Y, W_dic, b_dic, neuron, layer)
    # gen_model('./multi-logistic_model', W_dic, b_dic, neuron, layer)
    # W_dic, b_dic = read_model('./multi-logistic_model')
    # X = read_test('./spam_test.csv')
    # gen_ans('./multi-logistic.csv', X, W_dic, b_dic, neuron, layer)
