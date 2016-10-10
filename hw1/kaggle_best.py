
# coding: utf-8

# ### 1. Read Training Data
import sys, csv, random
import numpy as np
import scipy as sp

t_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 12, 13] # len = 12
data = np.genfromtxt('./train.csv', delimiter=',', dtype=None, skip_header=1) # (4320, 27)
row, col = np.shape(data)
data_format = [[] for i in range(18)]
for i in range(row):
    if i%18 in t_list:
        for val in data[i][3:]: data_format[i%18].append(float(val))
data = data_format


# ### 2. Setup Environment Parameters
num_type = len(data)
hr_len = 7
RMSE_BEST = 100
W_BEST = None
B_BEST = None


# ### 3. Generate All Samples
X_ALL = []
Y_ALL = []
for month in range(12):
    for hour_start in range(471):
        x = []
        for type_id in t_list:
            for hr in range(hour_start+9-hr_len, hour_start+9):
                x.append(data[type_id][month*480+hr])
        Y_ALL.append(data[9][month*480+hour_start+9])
        X_ALL.append(x)

# shuffle all the samples
XY_ALL = zip(X_ALL, Y_ALL)
np.random.shuffle(XY_ALL)
X_ALL, Y_ALL = zip(*XY_ALL)
X_ALL = np.array(X_ALL)
Y_ALL = np.array(Y_ALL)


# ### 4. Create Training & Validation Datasets + Calculate w & b + Validate the RMSE
X_TRAIN = X_ALL
Y_TRAIN = Y_ALL
X_VALIDATION = X_ALL
Y_VALIDATION = Y_ALL

W = np.zeros(X_TRAIN.shape[1])
b = 0
SUM_SQDW = np.zeros(X_TRAIN.shape[1])
SUM_SQDB = 0.
ada_alpha = 1.
nor_alpha = 0.000000001
Lambda = 10
adam_alpha = 0.001
beta1 = 0.9
beta2 = 0.999
Wmt = 0
Wvt = 0
Bmt = 0
Bvt = 0
t = 0
eps = 1e-8
epoch = 10000
while RMSE_BEST > 5 and t < 200000:
    t += 1
    WX_TRAIN = np.dot(X_TRAIN, W)
    ERR = Y_TRAIN - (b + WX_TRAIN)
    X_TRAIN_T = X_TRAIN.T
    DW = -2 * np.dot(X_TRAIN_T, ERR)
    DB = -2 * np.sum(ERR)

    # Compute Loss & Print
#     J = np.sum(ERR ** 2)
#     print "Epoch %s | Loss: %.7f" % (i, J)

    # Regularization
#     DW += Lambda * 2 * W

    # Normal
    W = W - nor_alpha * DW # / X_TRAIN.shape[0]
    b = b - nor_alpha * DB # / X_TRAIN.shape[0]

    # Adagrad
#     SUM_SQDW += np.square(DW)
#     SUM_SQDB += np.square(DB)
#     W = W - ada_alpha/np.sqrt(SUM_SQDW) * DW # / X_TRAIN.shape[0]
#     b = b - ada_alpha/np.sqrt(SUM_SQDB) * DB # / X_TRAIN.shape[0]

    # Adamgrad
#     Wmt = beta1 * Wmt + (1-beta1) * DW
#     Wvt = beta2 * Wvt + (1-beta2) * np.square(DW)
#     Wmthat = Wmt / (1-np.power(beta1, t))
#     Wvthat = Wvt / (1-np.power(beta2, t))
#     Bmt = beta1 * Bmt + (1-beta1) * DB
#     Bvt = beta2 * Bvt + (1-beta2) * np.square(DB)
#     Bmthat = Bmt / (1-np.power(beta1, t))
#     Bvthat = Bvt / (1-np.power(beta2, t))
#     W = W - (adam_alpha*Wmthat) / (np.sqrt(Wvthat) + eps)
#     b = b - (adam_alpha*Bmthat) / (np.sqrt(Bvthat) + eps)

WX_VALIDATION = np.dot(X_VALIDATION, W)
SUMSQERR = np.sum((Y_VALIDATION - (b + WX_VALIDATION)) ** 2)
RMSE_VALIDATION = np.sqrt(SUMSQERR/X_VALIDATION.shape[0])
if RMSE_VALIDATION < RMSE_BEST:
    RMSE_BEST = RMSE_VALIDATION
    W_BEST = W
    B_BEST = b
#     print 'RMSE: %.7lf' % (RMSE_VALIDATION)

# print 'Best RMSE: %.7lf' % RMSE_BEST
# print 'Best B: %.7lf' % B_BEST
# print "Best W:\n%s" % W_BEST
# Best RMSE: 5.6867897


# ### 5. Read Testing Data
test = np.genfromtxt('./test_X.csv', delimiter=',', dtype=None) # (4320, 11)
row, col = np.shape(test)
num_test = row / 18
test_format = [[] for i in range(18)]
for i in range(row):
    if i%18 in t_list:
        for j in range(2+9-hr_len, col):
            test_format[i%18].append(float(test[i][j]))
test = test_format


# ### 6. Create Testing Dataset
X_TEST = []
for i in range(num_test):
    x = []
    for type_id in t_list:
        for hr in range(hr_len):
            x.append(test[type_id][i*hr_len+hr])
    X_TEST.append(x)
X_TEST = np.array(X_TEST)


# ### 7. Calculate for Predicted PM2.5 & Generate CSV output
WX_TEST = np.dot(X_TEST, W_BEST)
Y_TEST = WX_TEST + B_BEST
with open('./kaggle_best.csv', 'wb') as file:
    writer = csv.writer(file, delimiter=',')
    writer.writerow(('id', 'value'))
    for i in range(num_test): writer.writerow(('id_{0}'.format(i), Y_TEST[i]))

