class Batch:
    def __init__(self, X, Y, W, b):
        self.X = X
        self.Y = Y
        self.W = W
        self.b = b

def get_batches(data, groundtruth, weight, intercept, batch_size):
    batch_list = []
    num_batch = data.shape[0] / batch_size
    for i in range(num_batch):
        X = []
        Y = []
        for j in range(i*batch_size, (i+1)*batch_size):
            X.append(data[j])
            Y.append(groundtruth[j])
        X = np.array(X)
        Y = np.array(Y)
        batch_list.append(Batch(X, Y, weight, intercept))
    return batch_list

def get_gradient_info(batch, Lambda):
    WX = np.dot(batch.X, batch.W)
    ERR = batch.Y - (batch.b + WX) 
    X_T = batch.X.T
    SUM_DW = -2 * np.dot(X_T, ERR)
    SUM_DB = -2 * np.sum(ERR)
    return SUM_DW, SUM_DB

def shuffle_data(X, Y):
    XY = zip(X, Y)
    np.random.shuffle(XY)
    X, Y = zip(*XY)
    X = np.array(X)
    Y = np.array(Y)
    return X, Y