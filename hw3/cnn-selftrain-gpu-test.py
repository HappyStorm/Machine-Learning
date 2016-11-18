# coding: utf-8
import cPickle, csv, json, keras, sys
import numpy as np
import tensorflow as tf
from keras.models import Sequential, load_model
from keras.utils import np_utils
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)
keras.backend.tensorflow_backend.set_session(session)


def load_data(prefix):
    all_label = cPickle.load(open(prefix + 'all_label.p', 'rb'))
    test_label = cPickle.load(open(prefix + 'test.p', 'rb'))
    un_label = cPickle.load(open(prefix + 'all_unlabel.p', 'rb'))
    return all_label, test_label, un_label

def normalize_data(X_normal, X_test, X_unlab):
    label_rgb_mean = np.mean(X_normal, axis=0)
    X_normal = (X_normal - label_rgb_mean) / 255.
    X_test = (X_test - label_rgb_mean) / 255.
    X_unlab = (X_unlab - label_rgb_mean) / 255.
    return X_normal, X_test, X_unlab

def load_trained_model(modelname):
    model = load_model(modelname)
    return model

def predict(X_test, model, predictname):
    classes = model.predict_classes(X_test, batch_size=50)
    with open(predictname, 'wb') as file:
        writer = csv.writer(file, delimiter=',')
        writer.writerow(('ID', 'class'))
        for i in range(len(classes)): writer.writerow((i, classes[i]))

if __name__ == '__main__':
    all_label, test_label, un_label = load_data(sys.argv[1])
    
    # CIFAR-10 images (in Theano format)
    img_channels, img_rows, img_cols = 3, 32, 32

    X_normal = np.reshape(all_label, (-1, img_channels, img_rows, img_cols))
    X_test = np.reshape(np.array(test_label['data']), (-1, 3, 32, 32))
    X_unlab = np.reshape(un_label, (-1, img_channels, img_rows, img_cols))
    X_normal, X_test, X_unlab = normalize_data(X_normal, X_test, X_unlab)

    model = load_trained_model(sys.argv[2])
    predict(X_test, model, sys.argv[3])