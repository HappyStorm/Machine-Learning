# # coding: utf-8
import cPickle, csv, json, keras, matplotlib, sys
matplotlib.use('Agg')
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from scipy.misc import toimage
from keras.preprocessing.image import ImageDataGenerator
from keras import regularizers
from keras.models import Model, Sequential, load_model
from keras.layers import Input, Dense, Dropout, SpatialDropout2D, Activation, Flatten
from keras.layers import normalization, Convolution2D, MaxPooling2D, AveragePooling2D, UpSampling2D
from keras.layers.advanced_activations import PReLU
from keras.callbacks import TensorBoard
from keras.optimizers import Adam
from keras.utils import np_utils
from keras.datasets import mnist
from sklearn.cluster import KMeans
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)
keras.backend.tensorflow_backend.set_session(session)



def plot(history, model_name=''):
    # summarize history for accuracy
    # plt.plot(history.history['acc'])
    # plt.plot(history.history['val_acc'])
    # plt.title(model_name + ' Model Accuracy')
    # plt.ylabel('accuracy')
    # plt.xlabel('epoch')
    # plt.legend(['train', 'test'], loc='upper left')
    # plt.savefig('img/%s-acc.png' % model_name)
    # plt.cla()
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title(model_name + ' Model Loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig('./img/%s-loss.png' % model_name)
    plt.cla()

def load_data(prefix):
    all_label = cPickle.load(open(prefix + 'all_label.p', 'rb'))
    test_label = cPickle.load(open(prefix + 'test.p', 'rb'))
    un_label = cPickle.load(open(prefix + 'all_unlabel.p', 'rb'))
    return all_label, test_label, un_label

def EncoderCNN(img_channels, img_rows, img_cols, batch_size, nb_epoch, nb_classes, X_train, Y_train, X_valid, Y_valid):
    input_img = Input(shape=(img_channels, img_rows, img_cols))

    x = Convolution2D(32, 3, 3, activation='relu', border_mode='same', dim_ordering='th')(input_img)
    x = MaxPooling2D(pool_size=(2, 2), border_mode='same', dim_ordering='th')(x)

    x = Convolution2D(16, 3, 3, activation='relu', border_mode='same', dim_ordering='th')(x)
    x = MaxPooling2D(pool_size=(2, 2), border_mode='same', dim_ordering='th')(x)

    x = Convolution2D(16, 3, 3, activation='relu', border_mode='same', dim_ordering='th')(x)
    encoded = MaxPooling2D(pool_size=(2, 2), border_mode='same', dim_ordering='th')(x)

    x = Convolution2D(16, 3, 3, activation='relu', border_mode='same', dim_ordering='th')(encoded)
    x = UpSampling2D((2, 2), dim_ordering='th')(x)

    x = Convolution2D(16, 3, 3, activation='relu', border_mode='same', dim_ordering='th')(x)
    x = UpSampling2D((2, 2), dim_ordering='th')(x)

    x = Convolution2D(32, 3, 3, activation='relu', border_mode='same', dim_ordering='th')(x)
    x = UpSampling2D((2, 2), dim_ordering='th')(x)
    decoded = Convolution2D(img_channels, 3, 3, activation='sigmoid', border_mode='same', dim_ordering='th')(x)

    adam = Adam(lr=1e-3, decay=2.5e-6)
    autoencoder = Model(input=input_img, output=decoded)
    autoencoder.compile(optimizer=adam, loss='binary_crossentropy')
    history = autoencoder.fit(X_train, Y_train,
                nb_epoch=nb_epoch,
                batch_size=batch_size,
                shuffle=True,
                validation_data=(X_valid, Y_valid))

    encoder = Model(input=input_img, output=encoded)
    # encoder.compile(optimizer=adam, loss='binary_crossentropy')
    plot(history, 'Auto-Encoder')
    return autoencoder, encoder

def normalize_data(X_normal, X_test, X_unlab):
    label_rgb_mean = np.mean(X_normal, axis=0)
    X_normal = (X_normal - label_rgb_mean) / 255.
    X_test = (X_test - label_rgb_mean) / 255.
    X_unlab = (X_unlab - label_rgb_mean) / 255.
    return X_normal, X_test, X_unlab

def normalize_noise(X_normal, X_test, X_unlab):
    X_normal = X_normal.astype(float) / 255.
    X_test = X_test.astype(float) / 255.
    X_unlab = X_unlab.astype(float) / 255.
    return X_normal, X_test, X_unlab

def split_data(num_valid, nb_classes, X_normal):
    test_per_class = num_valid / nb_classes
    rand_list = np.random.choice(num_img, test_per_class, replace=False)
    X_train, X_valid = [], []
    Y_train_list, Y_valid = [], []
    for i in range(num_class):
        for j in range(num_img):
            X_train.append(X_normal[i*num_img+j])
            Y_train_list.append(i)
            if j in rand_list:
                X_valid.append(X_normal[i*num_img+j])
                Y_valid.append(i)
    X_train, Y_train = np.array(X_train), np_utils.to_categorical(Y_train_list, nb_classes)
    X_valid, Y_valid = np.array(X_valid), np_utils.to_categorical(Y_valid, nb_classes)
    return X_train, Y_train, X_valid, Y_valid, Y_train_list

def add_noise(X_normal, X_train, X_valid, X_test):
    noise_factor = 0.5
    X_normal_noisy = X_normal + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=X_normal.shape)
    X_train_noisy = X_train + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=X_train.shape)
    X_valid_noisy = X_valid + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=X_valid.shape)
    X_test_noisy = X_test + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=X_test.shape)
    X_normal_noisy = np.clip(X_normal_noisy, 0., 1.)
    X_train_noisy = np.clip(X_train_noisy, 0., 1.)
    X_valid_noisy = np.clip(X_valid_noisy, 0., 1.)
    X_test_noisy = np.clip(X_test_noisy, 0., 1.)
    return X_normal_noisy, X_train_noisy, X_valid_noisy, X_test_noisy

def output_model(model, modelname):
    model.save_weights(modelname)
    # model.save(modelname + '.h5')

if __name__ == '__main__':
    all_label, test_label, un_label = load_data(sys.argv[1])
    
    # CIFAR-10 images (in Theano format)
    img_channels, img_rows, img_cols = 3, 32, 32
    num_valid = 500
    batch_size = 128
    nb_classes = 10
    nb_epoch = 100
    num_class, num_img, num_pixel = np.shape(all_label)
    
    X_normal = np.reshape(all_label, (-1, img_channels, img_rows, img_cols))#.astype(float)
    X_test = np.reshape(np.array(test_label['data']), (-1, img_channels, img_rows, img_cols))#.astype(float)
    X_unlab = np.reshape(un_label, (-1, img_channels, img_rows, img_cols))#.astype(float)
    
    # X_normal, X_test, X_unlab = normalize_data(X_normal, X_test, X_unlab)
    X_normal, X_test, X_unlab = normalize_noise(X_normal, X_test, X_unlab)

    X_train, Y_train, X_valid, Y_valid, Y_train_list = split_data(num_valid, nb_classes, X_normal)
    X_unlab = np.concatenate((X_unlab, X_test), axis=0)
    X_normal = np.concatenate((X_normal, X_unlab), axis=0)
    X_train = np.concatenate((X_train, X_test), axis=0)

    X_normal_noise, X_train_noise, X_valid_noise, X_test_noise = add_noise(X_normal, X_train, X_valid, X_test)

    autoencoder, encoder = EncoderCNN(img_channels, img_rows, img_cols, batch_size, nb_epoch, nb_classes, X_normal_noise, X_normal, X_valid_noise, X_valid)
    # autoencoder, encoder = EncoderCNN(img_channels, img_rows, img_cols, batch_size, nb_epoch, nb_classes, X_train, X_train, X_valid, X_valid)
    output_model(encoder, sys.argv[2])