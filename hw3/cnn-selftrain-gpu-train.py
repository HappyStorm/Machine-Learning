# coding: utf-8
import cPickle, csv, json, keras, matplotlib, sys
matplotlib.use('Agg')
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, SpatialDropout2D, Activation, Flatten
from keras.layers import normalization, Convolution2D, MaxPooling2D, AveragePooling2D
from keras.layers.advanced_activations import PReLU
from keras.optimizers import Adam
from keras.utils import np_utils
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)
keras.backend.tensorflow_backend.set_session(session)


# def plot(history, model_name=''):
#     # summarize history for accuracy
#     plt.plot(history.history['acc'])
#     plt.plot(history.history['val_acc'])
#     plt.title(model_name + ' Model Accuracy')
#     plt.ylabel('accuracy')
#     plt.xlabel('epoch')
#     plt.legend(['train', 'test'], loc='upper left')
#     plt.savefig('./img/%s-acc.png' % model_name)
#     plt.cla()
#     # summarize history for loss
#     plt.plot(history.history['loss'])
#     plt.plot(history.history['val_loss'])
#     plt.title(model_name + ' Model Loss')
#     plt.ylabel('loss')
#     plt.xlabel('epoch')
#     plt.legend(['train', 'test'], loc='upper left')
#     plt.savefig('./img/%s-loss.png' % model_name)
#     plt.cla()

def Generator(img_channels, img_rows, img_cols, batch_size, nb_epoch, nb_classes, X_train, Y_train, X_valid, Y_valid, generator):
    model = Sequential()
    
    model.add(Convolution2D(32, 5, 5, border_mode='same', dim_ordering='th', input_shape=[img_channels, img_rows, img_cols]))
    model.add(PReLU())
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2), dim_ordering='th'))
    model.add(SpatialDropout2D(0.35, dim_ordering='th'))
    model.add(normalization.BatchNormalization(mode=0, axis=1))

    model.add(Convolution2D(32, 5, 5, border_mode='same', dim_ordering='th'))
    model.add(PReLU())
    model.add(AveragePooling2D(pool_size=(3, 3), strides=(2, 2), dim_ordering='th'))
    model.add(Dropout(0.35))
    model.add(normalization.BatchNormalization(mode=0, axis=1))

    model.add(Convolution2D(64, 5, 5, border_mode='same', dim_ordering='th'))
    model.add(PReLU())
    model.add(AveragePooling2D(pool_size=(3, 3), strides=(2, 2), dim_ordering='th'))
    model.add(Dropout(0.35))
    model.add(normalization.BatchNormalization(mode=0, axis=1))

    model.add(Flatten())
    model.add(Dense(64))
    model.add(Activation('tanh'))
    model.add(Dropout(0.35))
    model.add(normalization.BatchNormalization(mode=0, axis=1))
    
    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))

    adam = Adam(lr=1e-3, decay=6e-7)
    model.compile(loss='categorical_crossentropy',
                  optimizer=adam,
                  metrics=['accuracy'])
    history = model.fit_generator(
                generator,
                samples_per_epoch=X_train.shape[0]*2,
                nb_epoch=nb_epoch,
                validation_data=(X_valid, Y_valid))
    return model

def GeneratorRetrain(img_channels, img_rows, img_cols, batch_size, nb_epoch, nb_classes, X_train, Y_train, X_valid, Y_valid, generator):
    model = Sequential()
    
    model.add(Convolution2D(32, 5, 5, border_mode='same', dim_ordering='th', input_shape=[img_channels, img_rows, img_cols]))
    model.add(PReLU())
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2), dim_ordering='th'))
    model.add(SpatialDropout2D(0.30, dim_ordering='th'))
    model.add(normalization.BatchNormalization(mode=0, axis=1))

    model.add(Convolution2D(32, 5, 5, border_mode='same', dim_ordering='th'))
    model.add(PReLU())
    model.add(AveragePooling2D(pool_size=(3, 3), strides=(2, 2), dim_ordering='th'))
    model.add(Dropout(0.30))
    model.add(normalization.BatchNormalization(mode=0, axis=1))

    model.add(Convolution2D(64, 5, 5, border_mode='same', dim_ordering='th'))
    model.add(PReLU())
    model.add(AveragePooling2D(pool_size=(3, 3), strides=(2, 2), dim_ordering='th'))
    model.add(Dropout(0.30))
    model.add(normalization.BatchNormalization(mode=0, axis=1))

    model.add(Flatten())
    model.add(Dense(64))
    model.add(Activation('tanh'))
    model.add(Dropout(0.30))
    model.add(normalization.BatchNormalization(mode=0, axis=1))
    
    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))

    adam = Adam(lr=1e-3, decay=6e-7)
    model.compile(loss='categorical_crossentropy',
                  optimizer=adam,
                  metrics=['accuracy'])
    history = model.fit(X_train, Y_train,
                batch_size=batch_size,
                nb_epoch=nb_epoch,
                shuffle=True,
                validation_data=(X_valid, Y_valid))
    # plot(history, 'Self-Train')
    return model

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

def gen_imgGenerator(X_train, Y_train, batch_size):
    train_datagen = ImageDataGenerator(
        rotation_range=10,
        width_shift_range=0.10,
        height_shift_range=0.10,
        shear_range=0.10,
        zoom_range=0.10,
        horizontal_flip=True,
        fill_mode='nearest',
        dim_ordering='th')
    train_datagen.fit(X_train)
    train_generator = train_datagen.flow(X_train, Y_train, batch_size=batch_size)
    return train_generator

def self_train(X_normal, X_train, X_unlab, Y_train_list, model):
    id_set = [False] * (X_normal.shape[0] + X_unlab.shape[0])
    final_batch_size, final_epoch = 128, 50
    st_epoch = 3
    self_train = 0
    while self_train < 5:
        predictions = model.predict(X_unlab)
        X_new_label = []
        for img_id, proba_vec in enumerate(predictions):
            if id_set[img_id]: continue
            class_id = np.argmax(proba_vec)
            if proba_vec[class_id] >= 0.987:
                id_set[img_id] = True
                X_new_label.append(X_unlab[img_id])
                Y_train_list.append(class_id)
        X_train = np.concatenate((X_train, X_new_label), axis=0)
        Y_train = np_utils.to_categorical(Y_train_list, nb_classes)
        model = GeneratorRetrain(img_channels, img_rows, img_cols, final_batch_size, final_epoch, nb_classes, X_train, Y_train, X_valid, Y_valid, train_generator)
        self_train += 1
    return model

def output_model(model, modelname):
    model.save(modelname)

if __name__ == '__main__':
    all_label, test_label, un_label = load_data(sys.argv[1])
    
    # CIFAR-10 images (in Theano format)
    img_channels, img_rows, img_cols = 3, 32, 32
    batch_size = 100
    nb_classes = 10
    nb_epoch = 1000
    num_valid = 500
    num_class, num_img, num_pixel = np.shape(all_label)
    
    X_normal = np.reshape(all_label, (-1, img_channels, img_rows, img_cols))#.astype(float)
    X_test = np.reshape(np.array(test_label['data']), (-1, img_channels, img_rows, img_cols))#.astype(float)
    X_unlab = np.reshape(un_label, (-1, img_channels, img_rows, img_cols))#.astype(float)
    X_normal, X_test, X_unlab = normalize_data(X_normal, X_test, X_unlab)
    X_unlab = np.concatenate((X_unlab, X_test), axis=0)

    X_train, Y_train, X_valid, Y_valid, Y_train_list = split_data(num_valid, nb_classes, X_normal)
    train_generator = gen_imgGenerator(X_train, Y_train, batch_size)
    model = Generator(img_channels, img_rows, img_cols, batch_size, nb_epoch, nb_classes, X_train, Y_train, X_valid, Y_valid, train_generator)
    model = self_train(X_normal, X_train, X_unlab, Y_train_list, model)    
    output_model(model, sys.argv[2])