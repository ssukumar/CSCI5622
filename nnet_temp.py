from keras.layers import Input, Convolution2D, MaxPooling2D, UpSampling2D
from keras.models import Model
from sklearn.cross_validation import train_test_split
from data_gen import get_data
import numpy as np


def train_and_test(nb_layers=3, nb_filter_layer=[16, 8, 8], nb_row=3, w_regularizer=None, activation='relu', weights=None, optimizer='rmsprop', loss='binary_crossentropy'):
    """
    Method call to train and test the autoencoder

    :param nb_layers: Number of convolution layers
    :param nb_filter_layer: Number of convolution filters to use
    :param nb_row: Number of rows in the convolution kernel.
    :param w_regularizer: instance of WeightRegularizer (eg. L1 or L2 regularization), applied to the main weights matrix
    :param activation: name of activation function to use
    :param weights: list of numpy arrays to set as initial weights
    :param optimizer: str (name of optimizer) or optimizer object
    :param loss: str (name of objective function) or objective function
    :return:
    """

    input_img = Input(shape=(1, 540, 1))
    x = input_img

    for i in range(0, nb_layers):
        x = Convolution2D(nb_filter_layer[i], nb_row, 1, activation=activation, weights=weights, border_mode='same', W_regularizer=w_regularizer)(x)
        x = MaxPooling2D((2, 1), border_mode='same')(x)

    for i in range(nb_layers-1, 0, -1):
        x = Convolution2D(nb_filter_layer[i], nb_row, 1, activation=activation, weights=weights, border_mode='same', W_regularizer=w_regularizer)(x)
        x = UpSampling2D((2, 1))(x)

    x = Convolution2D(nb_filter_layer[0], nb_row, 1, activation='relu')(x)
    x = UpSampling2D((2, 1))(x)
    decoded = Convolution2D(1, nb_row, 1, activation='sigmoid', weights=weights, border_mode='same', W_regularizer=w_regularizer)(x)

    autoencoder = Model(input_img, decoded)
    autoencoder.compile(optimizer=optimizer, loss=loss)

    x, y, idx = get_data(50000)
    xtrain, xval, ytrain, yval = train_test_split(x, y, test_size=0.2, random_state=42)

    x_train = np.reshape(xtrain, (len(xtrain), 1, 540, 1))
    y_train = np.reshape(xtrain, (len(ytrain), 1, 540, 1))
    x_test = np.reshape(xval, (len(xval), 1, 540, 1))
    y_test = np.reshape(yval, (len(yval), 1, 540, 1))

    autoencoder.fit(x_train, y_train,
                    nb_epoch=100,
                    batch_size=64,
                    shuffle=True,
                    validation_data=(x_test, y_test))


if __name__ == "__main__":
    train_and_test()
