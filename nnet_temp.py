from keras.layers import Input, Convolution2D, MaxPooling2D, UpSampling2D
from keras.models import Model
from sklearn.cross_validation import train_test_split
from data_gen import get_data
import numpy as np
from sklearn.cross_validation import StratifiedKFold
from sklearn.grid_search import GridSearchCV
from keras.regularizers import l1, l2
from keras.wrappers.scikit_learn import KerasRegressor


def build_nn(nb_layers=3, nb_filter_layer=[16, 8, 8], nb_row=3, w_regularizer=None, activation='relu', w_init=None, optimizer='rmsprop', loss='mse'):
    """
    Method call to train and test the autoencoder

    :param nb_layers: Number of convolution layers
    :param nb_filter_layer: Number of convolution filters to use
    :param nb_row: Number of rows in the convolution kernel.
    :param w_regularizer: instance of WeightRegularizer (eg. L1 or L2 regularization), applied to the main weights matrix
    :param activation: name of activation function to use
    :param w_init: name of initialization function for the weights of the layer
    :param optimizer: str (name of optimizer) or optimizer object
    :param loss: str (name of objective function) or objective function
    :return: the model built
    """

    input_img = Input(shape=(1, 540, 1))
    x = input_img

    for i in range(0, nb_layers):
        x = Convolution2D(nb_filter_layer[i], nb_row, 1, activation=activation, init=w_init, border_mode='same', W_regularizer=w_regularizer)(x)
        x = MaxPooling2D((2, 1), border_mode='same')(x)

    for i in range(nb_layers-1, 0, -1):
        x = Convolution2D(nb_filter_layer[i], nb_row, 1, activation=activation, init=w_init, border_mode='same', W_regularizer=w_regularizer)(x)
        x = UpSampling2D((2, 1))(x)

    x = Convolution2D(nb_filter_layer[0], nb_row, 1, activation=activation)(x)
    x = UpSampling2D((2, 1))(x)
    decoded = Convolution2D(1, nb_row, 1, activation=activation, init=w_init, border_mode='same', W_regularizer=w_regularizer)(x)

    autoencoder = Model(input_img, decoded)
    autoencoder.compile(optimizer=optimizer, loss=loss)
    return autoencoder


def find_best_model(train_data, train_labels, param_grid, folds=10):
    skf = StratifiedKFold(train_labels, n_folds=folds, shuffle=True)
    model = KerasRegressor(build_fn=build_nn)
    regressor = GridSearchCV(model, param_grid, scoring='mse', cv=skf, n_jobs=1)
    reg_results = regressor.fit(train_data, train_labels)

    print("Best AUC score: %f" % reg_results.best_score_)
    print("Best Parameters: %s" % reg_results.best_params_)

    return reg_results.best_estimator_


def train_and_test():

    param_grid = {'activation': ['relu', 'sigmoid'], 'optimizer': ['rmsprop'], 'loss': ['mse'], 'nb_layers': 3,
                  'nb_filter_layer': [[16, 8, 8]], 'nb_row': 3, 'w_init': ['glorot_normal', 'glorot_uniform'],
                  'w_regularizer': [l1(l=0.01), l2(l=0.01)]}
    x, y, idx = get_data(50000)
    xtrain, xval, ytrain, yval = train_test_split(x, y, test_size=0.2, random_state=42)
    x_train = np.reshape(xtrain, (len(xtrain), 1, 540, 1))
    y_train = np.reshape(xtrain, (len(ytrain), 1, 540, 1))
    x_test = np.reshape(xval, (len(xval), 1, 540, 1))
    y_test = np.reshape(yval, (len(yval), 1, 540, 1))

    m = find_best_model(x_train, y_train, param_grid)
    autoencoder = m.model
    autoencoder.fit(x_train, y_train,
                    nb_epoch=100,
                    batch_size=64,
                    shuffle=True,
                    validation_data=(x_test, y_test))


if __name__ == "__main__":
    train_and_test()
