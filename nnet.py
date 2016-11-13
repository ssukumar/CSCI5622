import keras
# from keras.callbacks import TensorBoard
from keras.layers import Input, Convolution2D, MaxPooling2D, UpSampling2D
from keras.models import Model
from keras import regularizers
import numpy as np
from data_gen import get_data 

input_img = Input(shape=(1,540,1))

x = Convolution2D(16, 3, 1, activation='relu', border_mode='same')(input_img)
x = MaxPooling2D((2,1), border_mode='same')(x)
x = Convolution2D(8, 3, 1, activation='relu', border_mode='same')(x)
x = MaxPooling2D((2,1), border_mode='same')(x)
x = Convolution2D(8, 3, 1, activation='relu', border_mode='same')(x)
encoded = MaxPooling2D((2,1), border_mode='same')(x)

x = Convolution2D(8, 3, 1, activation='relu', border_mode='same')(encoded)
x = UpSampling2D((2,1))(x)
x = Convolution2D(8, 3, 1, activation='relu', border_mode='same')(x)
x = UpSampling2D((2,1))(x)
x = Convolution2D(16, 3, 1, activation='relu')(x)
x = UpSampling2D((2,1))(x)
decoded = Convolution2D(1, 3, 1, activation='sigmoid', border_mode='same')(x)

autoencoder = Model(input_img, decoded)
autoencoder.compile(optimizer='rmsprop', loss='binary_crossentropy')

# this model maps an input to its encoded representation
##encoder = Model(input=input_img, output=encoded)
# create a placeholder for an encoded (32-dimensional) input
##encoded_input = Input(shape=(encoding_dim,))
# retrieve the last layer of the autoencoder model
##decoder_layer = autoencoder.layers[-1]
# create the decoder model
##decoder = Model(input=encoded_input, output=decoder_layer(encoded_input))


#autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

epochs = 100;

for i in range(1,epochs):

    x, y, idx = get_data(50000)
    xtrain = x[:40000]
    ytrain = y[:40000]
    xval = x[40000:]
    yval = y[40000:]

    x_train = np.reshape(xtrain, (len(xtrain), 1, 540, 1))
    x_test = np.reshape(xval, (len(xval), 1, 540, 1))

    x_train = np.reshape(xtrain, (len(xtrain), 1, 540, 1))
    y_train = np.reshape(xtrain, (len(ytrain), 1, 540, 1))
    x_test = np.reshape(xval, (len(xval), 1, 540, 1))
    y_test = np.reshape(yval, (len(yval), 1, 540, 1))
    
    autoencoder.train_on_batch(x_train, y_train);
# 
#     autoencoder.fit(x_train, y_train,
#                     nb_epoch=100,
#                     batch_size=64,
#                     shuffle=True,
#                     validation_data=(x_test, y_test))#callbacks=[TensorBoard(log_dir='/tmp/autoencoder')])

    decoded_imgs = autoencoder.predict_on_batch(x_test)

#encoded_imgs = encoder.predict(xtest)
#decoded_imgs = decoder.predict(encoded_imgs)

