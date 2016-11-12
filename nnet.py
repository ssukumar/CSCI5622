from keras.callbacks import TensorBoard
from keras.layers import Input, Dense, Convolution2D, MaxPooling2D, UpSampling2D
from keras.models import Model
from keras import regularizers

from data_gen import get_data 

input_img = Input(shape=(540,1))

x = Convolution1D(16, 3, activation='relu', border_mode='same')(input_img)
x = MaxPooling1D( border_mode='same')(x)
x = Convolution1D(8, 3, activation='relu', border_mode='same')(x)
x = MaxPooling1D(border_mode='same')(x)
x = Convolution1D(8, 3, activation='relu', border_mode='same')(x)
encoded = MaxPooling1D(border_mode='same')(x)

x = Convolution1D(8, 3, activation='relu', border_mode='same')(encoded)
x = UpSampling1D()(x)
x = Convolution1D(8, 3, activation='relu', border_mode='same')(x)
x = UpSampling1D()(x)
x = Convolution1D(16, 3, activation='relu')(x)
x = UpSampling1D()(x)
decoded = Convolution1D(1, 3, activation='sigmoid', border_mode='same')(x)

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

x, y, idx = get_data(50000)

xtrain = x[:40000]
ytrain = y[:40000]
xval = x[40000:]
yval = x[40000:]

autoencoder.fit(x_train, x_train,
                nb_epoch=100,
                batch_size=64,
                shuffle=True,
                validation_data=(x_test, x_test),
                callbacks=[TensorBoard(log_dir='/tmp/autoencoder')])


decoded_imgs = autoencoder.predict(x_test)

#encoded_imgs = encoder.predict(xtest)
#decoded_imgs = decoder.predict(encoded_imgs)

