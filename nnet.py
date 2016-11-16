import keras
# from keras.callbacks import TensorBoard
from keras.layers import Input, Convolution2D, MaxPooling2D, UpSampling2D
from keras.layers import merge, SpatialDropout2D, RepeatVector
from keras.models import Model
from keras import regularizers
import numpy as np
from data_gen import get_data 

input_img = Input(shape=(1,540,1))

# Inception model. Stolen from:
# http://dandxy89.github.io/ImageModels/googlenet/
# ...and from:
# http://joelouismarino.github.io/blog_posts/blog_googlenet_keras.html

# L2 regularizer **should** be changed to L1... question is if *all* 
# regularizers should be L1, or if some should be L2 weight decay

def inception_module(x, params, dim_ordering, concat_axis,
                     subsample=(1, 1), activation='relu',
                     border_mode='same', weight_decay=None):

    # https://gist.github.com/nervanazoo/2e5be01095e935e90dd8  #
    # file-googlenet_neon-py

    (branch1, branch2, branch3, branch4) = params

    if weight_decay:
        W_regularizer = regularizers.l2(weight_decay)
        b_regularizer = regularizers.l2(weight_decay)
    else:
        W_regularizer = None
        b_regularizer = None

    pathway1 = Convolution2D(branch1[0], 1, 1,
                             subsample=subsample,
                             activation=activation,
                             border_mode=border_mode,
                             W_regularizer=W_regularizer,
                             b_regularizer=b_regularizer,
                             bias=False,
                             dim_ordering=dim_ordering)(x)

    pathway2 = Convolution2D(branch2[0], 1, 1,
                             subsample=subsample,
                             activation=activation,
                             border_mode=border_mode,
                             W_regularizer=W_regularizer,
                             b_regularizer=b_regularizer,
                             bias=False,
                             dim_ordering=dim_ordering)(x)
    pathway2 = Convolution2D(branch2[1], 3, 1,
                             subsample=subsample,
                             activation=activation,
                             border_mode=border_mode,
                             W_regularizer=W_regularizer,
                             b_regularizer=b_regularizer,
                             bias=False,
                             dim_ordering=dim_ordering)(pathway2)

    pathway3 = Convolution2D(branch3[0], 1, 1,
                             subsample=subsample,
                             activation=activation,
                             border_mode=border_mode,
                             W_regularizer=W_regularizer,
                             b_regularizer=b_regularizer,
                             bias=False,
                             dim_ordering=dim_ordering)(x)
    pathway3 = Convolution2D(branch3[1], 5, 1,
                             subsample=subsample,
                             activation=activation,
                             border_mode=border_mode,
                             W_regularizer=W_regularizer,
                             b_regularizer=b_regularizer,
                             bias=False,
                             dim_ordering=dim_ordering)(pathway3)

    pathway4 = MaxPooling2D(pool_size=(1, 1), dim_ordering=DIM_ORDERING)(x)
    pathway4 = Convolution2D(branch4[0], 1, 1,
                             subsample=subsample,
                             activation=activation,
                             border_mode=border_mode,
                             W_regularizer=W_regularizer,
                             b_regularizer=b_regularizer,
                             bias=False,
                             dim_ordering=dim_ordering)(pathway4)

    return merge([pathway1, pathway2, pathway3, pathway4],
                 mode='concat', concat_axis=concat_axis)

# Main model encoding via inception

x = Convolution2D(16, 3, 1, activation='relu', border_mode='same')(input_img)
x = MaxPooling2D((2,1), border_mode='same')(x)

x = inception_module(x, params=[(64, ), (1, 96 ), (1, 16 ), (32, )],
                     concat_axis=CONCAT_AXIS)
x = MaxPooling2D((2,1), border_mode='same')(x)
x = inception_module(x, params=[(64, ), (1, 96 ), (1, 16 ), (32, )],
                     concat_axis=CONCAT_AXIS)
x = MaxPooling2D((2,1), border_mode='same')(x)
x = inception_module(x, params=[(64, ), (1, 96 ), (1, 16 ), (32, )],
                     concat_axis=CONCAT_AXIS)
encoded = MaxPooling2D((2,1), border_mode='same')(x)

# Main model decoding... Currently not fancy
# TODO: decoding via inception
# note that inverse of 'merge' is 'RepeatVector'
# ...also TODO: Spatial Dropout

x = Convolution2D(8, 3, 1, activation='relu', border_mode='same')(encoded)
x = UpSampling2D((2,1))(x)
x = Convolution2D(8, 3, 1, activation='relu', border_mode='same')(x)
x = UpSampling2D((2,1))(x)
x = Convolution2D(8, 3, 1, activation='relu', border_mode='same')(x)
x = UpSampling2D((2,1))(x)
# Change below to get to 544 samples (and change input to 544)
x = Convolution2D(16, 3, 1, activation='relu')(x)
x = UpSampling2D((2,1))(x)
decoded = Convolution2D(1, 3, 1, activation='relu', border_mode='same')(x)

autoencoder = Model(input_img, decoded)
autoencoder.compile(optimizer='rmsprop', loss='mse')

epochs = 10

# Note: For increased model complexity offered by inception,
# memory becomes an issue... able to train with batch size of 32,
# but run out of memory at batch size of 64 :-( 

for i in range(epochs):
    x, y, idx = get_data(10000)
    xtrain = x[:8000]
    ytrain = y[:8000]
    xval = x[8000:]
    yval = y[8000:]
    x_train = np.reshape(xtrain, (len(xtrain), 1, 540, 1))
    y_train = np.reshape(ytrain, (len(ytrain), 1, 540, 1))
    x_test = np.reshape(xval, (len(xval), 1, 540, 1))
    y_test = np.reshape(yval, (len(yval), 1, 540, 1))
    
    autoencoder.fit(x_train, y_train,
                     nb_epoch=10,
                     batch_size=32,
                     shuffle=False,
                     validation_data=(x_test, y_test))

# decoded_imgs = autoencoder.predict(x_test)

#encoded_imgs = encoder.predict(xtest)
#decoded_imgs = decoder.predict(encoded_imgs)

################ Save the model #####################
"""
# serialize model to JSON
model_json = autoencoder.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
autoencoder.save_weights("model.h5")
print("Saved model to disk")
"""
#####################################################

############### Restore the model ###################
"""
# load json and create model
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model.h5")
print("Loaded model from disk")
"""
#####################################################
