
from keras.layers.core import Dense
from keras.layers.pooling import MaxPooling2D
from keras.models import Sequential, model_from_json
import keras
import keras.models as models

from keras.models import Sequential, Model
from keras.layers.core import Dense,  Activation, Flatten
from keras.layers import BatchNormalization, Input
#from keras.layers.core import Dropout, Reshape
from keras.layers.convolutional import Convolution2D
from keras.optimizers import Adam
#from keras.optimizers import SGD,  RMSprop 

import sklearn.metrics as metrics

# start with model shown in https://images.nvidia.com/content/tegra/automotive\
#                                       /images/2016/solutions/pdf/end-to-end-dl-using-px.pdf

print("FIXME: verify imput image size")
def get_model(nrows, ncols):
    model = Sequential()
    model.add(BatchNormalization(epsilon=0.001,mode=2, axis=1,
                                 input_shape=(3, nrows,ncols)))
    model.add(Convolution2D(24,5,5,border_mode='valid', activation='relu',
                            subsample=(2,2)))
    model.add(Convolution2D(36,5,5,border_mode='valid', activation='relu',
                            subsample=(2,2)))
    model.add(Convolution2D(48,5,5,border_mode='valid', activation='relu',
                            subsample=(2,2)))
    model.add(Convolution2D(64,3,3,border_mode='valid', activation='relu',
                            subsample=(1,1)))
    model.add(Convolution2D(64,3,3,border_mode='valid', activation='relu',
                            subsample=(1,1)))
    model.add(Flatten())
    model.add(Dense(1164, activation='relu'))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(1, activation='tanh'))
    
    #adam optimizer and mean squared error loss fn
    model.compile(optimizer=Adam(lr=1e-3), loss='mse')
    return model
