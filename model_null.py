
#from keras.layers.core import Dense, Activation, Flatten, Dropout, Lambda
#from keras.layers.convolutional import Convolution2D
#from keras.layers.pooling import MaxPooling2D
#from keras.layers.advanced_activations import ELU
#from keras.regularizers import l2, activity_l2

# FIXME: get rid of the things we don't need for this project

from keras.layers.core import Dense
from keras.models import Sequential, model_from_json
from keras.optimizers import Adam

null_model = Sequential()
null_model.add(Dense(1, input_shape=(66,200,3)))

#adam optimizer and mean squared error loss fn
null_model.compile(optimizer=Adam(lr=1e-3), loss='mse')
