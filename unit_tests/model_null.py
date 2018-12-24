# just testing the basic infrastructure. as far as you'll get with this is keras
# complaining that the dense layer dimensions are wrong

from keras.layers.core import Dense
from keras.models import Sequential, model_from_json
from keras.optimizers import Adam

null_model = Sequential()
null_model.add(Dense(1, input_shape=(66,200,3)))

#adam optimizer and mean squared error loss fn
null_model.compile(optimizer=Adam(lr=1e-3), loss='mse')
