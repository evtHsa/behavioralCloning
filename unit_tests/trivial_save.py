#!/usr/bin/env python3
#

import sys
# unit tests are usually run from their own dir or parent dir
# unit tests care only of standard stuff and parent dir
sys.path.append(".") 
sys.path.append("..") 


# debug stuff
import pdb

# support code
from DataSet import DataSet
from model_gen0 import get_model
from keras.callbacks import ModelCheckpoint, Callback

import parm_dict as pd
pd.FIXME_RECORD_LIMITER = 138
from util import brk

ds = DataSet("data/driving_log.csv")

model = get_model(pd.model_input_sz['rows'], pd.model_input_sz['cols'])
checkpoint = ModelCheckpoint('model{epoch:02d}.h5')

#https://keras.io/models/sequential/
try:
    history = model.fit_generator(
        ds.gen_train,
        steps_per_epoch=ds.gen_train.steps_per_epoch(),
        epochs=pd.num_epochs, 
        verbose=pd.keras_verbosity,
        callbacks=[checkpoint])
except Exception as ex:
    print(ex)
    pdb.post_mortem()
    
print(model.summary())
model.save('saved_model.h5')
