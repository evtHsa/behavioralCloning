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

from ImgViewer import ImgViewer
from DataSet import DataSet
from DataSet import BatchGenerator #fixme: probalby should not be exposed?
from util import brk
from util import traceback_exception
import numpy as np
from model_gen0 import get_model
from keras.callbacks import ModelCheckpoint, Callback
import parm_dict as pd


def view_stuff(X,y, vwr):
    for i in range(len(X)):
        X[i].title += '(' + str(y[i]) + ')'
        vwr.push(X[i], str(y[i]))
    vwr.show()
    vwr.flush()

_vwr = ImgViewer(w=4, h=4, rows=2, cols=2, title="demo")

ds = DataSet("data/driving_log.csv")

# basic model stuff
model = get_model(pd.model_input_sz['rows'], pd.model_input_sz['cols'])

#checkpointing
#https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/ModelCheckpoint
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
