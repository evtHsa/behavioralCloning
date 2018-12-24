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
from dataset import DataSet
from dataset import Generator #fixme: probalby should not be exposed?
from util import brk
import numpy as np
from model_null import null_model as model
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

print("FIXME: show random sample disabled")
#ds.show_random_img_sample()

t_gen = Generator(2, 'train', ds)
v_gen = Generator(2, 'train', ds)

# basic model stuff

#checkpointing
#https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/ModelCheckpoint
checkpoint = ModelCheckpoint('model{epoch:02d}.h5')

history = model.fit_generator(t_gen, validation_data=v_gen,
                              nb_val_samples=v_gen.num_samples(), 
                              samples_per_epoch=v_gen.samples_per_epoch(),
                              nb_epoch=pd.num_epochs, verbose=pd.keras_verbosity,
                              callbacks=[checkpoint])
print(model.summary())
brk("wasn'''t that speshul")
