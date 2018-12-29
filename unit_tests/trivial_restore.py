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
from keras.models import load_model

import parm_dict as pd
pd.FIXME_RECORD_LIMITER = 138
from util import brk

ds = DataSet("data/driving_log.csv")
model = load_model('saved_model.h5')
    
print(model.summary())
