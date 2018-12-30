#!/usr/bin/env python3
#

# tests generator and hence dataset and hence csv_parser

import sys

# unit tests are usually run from their own dir or parent dir
# unit tests care only of standard stuff and parent dir
sys.path.append(".") 
sys.path.append("..") 


# debug stuff
import pdb

# support code

from ImgViewer import ImgViewer
from ImgUtil import Image
from DataSet import DataSet
from util import brk
from keras.models import load_model
import numpy as np
import parm_dict as pd

ds = DataSet("data/driving_log.csv")
vwr = ds._img_viewer

model = load_model('saved_model.h5')

print(model.summary())

X, y = ds.get_synthetic_image_batch(pd.train_batch_size)
_X = [x.img_data for x in X] # model wants ndarray images, we deal with ImgUtil.Image

for i in range(len(_X)):
    #https://stackoverflow.com/questions/43469281/how-to-predict-input-image-\
    #using-trained-model-in-keras
    #
    #https://stackoverflow.com/questions/43017017/keras-model-predict-for-a-single-image
    #
    #https://datascience.stackexchange.com/questions/31167/how-to-predict-an-\
    # image-using-saved-model
    _x = _X[i]
    _x = np.expand_dims(_x, axis=0)
    y_hat = model.predict(_x, batch_size = 1, verbose=0)
    print("%d: y = %s, y_hat = %s, delta = %.2f%%" % (i, str(y[i]), str(y_hat),
                                                   100 * (y[i] - y_hat) / y[i] ))
    #vwr.push(x)
#vwr.show()
