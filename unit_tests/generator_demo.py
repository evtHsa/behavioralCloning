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
import numpy as np

ds = DataSet("data/driving_log.csv")

for i in range(10):
    #for (X, y) in next(gen):
    X_lst, y_lst = next(ds.gen_train)
    for i in range(len(X_lst)):
        X_ndarray = X_lst[i]
        assert(type(X_ndarray) == np.ndarray)
        X_img = Image(X_ndarray, title = "(" + str(y_lst[i]) + ")",
                      img_type='yuv') #see assert in get_img() this  is what get_img emits
        ds._img_viewer.push(X_img)
    ds._img_viewer.show()
    ds._img_viewer.flush()
