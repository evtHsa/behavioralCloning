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
import parm_dict as pd

ds = DataSet("data/driving_log.csv")

cnt = 0
show_img = pd.unit_test_generator_show_imgs
for i in range(ds.gen_train.steps_per_epoch()):
    cnt += 1
    print("batch #%d" % cnt)
    X_lst, y_lst = next(ds.gen_train)
    for i in range(len(X_lst)):
        X_ndarray = X_lst[i]
        assert(type(X_ndarray) == np.ndarray)
        X_img = Image(X_ndarray, title = "(" + str(y_lst[i]) + ")",
                      img_type='yuv') #see assert in get_img() this  is what get_img emits
        if show_img:
            ds._img_viewer.push(X_img)
    if show_img:
        ds._img_viewer.show()
        ds._img_viewer.flush()
