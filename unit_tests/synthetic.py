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
from dataset import DataSet
from dataset import BatchGenerator
from util import brk
import numpy as np
import parm_dict as pd

ds = DataSet("data/driving_log.csv")
vwr = ds._img_viewer

X, y = ds.get_synthetic_image_batch(pd.train_batch_size)
for x in X:
    vwr.push(x)
vwr.show()
