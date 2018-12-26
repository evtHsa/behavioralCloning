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
from dataset import BatchGenerator
from util import brk
import numpy as np

def view_stuff(X,y, vwr):
    for i in range(len(X)):
        X[i].title += '(' + str(y[i]) + ')'
        vwr.push(X[i], str(y[i]))
    vwr.show()
    vwr.flush()

_vwr = ImgViewer(w=4, h=4, rows=2, cols=2, title="demo",disable_ticks=False)

ds = DataSet("data/driving_log.csv")

print("FIXME: show random sample disabled")
#ds.show_random_img_sample()

gen = BatchGenerator(2, 'train', ds)
for i in range(10):
    (X, y) = next(gen)
    for x in X:
        _vwr.push(x)
    _vwr.show()
    _vwr.flush()
