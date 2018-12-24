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

def view_stuff(X,y, vwr):
    for i in range(len(X)):
        X[i].title += '(' + str(y[i]) + ')'
        vwr.push(X[i], str(y[i]))
    vwr.show()
    vwr.flush()

_vwr = ImgViewer(w=4, h=4, rows=2, cols=2, title="demo")

ds = DataSet("data/driving_log.csv")
gen = Generator(2, 'train', ds)

for i in range(10):
    for (X, y) in next(gen):
        view_stuff(X,y, _vwr)
        if len(X) == 0:
            break
