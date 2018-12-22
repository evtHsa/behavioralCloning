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
from dataset import DataSet
from dataset import Generator #fixme: probalby should not be exposed?
from util import brk

ds = DataSet("data/driving_log.csv")
#FIXME:ds.show_random_img_sample()

_gen = Generator(2, 'train', ds)
gen = _gen.start()

for i in range(10):
    next(gen)
    
