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
#ds.show_random_img_sample()

_gen = Generator(2, [1,2,3,4,5], ds)
gen = _gen.start()

while True:
    next(gen)
    
