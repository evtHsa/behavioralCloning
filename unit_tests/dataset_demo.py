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
from util import brk

ds = DataSet("data/driving_log.csv")
ds.show_random_img_sample()

brk("split into test/train")
