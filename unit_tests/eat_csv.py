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
from csv_parser import CsvParser

csv_parser = CsvParser("data/driving_log.csv")
image_recs = csv_parser.get_image_recs()
print("%d image_recs remaining" % len(image_recs))

