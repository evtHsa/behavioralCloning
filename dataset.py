

import pdb
import csv
import parm_dict as pd
from util import brk
from csv_parser import CsvParser
from random import randint

class DataSet:
    def __init__(self, csv_path):
        self._csv_parser = CsvParser(csv_path)

    def get_random_ixes(self):
        max = self._csv_parser.size()
        L = []
        for i in range(pd.random_img_set_size):
            L.append(randint(0, max))
        return L
    
    def show_random_img_sample(self):
        ixes = self.get_random_ixes()
        brk("needs code")
        brk("change name and operation to show_random_img_samples")
            
    
        
