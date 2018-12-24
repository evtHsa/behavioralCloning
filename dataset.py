

import pdb
import csv
import parm_dict as pd

from util import brk, _assert, oneShotMsg

from csv_parser import CsvParser
from random import randint
from ImgViewer import ImgViewer
from sklearn.model_selection import train_test_split
import numpy as np

class DataSet:
    def __init__(self, csv_path):
        self._csv_parser = CsvParser(csv_path)
        self._img_viewer = ImgViewer(w=4, h=4, rows=2, cols=2, title="demo")
        self.ixes = { 'train' : None, 'test' : None }
        self.split_train_test()
        self.gen_train = BatchGenerator(pd.train_batch_size, 'train', self)
        self.gen_valid = BatchGenerator(pd.train_batch_size, 'train', self)
        self.gen_test = BatchGenerator(pd.test_batch_size, 'test', self)

    def size(self):
        return self._csv_parser.size()

    def ix_range(self, set, start, len):
        return self.ixes[set][start:start+len]

    def set_size(self, set):
        return len(self.ixes[set])
    
    def get_random_ixes(self):
        max = self._csv_parser.size()
        L = []
        for i in range(pd.random_img_set_size):
            L.append(randint(0, max))
        return L
    
    def show_random_img_sample(self):
        ixes = self.get_random_ixes()
        for ix in ixes:
            csv_rec = self.get_rec(ix)
            img = self.get_img(ix) # FIXME: slight waste looking up rec 2nd time
            self._img_viewer.push(img, ("<%d>" % ix) + csv_rec['img'])
        self._img_viewer.show()

    def get_rec(self, ix):
        return  self._csv_parser.get_rec(ix)

    def get_exemplar_img_size(self):
        # assume they're all the same size
        img = self._csv_parser.get_img(0)
        return img.img_data.shape
    
    def get_img(self, ix):
        ret = self._csv_parser.get_img(ix)
        oneShotMsg("img shape = " + str(ret.img_data.shape))
        return ret

    def get_label(self, ix):
        return  self._csv_parser.get_label(ix)

    def split_train_test(self):
        ixes = range(self._csv_parser.size())
        ixes = ixes[1:10]
        #the X and y indices are the same since both come from the same rec
        self.ixes['train'], self.ixes['test'], _, _ = \
                train_test_split(ixes, ixes, test_size = pd.test_set_fraction,
                                 random_state = pd.train_test__random_state)

class BatchGenerator:
    #https://stackoverflow.com/questions/42983569/how-to-write-a-generator-class
    def __init__(self, batch_size, set_slct, dataset):
        _assert((set_slct == 'train') or (set_slct == 'test'))
        self.batch_size = batch_size
        self.set_slct = set_slct
        self._dataset = dataset
        self.batch_start = 0

    def __iter__(self):
        return self
    
    def __next__(self):
        while True:
            print("BatchGenerator(%s): ix = %d, bs = %d" % (self.set_slct,
                                                       self.batch_start, self.batch_size))
            X = []
            y = []
            for ix in self._dataset.ix_range(self.set_slct, self.batch_start,
                                             self.batch_size):
                X.append(self._dataset.get_img(ix))
                y.append(self._dataset.get_label(ix))
            self.batch_start += self.batch_size
            ret = (np.array(X), np.array(y))
            return ret

    def num_samples(self):
        return self._dataset.set_size(self.set_slct)
                
    def samples_per_epoch(self):
        return self.num_samples()  // self.batch_size #warning floor division

        
