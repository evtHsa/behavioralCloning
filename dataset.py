

import pdb
import csv
import parm_dict as pd
from util import brk
from util import _assert
from csv_parser import CsvParser
from random import randint
from ImgUtil import imRead
from ImgViewer import ImgViewer
from sklearn.model_selection import train_test_split

class DataSet:
    def __init__(self, csv_path):
        self._csv_parser = CsvParser(csv_path)
        self._img_viewer = ImgViewer(w=4, h=4, rows=2, cols=2, title="demo")
        self.ixes = { 'X' : {'train' : None, 'test' : None},
                           'y' : {'train' : None, 'test' : None}}
        self.split_train_test()
        self.gen_train = Generator(pd.train_batch_size, 'train', self)
        self.gen_test = Generator(pd.test_batch_size, 'test', self)

    def size(self):
        return self._csv_parser.size()

    def ix_range(self, xory, set, lo, hi):
        return self.ixes[xory][set][lo:hi]
    
    def get_random_ixes(self):
        max = self._csv_parser.size()
        L = []
        for i in range(pd.random_img_set_size):
            L.append(randint(0, max))
        return L
    
    def show_random_img_sample(self):
        ixes = self.get_random_ixes()
        for ix in ixes:
            csv_rec = self.parse_csv_rec(ix)
            img_name = "./data/"+csv_rec['features']['img']
            img = imRead(img_name, reader="cv2")
            self._img_viewer.push(img, ("<%d>" % ix) + csv_rec['features']['img'])
        self._img_viewer.show()

    def parse_csv_rec(self, ix):
        rec = self._csv_parser.get_rec(ix)
        ret = { 'features' : { 'img' : rec['img']}, 'label' : rec['steering'] }
        return ret

    def split_train_test(self):
        ixes = range(self._csv_parser.size())
        ixes = ixes[1:10]
        (self.ixes['X']['train'], self.ixes['X']['test'],
         self.ixes['y']['train'], self.ixes['y']['test']) = train_test_split(ixes, ixes,
                                                                             test_size = pd.test_set_fraction,
                                                                             random_state =
                                                                             pd.train_test__random_state)
class Generator:
    def __init__(self, batch_size, set_slct, dataset):
        _assert((set_slct == 'train') or (set_slct == 'test'))
        self.batch_size = batch_size
        self.set_slct = set_slct
        self._dataset = dataset
        self.batch_start = 0

    def start(self):
        brk("now you need to read the images and return images & labels")
        while True:
           #for ix in self._dataset.ix_range(
            print("Generator(%s): ix = %d, bs = %d" % (self.set_slct,
                                                       self.batch_start, self.batch_size))
            self.batch_start += self.batch_size
            yield(self.batch_start - self.batch_size)
    
        
