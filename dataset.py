

import pdb
import csv
import parm_dict as pd
from util import brk
from csv_parser import CsvParser
from random import randint
from ImgUtil import imRead
from ImgViewer import ImgViewer
from sklearn.model_selection import train_test_split

class DataSet:
    def __init__(self, csv_path):
        self._csv_parser = CsvParser(csv_path)
        self._img_viewer = ImgViewer(w=4, h=4, rows=2, cols=2, title="demo")
        self.split_train_test()
        self.gen_train = Generator(pd.train_batch_size, self.X_train_ixes, self)
        self.gen_test = Generator(pd.test_batch_size, self.X_test_ixes, self)

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
        self.X_train_ixes, self.X_test_ixes, self.y_train_ixes, self.y_test_ixes = train_test_split(
            ixes, ixes, test_size = pd.test_set_fraction,
            random_state = pd.train_test__random_state)

class Generator:
    def __init__(self,batch_size, ixes, dataset):
        self.batch_size = batch_size
        self.next_ix = batch_size
        self._dataset = dataset
        self.batch_ix = 0

    def start(self):
        brk("now you need to read the images and return images & labels")
        i = 0
        while i < 3:
            print("Generator::init -> %d" % i)
            i += 1
            yield(i - 1)
    
        
