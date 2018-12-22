

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
        X_train, X_test, y_train, y_test = train_test_split(ixes, ixes,
                                                            test_size = pd.test_set_fraction,
                                                            random_state = pd.train_test__random_state)
        brk("booger")
            
    
        
