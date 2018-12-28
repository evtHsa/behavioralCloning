

import pdb
import csv
import parm_dict as pd

from util import brk, _assert, oneShotMsg

from csv_parser import CsvParser
from random import randint
from ImgViewer import ImgViewer
import ImgUtil as iu
from sklearn.model_selection import train_test_split
import numpy as np
import cv2
#https://scikit-learn.org/stable/modules/generated/sklearn.utils.shuffle.html
from sklearn.utils import shuffle

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

    def preprocess_img(self, img):
        assert(type(img) == iu.Image)
        # chop out the sky and car hood
        img_data = img.img_data[30:96,:,:] # sort of a dull axe
        # forcible resize to what nv model expects which is 66x200x3
        img_data = cv2.resize(img_data,(200, 66), interpolation = cv2.INTER_AREA)
        img.img_data = img_data
        # per "End to End Learning for Self-Driving Cars", convert to YUV
        img = iu.cv2CvtColor(img, 'yuv')
        return img
        
    def get_img(self, ix):
        img = self._csv_parser.get_img(ix)
        assert(type(img) == iu.Image)
        oneShotMsg("img shape(from csv) = " + str(img.img_data.shape))
        img = self.preprocess_img(img)
        oneShotMsg("img shape(after processing) = " + str(img.img_data.shape))
        assert(img.img_data.shape == (pd.model_input_sz['rows'],
                                        pd.model_input_sz['cols'],
                                        3))
        assert(img.img_type == 'yuv')
        return img

    def synthesize_img(self, ix):
        img = self.get_img(ix) # img_data already has pre-processing
        # mess with the brightness
        delta = pd.yuv_max_brightness_boost
        img = img.adjust_yuv_brightness(np.random.randint(-1 * delta, delta))
        assert(img.img_type == 'yuv')
        return img
        
    def get_synthetic_image_batch(self, batch_size):
        # only get imgs from training set(dont want 2look@ test data)
        # shuffle the indices
        self.ixes['train'] = shuffle(self.ixes['train'])
        # slice off a batch full
        _ixes = self.ixes['train'][0:batch_size]
        # get the images & labels
        imgs = [self.synthesize_img(ix) for ix in _ixes]
        labels = [self.get_label(ix) for ix in _ixes]
        return np.array(imgs), np.array(labels)

    def get_label(self, ix):
        return  self._csv_parser.get_label(ix)

    def split_train_test(self):
        ixes = range(self._csv_parser.size())
        #the X and y indices are the same since both come from the same rec
        self.ixes['train'], self.ixes['test'], _, _ = \
                train_test_split(ixes, ixes, test_size = pd.test_set_fraction,
                                 random_state = pd.train_test__random_state)

class BatchGenerator:
    #https://stackoverflow.com/questions/42983569/how-to-write-a-generator-class
    def __init__(self, batch_size, set_slct, dataset):
        _assert((set_slct == 'train') or (set_slct == 'test'))
        assert(batch_size > 0)
        assert(type(dataset) == DataSet)
        self.batch_size = batch_size
        self.set_slct = set_slct
        self._dataset = dataset
        self.batch_start = 0

    def __iter__(self):
        return self
    
    def __next__(self):
        # returns tuple(<array of iu.Image>, <steering angle>)
        while True:
            #print("BatchGenerator(%s): ix = %d, bs = %d" % (self.set_slct, self.batch_start, self.batch_size))
            X = []
            y = []
            for ix in self._dataset.ix_range(self.set_slct, self.batch_start,
                                             self.batch_size):
                out_img = self._dataset.get_img(ix).img_data
                X.append(out_img)
                oneShotMsg("WARNING: __next__ client sees " + str(type(out_img)))
                assert(type(out_img) == np.ndarray)
                y.append(self._dataset.get_label(ix))
            self.batch_start += self.batch_size
            if (len(X) == 0):
                raise StopIteration
            return np.array(X), np.array(y)

    def num_samples(self):
        return self._dataset.set_size(self.set_slct)
                
    def samples_per_epoch(self):
        ret = self.num_samples()  // self.batch_size #warning floor division
        brk("boy howdy")
        return ret

        
