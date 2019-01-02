#!/usr/bin/env python3
#

import sys

# unit tests are usually run from their own dir or parent dir
# unit tests care only of standard stuff and parent dir
sys.path.append(".") 
sys.path.append("unit_tests")  #FIXME: marked for doom


# support code
import pdb
import csv
import cv2
import matplotlib.image as mpimg
import numpy as np
import copy

# the creeping marker of impending monoliticizaton <<<<<<<<<<<<<<<<<<
from keras.callbacks import ModelCheckpoint, Callback
from sklearn.model_selection import train_test_split
import parm_dict as pd #FIXME: this just go away
import keras
from keras.layers.core import Dense
from keras.layers.pooling import MaxPooling2D
from keras.models import Sequential, model_from_json
import keras.models as models
from keras.models import Sequential, Model
from keras.layers.core import Dense,  Activation, Flatten
from keras.layers import BatchNormalization, Input
#from keras.layers.core import Dropout, Reshape
from keras.layers.convolutional import Convolution2D
from keras.optimizers import Adam
#from keras.optimizers import SGD,  RMSprop 

import sklearn.metrics as metrics

#################################################
# constants & hyperparms
#################################################
_too_slow = 0.1
_side_cam_angle_correction = 0.25
_debug_on_assert = True
_random_img_set_size = 4
_test_set_fraction = 0.2
_train_test__random_state = 42
_train_batch_size = 64
_valid_batch_size = 64
#FIXME: delete any of these constants that are unreferenced after all code subsumed
_test_batch_size = 64
_image_dir = "./data"
_yuv_max_brightness_boost = 24
# model hyperparms
_num_epochs = 5
_keras_verbosity = 2
_model_input_sz  = {'rows': 66, 'cols' : 200}
_FIXME_RECORD_LIMITER = 0 # make it non zero to restrict size of data we use
_batch__next__debug = False

#################################################
# support code
#################################################
def brk(msg=""):
        print("\n==========\n" +msg + "\n==========\n")
        pdb.set_trace()

def _assert(cond):
        if not cond:
                really_assert = True
                print("assertion failed")

                brk()
                if _debug_on_assert:
                        pdb.set_trace()
                _quit()

def existsKey(dict, key):
        try:
                val = dict[key]
        except KeyError:
                return False
        return True

one_shot_dict = { 'stooge' : 'curly'}

def oneShotMsg(msg):
        if not existsKey(one_shot_dict, msg):
                print(msg)
        one_shot_dict[msg] = msg

class CsvParser:
    def __init__(self, csv_path):
        self.image_recs = list()
        with open(csv_path) as f:
            for line in list(csv.reader(f, skipinitialspace=True, delimiter=',',
                                        quoting=csv.QUOTE_NONE))[1:]:
                # note slice above where we skip 1st line which specifies fields in next line
                center,left,right,steering,throttle,brake,speed = line
                common_attrs = { 'steering' : float(steering),
                                 'throttle' : float(throttle), 
                                 'brake' : float(brake),
                                 'speed' : float(speed)}
                # the ** notation merges 2 dicts in python > 3.5
                self.image_recs.append({**{'cam' : 'left', 'img' : left}, **common_attrs})
                self.image_recs.append({**{'cam' : 'center', 'img' : center}, **common_attrs})
                self.image_recs.append({**{'cam' : 'right', 'img' : right}, **common_attrs})
            print('read %d images' % len(self.image_recs))
            self.condition_data()
            
    def eliminate_very_slow_data(self):
        # preserves image_recs that are not implausibly slow
        self.image_recs =[ image_rec for image_rec in self.image_recs if image_rec['throttle'] >= pd.too_slow ]
        print("\teliminate_very_slow_data => %d image_recs" % len(self.image_recs))

    def correct_side_cam_steering_angles(self):
        #https://towardsdatascience.com/teaching-cars-to-drive-using-deep-\
        #        learning-steering-angle-prediction-5773154608f2
        print("\tcorrect_side_cam_steering_angles")
        for image_rec in self.image_recs:
            if image_rec['cam'] == 'left':
                image_rec['steering'] += pd.side_cam_angle_correction
            elif  image_rec['cam'] == 'right':
                image_rec['steering'] -= pd.side_cam_angle_correction
            
    def condition_data(self):
        print("condition_data")
        self.eliminate_very_slow_data()
        self.correct_side_cam_steering_angles()

    def get_image_recs(self):
        return self.image_recs
    
    def size(self):
        return len(self.image_recs)

    def get_rec(self, ix):
        return self.image_recs[ix]

    def img_path(self, rel_path):
        return pd.image_dir + '/' + rel_path

    def get_img(self,ix):
        rec = self.get_rec(ix)
        img = imRead(self.img_path(rec['img']), reader="cv2")
        return img
    
    def get_label(self,ix):
        rec = self.get_rec(ix)
        return rec['steering']

###################################################
#
###################################################

#######################################################
# ImgUtil code (reused from Advanced Lane Finding
#######################################################
type_2_cmap = {
     'gray': 'Greys_r',
     'rgb' : None,
     'bgr' : None,
     'yuv' : None
}

color_conversion_dict = {
     'rgb' : {
          'bgr' : cv2.COLOR_RGB2BGR, 'gray' : cv2.COLOR_RGB2GRAY,
          'hls' : cv2.COLOR_RGB2HLS, 'lab' : cv2.COLOR_RGB2Lab,
          'luv' : cv2.COLOR_RGB2Luv, 'yuv' : cv2.COLOR_RGB2YUV},
     'bgr' : {
          'rgb' : cv2.COLOR_BGR2RGB, 'gray' : cv2.COLOR_BGR2GRAY,
          'hls' : cv2.COLOR_BGR2HLS, 'lab' : cv2.COLOR_BGR2Lab,
          'luv' : cv2.COLOR_BGR2Luv, 'yuv' : cv2.COLOR_BGR2YUV},
     'yuv' : { 'rgb' : cv2.COLOR_YUV2RGB, 'bgr' : cv2.COLOR_YUV2BGR}
}

def get_color_conversion_constant(from_type, to_type):
     return color_conversion_dict[from_type][to_type]

class Image:
     def __init__(self, img_data=None, title="", img_type='bgr'):
          self.img_data = img_data
          self.cmap = type_2_cmap[img_type] # fall down if not in dict
          self.img_type = img_type
          self.title = title
          self.msgs = []

     def shape(self):
          return self.img_data.shape
     
     def show(self):
          print("title = %s, img_type = %s" % (self.title, self.img_type))
          
     def is2D(self):
          return len(self.img_data.shape) == 2
          
     def putText(self):
          # https://stackoverflow.com/questions/37191008/load-truetype-font-to-opencv
          # questions/16615662/how-to-write-text-on-a-image-in-windows-using-python-opencv2
          r, g, b = (255, 255, 255) # FIXME: what if img is rbg, enforcement??
          font=cv2.FONT_HERSHEY_DUPLEX #why are hershey the only fonts in opencv
          y = 50
          for msg in self.msgs:
               cv2.putText(self.img_data, msg, (50, y), font, 2, (r, g, b), 2, cv2.LINE_AA)
               y += 50 # I hate guessing about magic #s! ho w tall is the text??
                    
     def plot(self, _plt):
          #self.show()
          tmp = cv2CvtColor(self, 'rgb')
          if len(tmp.img_data.shape) == 1:
               _plt.plot(tmp.img_data) # histogram or other 1D thing
          else:
               _plt.imshow(tmp.img_data, cmap=self.cmap)
          _plt.xlabel(self.title)

     def adjust_yuv_brightness(self, adjustment):
          # docs.scipy.org/doc/numpy/reference/generated/numpy.ndarray.astype.html
          # docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.where.html
          assert(self.img_type == 'yuv')
          if pd.synthetic_debug:
               print("adjust_yuv_brightness ", adjustment)
          if adjustment == 0:
               return self
          if adjustment > 0:
               mask = (self.img_data[:,:,0] + adjustment) > 255
          if adjustment < 0:
               mask = (self.img_data[:,:,0] - adjustment) < 0
          self.img_data[:,:,0] = self.img_data[:,:,0] + np.where(mask, 0, adjustment)
          return self

# signature has changed from the version used with lane finding
def cv2CvtColor(img_obj, to_type, vwr=None):
     assert(type(img_obj) is Image)
     if img_obj.img_type == to_type: # no conversion
          return img_obj 
     color = get_color_conversion_constant(img_obj.img_type, to_type)

     ret = Image(img_data = cv2.cvtColor(img_obj.img_data, color),
                 title = "cvtColor: " + str(color),
                 img_type=to_type)
     return ret

def img_rgb2gray(img, vwr=None):
     assert(type(img) is Image)
     gray = cv2CvtColor(img, cv2.COLOR_RGB2GRAY, vwr)
     return gray

def imRead(path, flags = None, reader=None, vwr=None):
     # note: see cv dox for imread and remember cv2.IMREAD_GRAYSCALE
     assert(reader == 'cv2' or reader == 'mpimg')

     if flags is None:
          flags = cv2.IMREAD_COLOR
     title = reader + ":imread"
     if (reader == 'cv2'):          
          if flags == cv2.IMREAD_GRAYSCALE:
               _type = 'gray'
          else:
               _type = 'bgr'
          img_obj = Image(img_data = cv2.imread(path, flags), title = title, img_type = _type)
     else:
          img_obj = Image(img_data = mpimg.imread(path), title = title, img_type = 'rgb')
     return img_obj

def copy_image(in_img):
     assert(type(in_img) is Image)
     return copy.deepcopy(in_img)

def cv2AddWeighted(src1, src2, alpha=None, beta=None, gamma=None, title=None):
     assert(type(src1) is Image)
     assert(type(src2) is Image)
     assert(src1.shape() == src2.shape())
     assert(not  alpha is None)
     assert(not  beta is None)
     assert(not  gamma is None)
     out_img = Image(img_data = cv2.addWeighted(src1.img_data, alpha,
                                                src2.img_data, beta, gamma),
                     title=title,
                     img_type = src1.img_type)
     return out_img

###################################################
# ImgViewer (adapted from advanced lane finding project)
###################################################
class ImgViewer:
    
    def __init__(self, w=4, h=4, rows=1, cols=1, title = "", svr=None,
                 auto_save=False, disable_ticks=True):
        self.enabled = True
        self.img_obj_list = []
        self.w = w
        self.h = h
        self.title = title
        self.rows = rows
        self.cols = cols
        self.svr = svr
        self.auto_save = auto_save

    def set_enabled(self, enabled):
        self.enabled = enabled
        
    def bce_set_enabled(self, enabled): #"bce" means "everything" in russian
        self.set_enabled(enabled)
        if self.svr:
            self.svr.set_enabled(enabled)
        
    def push(self, img,  debug=False):
        if not self.enabled:
            return
        assert(type(img) is ImgUtil.Image)

        self.img_obj_list.append(img)

        if (debug):
            img.show()
    
        if self.svr and self.auto_save:
            self.svr.save(img)

    def pop(self):
        return self.img_obj_list.pop()

    def flush(self):
        self.img_obj_list = []

    def show_1_grid(self, start):
        L = self.img_obj_list
        n_imgs = len(L)

        plt.figure(figsize=(self.w, self.h))
        for i in range(self.rows * self.cols):
            ix = start + i
            if ix >= n_imgs:
                break
            plt.subplot(self.rows, self.cols, i + 1)
            L[ix].plot(_plt=plt)
            #brk("booger")
            plt.xticks(np.arange(0, L[ix].img_data.shape[1], step=10), [])
            plt.yticks(np.arange(0, L[ix].img_data.shape[0], step=10), [])
        plt.show()
        
    def show(self, clear=False):
        if not self.enabled:
            return
        L = self.img_obj_list
        n_imgs = len(L)
        grid_size = self.rows * self.cols

        for i in range(0, n_imgs, grid_size):
            self.show_1_grid(i)
        if clear:
            self.flush()

    def show_immed(self, img_list, title=""):
        # note: you can use this via pdb: vwr.show_immed(img)
        # ex: vwr.show_immed([in_img, top_down])
        # ex: cache_dict['viewer'].show_immed([hls_binary_l])
        for img in img_list:
            assert(type(img) is ImgUtil.Image)
            plt.figure()
            plt.title(img.title)
            plt.imshow(img.img_data, cmap=img.cmap)
            plt.show()

    def show_immed_ndarray(self, img=None, title=None, img_type=None):
        assert(type(img) is np.ndarray)
        tmp = ImgUtil.Image(img_data = img, title=title, img_type=img_type)
        self.show_immed(tmp, title)
        
def _view(vwr, img, title):
    # turn off viewing by passing None as viewer
    if vwr:
        vwr.show_immed(img, title)
    
def _push(vwr, img_obj):
    assert(type(img_obj) is ImgUtil.Image)
    if vwr:
        vwr.push(img_obj)

def _push_deprecated(vwr, img_obj):
    ut.brk("you didn't mean that")
    raise Exception("I TOLD you that you didn't mean that")


def _flush(vwr):
    if vwr:
        vwr.flush()

def _pop(vwr):
    if vwr:
        vwr.pop()

def _show(vwr, clear=False):
    if vwr:
        vwr.show(clear)

###################################################
# BatchGenerator
###################################################
class BatchGenerator:
    #https://stackoverflow.com/questions/42983569/how-to-write-a-generator-class
    def __init__(self, batch_size, set_slct, dataset):
        _assert((set_slct == 'train') or (set_slct == 'test'))
        assert(batch_size > 0)
        self.batch_size = batch_size
        self.set_slct = set_slct
        self._dataset = dataset
        self.batch_start = 0
        print("BatchGenerator(%d, %s) " % (batch_size, set_slct))
        print("\tnum samples = ", self.num_samples())
        print("\tsteps per epoch = ", self.steps_per_epoch())
        assert(self.steps_per_epoch() > 0)

    def __iter__(self):
        return self
    
    def __next__(self):
        # returns tuple(<array of Image>, <steering angle>)
        while True:
            if _batch__next__debug:
                print("BatchGenerator(%s): ix = %d, bs = %d" % (
                    self.set_slct, self.batch_start, self.batch_size))
            X = []
            y = []
            for ix in self._dataset.ix_range(self.set_slct, self.batch_start,
                                             self.batch_size):
                out_img = self._dataset.get_img(ix).img_data
                X.append(out_img)
                assert(type(out_img) == np.ndarray)
                y.append(self._dataset.get_label(ix))
            self.batch_start += self.batch_size
            if (len(X) == 0):
                self.batch_start = 0
                break
            return np.array(X), np.array(y)

    def num_samples(self):
        return self._dataset.set_size(self.set_slct)
                
    def steps_per_epoch(self):
        ret = self.num_samples()  // self.batch_size #warning floor division
        return ret

class DataSet:
    def __init__(self, csv_path):
        self._csv_parser = CsvParser(csv_path)
        self._img_viewer = ImgViewer(w=8, h=8, rows=8, cols=8, title="demo")
        self.ixes = { 'train' : None, 'test' : None }
        self.split_train_test()
        self.gen_train = BatchGenerator(_train_batch_size, 'train', self)
        self.gen_test = BatchGenerator(_test_batch_size, 'test', self)

    def size(self):
        return self._csv_parser.size()

    def ix_range(self, set, start, len):
        return self.ixes[set][start:start+len]

    def set_size(self, set):
        return len(self.ixes[set])
    
    def get_random_ixes(self):
        max = self._csv_parser.size()
        L = []
        for i in range(_random_img_set_size):
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
        assert(type(img) == Image)
        # chop out the sky and car hood
        img_data = img.img_data[30:96,:,:] # sort of a dull axe
        # forcible resize to what nv model expects which is 66x200x3
        img_data = cv2.resize(img_data,(200, 66), interpolation = cv2.INTER_AREA)
        img.img_data = img_data
        # per "End to End Learning for Self-Driving Cars", convert to YUV
        img = cv2CvtColor(img, 'yuv')
        return img
        
    def get_img(self, ix):
        img = self._csv_parser.get_img(ix)
        assert(type(img) == Image)
        oneShotMsg("img shape(from csv) = " + str(img.img_data.shape))
        img = self.preprocess_img(img)
        oneShotMsg("img shape(after processing) = " + str(img.img_data.shape))
        assert(img.img_data.shape == (_model_input_sz['rows'],
                                        _model_input_sz['cols'],
                                        3))
        assert(img.img_type == 'yuv')
        return img

    def synthesize_img(self, ix):
        img = self.get_img(ix) # img_data already has pre-processing
        # mess with the brightness
        delta = _yuv_max_brightness_boost
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

    def size(self):
        ret = self._csv_parser.size()
        if _FIXME_RECORD_LIMITER:
            ret = _FIXME_RECORD_LIMITER
            msg = "========\nWARNING: restricting to %d records\n========" % ret
            brk(msg)
        return ret

    def split_train_test(self):
        ixes = range(self.size())
        #the X and y indices are the same since both come from the same rec
        self.ixes['train'], self.ixes['test'], _, _ = \
                train_test_split(ixes, ixes, test_size = _test_set_fraction,
                                 random_state = _train_test__random_state)

#####################################################
# model: please see writeup_report.md for detail
#####################################################
model = Sequential()
# keras no longer supports mode=2 for BN????
model.add(BatchNormalization(epsilon=0.001, axis=1,
                             input_shape=(_model_input_sz['rows'],
                                          _model_input_sz['cols'],
                                          3)))
model.add(Convolution2D(24,5,5,border_mode='valid', activation='relu',
                        subsample=(2,2)))
model.add(Convolution2D(36,5,5,border_mode='valid', activation='relu',
                        subsample=(2,2)))
model.add(Convolution2D(48,5,5,border_mode='valid', activation='relu',
                        subsample=(2,2)))
model.add(Convolution2D(64,3,3,border_mode='valid', activation='relu',
                        subsample=(1,1)))
model.add(Convolution2D(64,3,3,border_mode='valid', activation='relu',
                            subsample=(1,1)))
model.add(Flatten())
model.add(Dense(1164, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(1, activation='tanh'))

#adam optimizer and mean squared error loss fn
try:
        model.compile(optimizer=Adam(lr=1e-3), loss='mse')
except Exception:
        pdb.post_mortem()

#####################################################
# begin - main script                                                                                                       #
#####################################################

ds = DataSet("data/driving_log.csv")

#checkpointing
#https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/ModelCheckpoint
checkpoint = ModelCheckpoint('model{epoch:02d}.h5')

#https://keras.io/models/sequential/
try:
    history = model.fit_generator(
        ds.gen_train,
        steps_per_epoch=ds.gen_train.steps_per_epoch(),
        epochs=_num_epochs, 
        verbose=_keras_verbosity,
        callbacks=[checkpoint])
except Exception as ex:
    print(ex)
    pdb.post_mortem()
    
print(model.summary())
