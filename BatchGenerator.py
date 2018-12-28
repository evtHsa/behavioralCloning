#
import pdb
import parm_dict as pd
from util import brk, _assert, oneShotMsg
import numpy as np

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
        print("\tsamples per epoch = ", self.samples_per_epoch())
        assert(self.samples_per_epoch() > 0)

    def __iter__(self):
        return self
    
    def __next__(self):
        # returns tuple(<array of iu.Image>, <steering angle>)
        while True:
            print("BatchGenerator(%s): ix = %d, bs = %d" % (self.set_slct, self.batch_start, self.batch_size))
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

        
