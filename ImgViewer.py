#  adapted from advanced lane finding project

import numpy as np
import matplotlib.pyplot as plt
import util as ut
from util import brk
import ImgUtil
    
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
