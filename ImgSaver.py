#  adapted from advanced lane finding project

import datetime as dt
import os
import matplotlib.image as mpimg
import os.path
import util as ut
import ImgUtil as iu
import cv2

class ImgSaver:
    # abstracts saving a bunch of files in a timestamp directory
    def __init__(self, d=None):
        if d is None:
            d = ut.get_out_dir()
        d = d + dt.datetime.now().strftime("%a_%m%d%y_%H%M%S") + "/"
        self.outdir = d
        self.ix = 0
        self.enabled = True
        os.mkdir(d)

    def set_enabled(self, enabled):
        self.enabled = enabled
        
    def save(self, img):
        assert(type(img) is iu.Image)
        if not self.enabled:
            return

        img_data = img.img_data

        if img.img_type == 'bgr':
            img_data = cv2.cvtColor(img.img_data, cv2.COLOR_BGR2RGB)
            
        outf = self.outdir + "%03d" % self.ix + "_" + os.path.basename(img.title) + ".png"
        self.ix = self.ix + 1
        mpimg.imsave(outf, img_data, cmap=img.cmap)

