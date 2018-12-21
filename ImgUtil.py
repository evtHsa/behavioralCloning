#  adapted from advanced lane finding project

# functions in this file should be only those that operate on images and/or
# return images

import pdb
import ImgViewer as iv
import cv2
import matplotlib.image as mpimg
import numpy as np
import glob
import os
import util as ut
import parm_dict as pd
import copy

type_2_cmap = {
     'gray': 'Greys_r',
     'rgb' :  None,
     'bgr' : None
}

color_2_src_type = {
     str(cv2.COLOR_RGB2BGR) : 'rgb',
     str(cv2.COLOR_BGR2RGB) : 'bgr',
     str(cv2.COLOR_RGB2GRAY) : 'rgb',
     str(cv2.COLOR_BGR2GRAY) : 'bgr',
     str(cv2.COLOR_BGR2GRAY) : 'bgr',
     str(cv2.COLOR_RGB2HLS)   : 'rgb',
     str(cv2.COLOR_BGR2HLS)   : 'bgr',
     str(cv2.COLOR_BGR2HSV)   : 'bgr',
     str(cv2.COLOR_RGB2Lab)   : 'rgb',
     str(cv2.COLOR_BGR2Lab)    : 'bgr',
     str(cv2.COLOR_RGB2Luv)    : 'rgb',
     str(cv2.COLOR_BGR2Luv)    : 'bgr'
}

colorConversion2DestTypeDict = { # FIXME: this is misguided, misleading, and bad
     str(cv2.COLOR_BGR2RGB) : 'rgb',
     str(cv2.COLOR_RGB2BGR) : 'bgr',
     str(cv2.COLOR_RGB2GRAY) : 'gray',
     str(cv2.COLOR_BGR2GRAY) : 'gray',
     str(cv2.COLOR_RGB2HLS)   : 'gray',
     str(cv2.COLOR_BGR2HLS)   : 'gray',
     str(cv2.COLOR_BGR2HSV)   : 'gray',
     str(cv2.COLOR_RGB2Lab) : 'gray',
     str(cv2.COLOR_BGR2Lab) : 'gray',
     str(cv2.COLOR_RGB2Luv) : 'gray',
     str(cv2.COLOR_BGR2Luv) : 'gray'
}

colorConversion2DestColorName = {
     str(cv2.COLOR_RGB2BGR) : 'bgr',
     str(cv2.COLOR_BGR2RGB) : 'rgb',
     str(cv2.COLOR_RGB2GRAY) : 'gray',
     str(cv2.COLOR_BGR2GRAY) : 'gray',
     str(cv2.COLOR_RGB2HLS)   : 'hls',
     str(cv2.COLOR_BGR2HLS)   : 'hls',
     str(cv2.COLOR_BGR2HSV)   : 'hsv',
     str(cv2.COLOR_RGB2Lab)   : 'lab',
     str(cv2.COLOR_BGR2Lab)   : 'lab',
     str(cv2.COLOR_RGB2Luv)   : 'luv',
     str(cv2.COLOR_BGR2Luv)   : 'luv'
}

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
          rgb = self.img_data
          if self.img_type == 'bgr':
               rgb = cv2.cvtColor(self.img_data, cv2.COLOR_BGR2RGB)
          if len(rgb.shape) == 1:
               _plt.plot(rgb) # histogram or other 1D thing
          else:
               _plt.imshow(rgb, cmap=self.cmap)
          _plt.xlabel(self.title)
          
     def legalColorConversion(self, color):
          # FIXME: just say yes until we have better fix for the fact that
          # cv2.COLOR_RGB2BGR == cv2.COLOR_BGR2RGB
          #return self.img_type == color_2_src_type[str(color)]
          return True
     
def cv2CvtColor(img_obj, color, vwr=None):
     assert(type(img_obj) is Image)
     ut._assert(img_obj.legalColorConversion(color))

     ret = Image(img_data = cv2.cvtColor(img_obj.img_data, color),
                 title = "cvtColor: " + str(color),
                 img_type=getColorConversionDestType(color))
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

def getColorConversionDestType(color):
     return colorConversion2DestTypeDict[str(color)]


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
