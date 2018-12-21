#!/usr/bin/env python3

#  adapted from advanced lane finding project
#!/usr/bin/env python3
#

import sys

# unit tests are usually run from their own dir or parent dir
# unit tests care only of standard stuff and parent dir
sys.path.append(".") 
sys.path.append("..") 


# debug stuff
import pdb

import ImgViewer as iv
import ImgUtil as iu
import pdb

#gViewer = iv.ImgViewer()
gViewer = iv.ImgViewer(w=4, h=4, rows=2, cols=2, title="demo")

for i in range(1, 11):
    fname = "old_test_images/test"+str(i)+".jpg"
    img = iu.imRead(fname, reader="cv2")
    pdb.set_trace
    gViewer.push(img, fname)

gViewer.show(clear=True)
img = iu.imRead("old_test_images/test3.jpg", reader="cv2")
gViewer.show_immed(img1, "booger", None)
gViewer.show()


