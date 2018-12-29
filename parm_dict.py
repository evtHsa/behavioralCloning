
#
# as for all preceding self driving car assignments collect magic numbers and
# hyper parameters in a config file so we can easily change them w/o perturbing
# code
#

too_slow = 0.1
side_cam_angle_correction = 0.25
debug_on_assert = True
random_img_set_size = 4
test_set_fraction = 0.2
train_test__random_state = 42
train_batch_size = 64
valid_batch_size = 64
test_batch_size = 64
image_dir = "./data"
yuv_max_brightness_boost = 24
# model hyperparms
num_epochs = 4
keras_verbosity = 2
model_input_sz  = {'rows': 66, 'cols' : 200}
FIXME_RECORD_LIMITER = 0 # make it non zero to restrict size of data we use
batch__next__debug = False
unit_test_generator_show_imgs = True
