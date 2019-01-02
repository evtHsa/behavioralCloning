

# **Behavioral Cloning Project** 
## **Typographical Convention** 
Bold text in square brackets (ex: [**udacity-dataset**] ) means see the item in the **References** section with that heading.
## **Overview** 
The purposes of this project include

 - Utilizing Keras to build a CNN faster and with less code than directly using the TensorFlow Python API directly.
 - Teach a CNN to derive steering corrections from the camera images and steering angles recorded from a human driver.
### Required Project Files
 - **model.py** : train the model. **Important note**: this file imports the actual model definition from unit_tests/**model_gen0.py** as it was incorrectly anticipated that we would be experimenting with variations on the Nvidia model and other models.
	 - **DataSet class**
		 - creates a CsvParser instance
		 - effects the train/test split
		 - creates BatchGenerator instances
	 -  **BatchGenerator class**:  As is my habit I packaged the generator code in class for ease of reuse. This caused a bit of a problem until I found, in [**generator-class**], that from a class object I need to use **return** and not **yield**.
	 - **CsvParser class**: used by DataSet, which has a _csv_parser instance variable to read and parse the csv file and do some trivial data conditioning. All of this is done in the instance's __init__() method

 - **model.h5**: weights & compiled model saved from training phase
 - **drive.py**: the interface to the simulator modified to do the same preprocessing to the image as was done from model.py with the slight change that images from the simulator are RGB and the training data was in BGR format.
 - **video.mp4**: video composed by video.py from frames generated using the model to drive the simulator in autonomous mode.
 -  **writeup_report.md**: this file
## **Dataset** 
### Collection
When initial efforts to collect sufficient data seemed to be time consuming and probably generating insufficient training data, I ran across information, in the Student Hub, about a large Udacity provided dataset at [**udacity-dataset**]

The data is a large csv file and a collection of images. The CSV fields are

 - center: string referring to an image in images directory
 - left: left camera
 - right: right camera
 - steering: -1 full left, +1 full rght
 - throttle
 - brake
 - speed: I just now realize I don't know if the units for this measurement ore mi/hr or km/hr.

### Conditioning
Initially the dataset has 24,108 entries as defined below ("from each csv line...") and after removing the entries where the car is barely moving, we are left with 21, 774 entries
#### CsvParser

 - discard records where speed < parm_dict.**too_slow** as they contribute little useful info because the car was barely moving. 0.1 was a reasonable value 
 - Correct steering angle in left and right camera records per [**side-cam-adj**]
 - **from each csv line** with center, left, right, steering, throttle, brake, speed fields, **produce 3 dicts** in the parser's image_recs list resembling { 'cam' :  'left' | 'right' | 'center', 'img" : <corresponding image from csv line>, 'steering': ..., 'trhottle' : ..., 'brake': ..., 'speed': ...}
#### DataSet
In DataSet's **preprocess_img**() method called from **get_img**( which is called from the BatchGenerator class' __next__() method we
 - crop off the hood of the car and the sky with a simple slice operation
 - resize the result to 66x200x3 which is the model from Nvidia (see References) expects
 - convert to YUV as suggested by the Nvidia paper.

### Train/Test Split
I used an 80/20 train_test_split which was imported from sklearn
### Validation
I generated previously unseen images by randomly altering the brightness of images from the data set. While I think I should have done this more extensively than shown in unit_tests/**synthetic.py**, it did seem adequate and guided the tuning of the number of epochs of training
### Overfitting

 - One small action to reduce overfitting was removing samples where the car was barely moving
 - The relatively large dataset and small number of epochs did not seem to overfit in that the solution generalized acceptably on track 1

# Model
We use the DNN proposed by nvidia in [**nv-model**] without modification. Reports by previous students indicated that many of the optimizations they tried did not producte worthwhile results. I think I would like to revisit this after finishing the last assignment of the Nanodegree Program.

## Nvidia Model Graphic

The model looks like (from [**nv-model**])
![Nvidia model](https://github.com/evtHsa/behavioralCloning/blob/master/nv-model.png)
## Keras model summary

Layer (type)                 Output Shape              Param #   
batch_normalization_1 (Batch (None, 66, 200, 3)        264       
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 31, 98, 24)        1824      
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 14, 47, 36)        21636     
_________________________________________________________________
conv2d_3 (Conv2D)            (None, 5, 22, 48)         43248     
_________________________________________________________________
conv2d_4 (Conv2D)            (None, 3, 20, 64)         27712     
_________________________________________________________________
conv2d_5 (Conv2D)            (None, 1, 18, 64)         36928     
_________________________________________________________________
flatten_1 (Flatten)          (None, 1152)              0         
_________________________________________________________________
dense_1 (Dense)              (None, 1164)              1342092   
_________________________________________________________________
dense_2 (Dense)              (None, 100)               116500    
_________________________________________________________________
dense_3 (Dense)              (None, 50)                5050      
_________________________________________________________________
dense_4 (Dense)              (None, 10)                510       
_________________________________________________________________
dense_5 (Dense)              (None, 1)                 11       

=================================================================
Total params: 1,595,775
Trainable params: 1,595,643
Non-trainable params: 132

## Training Protocol
The rudimentary protocol used was:
 - optimizer: adam 
 - learning rate: 0.001
 - loss function: mean squared error
 - batch size = 64
 - N =1 
 - repeat
	 - train for N epochs
	 - run unit_test/synthetic.py to compare model predicitons to ground truth(before brightness adjustment)
	 - if accuracy seems reasonable, test the model in the simulator
	 - if the model completes track 1, **stop** (it turns out N = 5)
	 - N += 1
All training was done on CPU (Intel(R) Core(TM) i7-7700HQ CPU @ 2.80GHz) the GPU  enabled workspace was only used to excercise the simulator in autonomous mode. Training for 5 epochs completed in

## Model Performance
### Track 1
The model "won ugly" on track 1 in that it met the specification but was driving slowly (throttle control was not implemented) and a bit erratically. On a real street this would certainly attact attention from law enforcement.

### Track 2
The model failed promptly on track 2 as can be seen in ![enter image description here](https://github.com/evtHsa/behavioralCloning/blob/master/track_2_failure.png)
A few observations and guesses:

 - we successfully avoided the barrier objects that look something like mail boxes or guard rails on track 1 but there were only a few of them and we apparently did not learn that the were impassable since there were enough other cues as to the direction we should take.
 - clearly repeatedly setting up the situation in the simulator were we were driving at such an obstacle and having the human driver avoid it coupled with augmentation techniques to generate more samples from the collected samples would help this
 - This section of road was initially confusing to me where the two sections split and is very reminiscent of an area in Austin, Texas where 4 lanes diverge into two  x two lane sections before where West Ben White Boulevard goes under I-35.

# Further Work

 - Collect simulator data in track 2
 - prune the dataset to reduce the predominance of straight ahead or nearly so samples
 - specifically collect samples of small segments where the human driver avoids errors the model is making
 - test to see if more epochs would further improve accuracy
 - add throttle control so there is more variation in throttle values and that the care tries to negotiate the road as rapidly as possible(isn't that what we all do?)
 - further minimize the risk of overfitting by using image augmentation techniques to enable better generalization. Such techniques might include:
	 - flipping about the vertical axis
	 - blurring
	 - shearing
	 - rotating slightly about the cars centerline which might simulate bumps or pavement wear(ex: where they use truck chains in the Sierras resulting in trenches in the lanes)
	 - reverse the camera calibration techniques used in advanced lane finding to cause camera distortion

# References

 1. [**nv-model**] "End  to  End  Learning  for  Self-Driving  Cars*, https://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf
 2. [**udacity-dataset**] Udacity provided dataset:  https://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf
 3. [**side-cam-adj**] steering correction for side camera images, https://towardsdatascience.com/teaching-cars-to-drive-using-deep-learning-steering-angle-prediction-5773154608f2
 4. [**generator-class**] https://stackoverflow.com/questions/42983569/how-to-write-a-generator-class

> Written with [StackEdit](https://stackedit.io/).
