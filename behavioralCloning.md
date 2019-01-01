

# **Behavioral Cloning Project** 
## **Overview** 
The purposes of this project include

 - Utilizing Keras to build a CNN faster and with less code than directly using the TensorFlow Python API directly.
 - Teach a CNN to derive steering corrections from the camera images and steering angles recorded from a human driver.
### Required Project Files
 - **model.py** : train the model
 - **model.h5**: weights & compiled model saved from training phase
 - **drive.py**: the interface to the simulator modified to do the same preprocessing to the image as was done from model.py with the slight change that images from the simulator are RGB and the training data was in BGR format.
 - **video.mp4**: video composed by video.py from frames generated using the model to drive the simulator in autonomous mode.
### Ancillary Project Files
 - **BatchGenerator.py**:  As is my habit I packaged the generator code in class for ease of reuse. This caused a bit of a problem until I found, in https://stackoverflow.com/questions/42983569/how-to-write-a-generator-class, that from a class object I need to use **return** and not **yield**.
 - **CsvParser.py**: used by DataSet, which has a _csv_parser instance variable to read and parse the csv file and do some trivial data conditioning. All of this is done in the instance's __init__() method
 - **DataSet.py**
	 - creates a CsvParser instance
	 - effets the train/test split
	 - creates BatchGenerator instances
 - **parm_dict.py**:  no longer implemented as a dict but the name survives from ealier self driving car projects
 - **unit_tests**/* : tests of key functionality as it was developed

## **Dataset** 
### Collection
When initial efforts to collect sufficient data seemed to be time consuming and probably generating insufficient training data, I ran across information, in the Student Hub, about a large Udacity provided dataset at https://d17h27t6h515a5.cloudfront.net/topher/2016/December/584f6edd_data/data.zip

The data is a large csv file and a collection of images. The CSV fields are

 - center: string referring to an image in images directory
 - left: left camera
 - right: right camera
 - steering: -1 full left, +1 full rght
 - throttle
 - brake
 - speed: I just now realize I don't know if the units for this measurement ore mi/hr or km/hr.

### Conditioning
#### CsvParser

 - discard records where speed < parm_dict.**too_slow** as they contribute little useful info because the car was barely moving. 0.1 was a reasonable value 
 - Correct steering angle in left and right camera records per https://towardsdatascience.com/teaching-cars-to-drive-using-deep-learning-steering-angle-prediction-5773154608f2
 - **from each csv line** with center, left, right, steering, throttle, brake, speed fields, **produce 3 dicts** in the parser's image_recs list resembling { 'cam' :  'left' | 'right' | 'center', 'img" : <corresponding image from csv line>, 'steering': ..., 'trhottle' : ..., 'brake': ..., 'speed': ...}
#### DataSet
In DataSet's **preprocess_img**() method called from **get_img**( which is called from the BatchGenerator class' __next__() method we
 - crop off the hood of the car and the sky with a simple slice operation
 - resize the result to 66x200x3 which is the model from Nvidia (see References) expects
 - convert to YUV as suggested by the Nvidia paper.

#### XXXXX

### Train/Test Split
I used an 80/20 train_test_split which was imported from sklearn
### Validation
I generated previously unseen images by randomly altering the brightness of images from the data set. While I think I should have done this more extensively than shown in unit_tests/**synthetic.py**, it did seem adequate and guided the tuning of the number of epochs of training
# Model
We use the DNN proposed by nvidia in [**nv-model**] without modification. Reports by previous students indicated that many of the optimizations they tried did not producte worthwhile results. I think I would like to revisit this after finishing the last assignment of the Nanodegree Program.

The model looks like (from [**nv-model**])
![Nvidia model](https://github.com/evtHsa/behavioralCloning/blob/master/nv-model.png)
# References

 1. [**nv-model**] "End  to  End  Learning  for  Self-Driving  Cars*, https://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf
 2. [**udacity-dataset**] Udacity provided dataset:  https://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf
 3. [**side-cam-adj**] steering correction for side camera images, https://towardsdatascience.com/teaching-cars-to-drive-using-deep-learning-steering-angle-prediction-5773154608f2

> Written with [StackEdit](https://stackedit.io/).
