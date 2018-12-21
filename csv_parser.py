

import pdb
import csv
import parm_dict as pd

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
    
        
