

import pdb
import csv

def parse_csv(csv_path):
    lines = 0
    with open(csv_path) as f:
        for line in list(csv.reader(f, skipinitialspace=True, delimiter=',',
                                    quoting=csv.QUOTE_NONE))[1:]:
            # note slice above where we skip 1st line which specifies fields in next line
            center,left,right,steering,throttle,brake,speed = line
            steering = float(steering)
            throttle = float(throttle)
            brake = float(brake)
            if (throttle < 0.1): weed out the really, really slow parts
                continue
            lines += 1
            assert("find a citation to calculated adj for left and right cams" == None)
            assert("return some sort of array of dicts" == None)
    print('read %d lines' % lines)
