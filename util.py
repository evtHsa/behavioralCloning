

import pdb
import csv
import parm_dict as pd

def brk(msg=""):
        print("\n==========\n" +msg + "\n==========\n")
        pdb.set_trace()

def _assert(cond):
        if not cond:
                really_assert = True
                print("assertion failed")

                brk()
                if pd.debug_on_assert:
                        pdb.set_trace()
                _quit()
                
