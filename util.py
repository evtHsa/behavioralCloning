

import pdb
import csv
import parm_dict as pd
import traceback
import sys
import os


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

          
def traceback_exception(ex):
        template = '{filename:<23}:{linenum}:{funcname}:\n    {source}'
        print("BAD JU-JU: ", ex)
        print("=======================")
        print('format_exception():')
        exc_type, exc_value, exc_tb = sys.exc_info()
        for tb_info in traceback.extract_tb(exc_tb):
                filename, linenum, funcname, source = tb_info
                if funcname != '<module>':
                        funcname = funcname + '()'
                        print(template.format(
                                filename=os.path.basename(filename),
                                linenum=linenum, source=source,
                                funcname=funcname))

def existsKey(dict, key):
        try:
                val = dict[key]
        except KeyError:
                return False
        return True
                        
one_shot_dict = { 'stooge' : 'curly'}

def oneShotMsg(msg):
        if not existsKey(one_shot_dict, msg):
                print(msg)
        one_shot_dict[msg] = msg
