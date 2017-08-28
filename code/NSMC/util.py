#! /usr/bin/python3.6
# -*- coding: utf-8 -*-
 
import json
from collections import OrderedDict

def write_json(items,filenames):
        with open(filenames,'a+t', encoding='utf8') as outfile:
            json.dump(items,outfile, ensure_ascii=False, indent=4)

def write_txt(contents, filename):
        with open(filename,'a') as f:
                f.write(contents)

def read_txt(filename):
    with open(filename,'r') as f:
        x = f.read()
        return x


def read_json(filenames):
    with open(filenames, 'r', encoding='utf8') as f:
        js = json.loads(f.read(), object_pairs_hook = OrderedDict)
        return js



# func def
#def read_json(fname):
#     with open(fname, 'r') as f:
#         return json.load(f)
# func end

# func def
#def write_json(data, fn):
#    with open(fn, 'w') as f:
#        json.dump(data, f)
        #json.dump(data, f, indent=2)
# func end