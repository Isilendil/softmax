#!/usr/bin/env python
from os import *

data_list = ['a9a', 'real-sim', 'glass.scale', 'iris.scale', 'leu', 'mnist.scale', 'pendigits', 'usps', 'news20.scale'];

def compare(data):


  # FW
  cmd = "./main -s 18 %s > result/%s_18" %(data, data)
  print cmd
  system(cmd)

  
for data in data_list:
  compare(data)
