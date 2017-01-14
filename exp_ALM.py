#!/usr/bin/env python
from os import *

data_list = ['a9a', 'real-sim', 'glass.scale', 'iris.scale', 'leu', 'mnist.scale', 'pendigits', 'usps', 'news20.scale'];

def compare(data):


  # ALM
  cmd = "./main -s 15 %s > result/%s_15" %(data, data)
  print cmd
  system(cmd)

  '''
  # ADMM2
  cmd = "./main -s x %s > result/%s_x"\ %(data, data)
  print cmd
  system(cmd)

  # ALM_FW
  cmd = "./main -s x %s > result/%s_x"\ %(data, data)
  print cmd
  system(cmd)

  # FW
  cmd = "./main -s x %s > result/%s_x"\ %(data, data)
  print cmd
  system(cmd)

  # BLG_DUAL
  cmd = "./main -s x %s > result/%s_x"\ %(data, data)
  print cmd
  system(cmd)

  # ALM_NEWTON
  cmd = "./main -s x %s > result/%s_x"\ %(data, data)
  print cmd
  system(cmd)
  '''
  
for data in data_list:
  compare(data)
