#!/usr/bin/env python
from os import *

data_list = ['a9a', 'real-sim', 'glass.scale', 'iris.scale', 'leu', 'mnist.scale', 'pendigits', 'usps', 'news20.scale'];

def compare(data):

  # SGD
  cmd = "./main -s 8 %s > result/%s_8" %(data, data)
  print cmd
  system(cmd)

  # EG
  cmd = "./main -s 9 %s > result/%s_9" %(data, data)
  print cmd
  system(cmd)

  # CD_DUAL
  cmd = "./main -s 10 %s > result/%s_10" %(data, data)
  print cmd
  system(cmd)

  '''
  # ADMM
  cmd = "./main -s x %s > result/%s_x"\ %(data, data)
  print cmd
  system(cmd)

  # ALM
  cmd = "./main -s x %s > result/%s_x"\ %(data, data)
  print cmd
  system(cmd)

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
