#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 19 10:19:19 2022

@author: mnv
"""
# In[1]
import pickle

title = 't' + ', ' + 'w_a' + ', ' + 'w_e\n'

data1 = str(1.0) + ', ' + str(1.1287879) + ', ' + str(1.383646367738) + '\n'
data2 = str(2.0) + ', ' + str(1.1287879) + ', ' + str(1.383646367738) + '\n'
file = open('data_exp.txt', 'w')
file.write(title)

#pickle.dump(data,file)
file.write(data1)
file.write(data2)

file.close()

# In[2]

from fenics import *

vtkfile_m = File('/media/mnv/A2E41E9EE41E74AF/N_mid_graphs/m.pvd')

m >> vtkfile_m