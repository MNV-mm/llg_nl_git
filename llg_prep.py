#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 23 17:53:22 2022

@author: mnv
"""
import DD_Hd
import funcs_2
#import dolfin as dp

Lx = 60 # 60 150 80
Ly = 40 # 30 80 40

#hd_ext_expr = funcs_2.n_pair(Ly, 30, 10, 9, 3)
#hd = dp.Expression(hd_ext_expr, degree = 3)

DD_Hd.pe_EF(5,30,1,Lx,Ly,20,'/home/mnv/Documents/python_doc/llg_nl/')