#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 22 11:28:15 2022

@author: mnv
"""

import sympy as sp
sp.init_printing() 
a, b = sp.symbols('a b')

m1 = sp.Array([[1, 0, 0],
               [0, sp.cos(a), sp.sin(a)],
               [0, -sp.sin(a), sp.cos(a)]])

m2 = sp.Array([[sp.cos(b), 0, sp.sin(b)],
               [0, 1, 0],
               [-sp.sin(b), 0, sp.cos(b)]])

m = sp.tensorcontraction(sp.tensorproduct(m2, m1), (1,2))