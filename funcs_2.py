#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  9 20:10:13 2022

@author: mnv
"""
import fenics as fen
import sympy as sp
def n_pair(Ly1,l1,Z1,Z01,n):
    y, z = sp.symbols('x[1] z')
    ly, z1 = sp.symbols('ly z1')
    Ly,l,Z,Z0 = sp.symbols('Ly l Z Z0')
    z_m = z-z1
    z_p = z+z1
    y_m = y-ly
    y_p = y+ly
    
    phi_1 = 2*z_m*(sp.atan(y_m/z_m) - sp.atan(y_p/z_m)) \
        + 2*z_p*(sp.atan(y_p/z_p) - sp.atan(y_m/z_p)) \
             - 2*y_m*sp.atanh(2*z*z1/(y_m**2+z**2+z1**2)) \
                 + 2*y_p*sp.atanh(2*z*z1/(y_p**2+z**2+z1**2)) ## y, z, ly, z1
    #phi_1 = sp.Heaviside(y+ly) - sp.Heaviside(y-ly) #sp.exp(-y**2)
    
    dy = l/2+Ly/4
    phi = -phi_1.subs([(y,y-dy), (ly,(l-Ly/2)/2), (z1,Z)]) + phi_1.subs([(y,y+dy), (ly,(l-Ly/2)/2), (z1,Z)])
    ## y,z,l,Ly,Z
    dy = 3*l/2
    i = 1
    while i < n:
        phi = phi + (-1)**i*(phi_1.subs([(y,y-dy), (ly,l/2), (z1,Z)]) - phi_1.subs([(y,y+dy), (ly,l/2), (z1,Z)]))
        dy = dy+l
        i += 1
    
    #phi_n = phi.subs([(l,l1), (z,Z01), (Z,Z1), (Ly,Ly1)])
    #print(phi_n)
    #sp.plot(phi_n,(y,-2*n*l1,2*n*l1))
    
    hy = -sp.diff(phi,y)#.subs([(l,l1), (z,Z01), (Z,Z1), (Ly,Ly1)])
    hz = -sp.diff(phi,z)#.subs([(l,l1), (z,Z01), (Z,Z1), (Ly,Ly1)])
    #(y,-2*n*l1,2*n*l1)
    p1 = sp.plot(hy.subs([(l,l1), (z,Z01), (Z,Z1), (Ly,Ly1)]),(y,-Ly1/2,Ly1/2), show = False)
    p2 = sp.plot(hz.subs([(l,l1), (z,Z01), (Z,Z1), (Ly,Ly1)]),(y,-Ly1/2,Ly1/2), show = False)
    p1.append(p2[0])
    p1.show()
    
    hy_c = sp.ccode(hy)
    hz_c = sp.ccode(hz)
    # print(hy_c)
    # llog = sp.ln(y)
    # llog = sp.printing.ccode(llog)
    
    out = fen.Expression(('0',hy_c,hz_c), degree = 3, l = l1, Ly = Ly1, Z = Z1, z = Z01)#, degree = 3, z=Z-1
    return out