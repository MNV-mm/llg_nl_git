#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 30 13:11:55 2022

@author: mnv
"""

from fenics import *
import math
L = 4.

mesh = IntervalMesh(100, -L, L)

El = FiniteElement('CG', mesh.ufl_cell(), 2)
V = FunctionSpace(mesh, El)
#V = FunctionSpace(mesh, "CG", 1)

u = Function(V)
v = TestFunction(V)
u0 = Constant(math.pi/4)
u2 = Constant(0*math.pi/2)
Hy = Constant(-0.)
k = Constant(1000)
Ms = Constant(3.46)


def boundary_l(x, on_boundary):
    return on_boundary and x[0] <= -L + 1E-14

def boundary_r(x, on_boundary):
    return on_boundary and x[0] >= L - 1E-14

bc_l = DirichletBC(V, Constant(math.pi/4), boundary_l)
bc_r = DirichletBC(V, Constant(math.pi/4 + math.pi), boundary_r)

#a = -dot(grad(u), grad(v))*dx +(- sin(u-u0)*cos(u-u0) + Hy*Ms/2/k*cos(u)*sin(u2))*v*dx  
a = 0.5*(sin(2*u)*math.cos(2*u0) - math.sin(2*u0)*cos(2*u))*v*dx + dot(grad(u), grad(v))*dx - Hy*Ms/2/k*cos(u)*sin(u2)*v*dx +2*math.pi*Ms*Ms/k*sin(u)*cos(u)*sin(u2)**2*v*dx  
#- 2*math.pi*Ms*Ms/k*sin(u)*cos(u)*sin(u2)**2

Jac = derivative(a,u)
solve(a==0,u,[bc_l, bc_r],J=Jac)

