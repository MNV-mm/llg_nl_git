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

El = FiniteElement('CG', mesh.ufl_cell(), 1)
#V = FunctionSpace(mesh, "CG", 1)
P = El*El
V = FunctionSpace(mesh, P)

u = Function(V)
u1, u2 = split(u)
v1, v2 = TestFunctions(V)


u0 = Constant(0*math.pi/4)
Hy = Constant(-0)
k = Constant(1000)
Ms = Constant(3.46)


def boundary_l(x, on_boundary):
    return on_boundary and x[0] <= -L + 1E-14

def boundary_r(x, on_boundary):
    return on_boundary and x[0] >= L - 1E-14



bc_l_1 = DirichletBC(V.sub(0), Constant(0), boundary_l)
bc_l_2 = DirichletBC(V.sub(1), Constant(0*math.pi/2), boundary_l)
bc_r_1 = DirichletBC(V.sub(0), Constant(math.pi), boundary_r)
bc_r_2 = DirichletBC(V.sub(1), Constant(0*math.pi/2), boundary_r)

a =  -dot(grad(u1), grad(v1))*dx +(-sin(u1)*cos(u1)*dot(grad(u2),grad(u2)) - sin(u1-u0)*cos(u1-u0) - 2*math.pi*Ms*Ms/k*sin(u1)*cos(u1)*sin(u2)**2 + Hy*Ms/2/k*cos(u1)*sin(u2))*v1*dx \
    -dot(grad(u2), grad(sin(u1)**2*v2))*dx +(- 2*math.pi*Ms*Ms/k*sin(u1)**2*sin(u2)*cos(u2) + Hy*Ms/2/k*sin(u1)*cos(u2))*v2*dx

Jac = derivative(a,u)
solve(a==0, u, [bc_l_1, bc_r_1, bc_l_2, bc_r_2], J = Jac)

u1, u2 = u.split()
