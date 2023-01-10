#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  7 20:52:30 2022

@author: mnv
"""

from fenics import *
import math
L = 3

mesh = IntervalMesh(1000, -L, L)

Elv = VectorElement('CG', mesh.ufl_cell(), 1, dim = 3)
Els = FiniteElement('CG', mesh.ufl_cell(), 1)
El = Elv*Els
V = FunctionSpace(mesh, El)
Vv = FunctionSpace(mesh, Elv)
Vs = FunctionSpace(mesh, Els)

func = Function(V)
u, lamb = split(func)
#u, lamb = TrialFunctions(V)
v, sigma = TestFunctions(V)

u0 = Constant(0*math.pi/4)
u2 = Constant(0*math.pi/2)
Hy = Constant(-0.)
k = Constant(1000)
Ms = Constant(3.46)


def boundary_l(x, on_boundary):
    return on_boundary and x[0] <= -L + 1E-14

def boundary_r(x, on_boundary):
    return on_boundary and x[0] >= L - 1E-14

bc_l_m = DirichletBC(V.sub(0), Constant((0,0,1)), boundary_l)
bc_r_m = DirichletBC(V.sub(0), Constant((0,0,-1)), boundary_r)

u1,u2,u3 = split(u)
H_add = as_vector((0, sin(u0)*(sin(u0)*u2 + cos(u0)*u3) - 2*math.pi*Ms**2/k*u2 + Hy*Ms/2/k, cos(u0)*(sin(u0)*u2 + cos(u0)*u3)))

F = -inner(grad(u), grad(v))*dx + dot(H_add,v)*dx - lamb*dot(u,v)*dx + (u1*u1+u2*u2+u3*u3-1)*sigma*dx

#func = Function(V)
#func2 = Function(Vs)
Jac = derivative(F,func)

solve(F==0,func,[bc_l_m, bc_r_m], J = Jac, solver_parameters={"nonlinear_solver": "newton","newton_solver": {"maximum_iterations": 100}})

#solve(F==0,func,[bc_l_m, bc_r_m], J = Jac, solver_parameters={"nonlinear_solver": "snes","snes_solver": {"maximum_iterations": 100}})
