#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 28 19:49:33 2022

@author: mnv
"""

from fenics import *
import math

Lx = 60 # 60 150 80
Ly = 40 # 30 80 40

AA = 9.5*10**(-8) #4.3e-6 #2*10**(-8) #(erg/cm) - exchange constant
M_s = 3.46
kk = 1000# erg/cm**3 - unaxial anisotropy constant
theta_0 = 1*math.pi/4

rr0 = 0.00003 # cm - effective electrode radius
dd = math.sqrt(AA/kk)# characteristic domain wall width
beta = math.sqrt(1+2*math.pi*M_s**2/kk)

mesh = Mesh("/home/mnv/Documents/python_doc/llg_nl/MESH.xml")

El = VectorElement('CG', triangle, 1, dim=3)
FS = FunctionSpace(mesh, El)

El_1 = FiniteElement('CG', triangle, 1)
FS_1 = FunctionSpace(mesh, El_1)

u = TrialFunction(FS_1)
v = TestFunction(FS_1)

ub = Expression(("0", "sin(2*atan(exp(x[1]/d))+a)", "cos(2*atan(exp(x[1]/d))+a)"), degree = 4, d=1/beta, a = theta_0)
phi_nl = Expression("4*p*(2*b*atan(exp(x[1]/b))*cos(t0) + x[1]*sin(t0) - b*asinh(1/2*(2*exp(2*x[1]/b) + exp(4*x[1]/b))/(1+exp(2*x[1]/b)))*sin(t0))", degree=4, p = math.pi, b = beta, t0 = theta_0)
phi_0 = project(phi_nl,FS_1)
m_b = project(ub,FS)
m1, m2, m3 = split(m_b)
m_b_2 = as_vector((m1,m2))
n = FacetNormal(mesh)

pi_c = Constant(math.pi)
F = v*dot(4*pi_c*m_b_2,n)*ds - v*4*pi_c*dot(m_b_2,n)*ds - dot(grad(v),4*pi_c*m_b_2 - grad(u))*dx

a = lhs(F)
L = rhs(F)

A = assemble(a)
b = assemble(L)

solver = KrylovSolver('gmres', 'ilu')
#solver.parameters["nonzero_initial_guess"] = True
u = Function(FS_1)
#u.vector()[:] = phi_0.vector()
U = u.vector()
 
solver.solve(A, U, b)
u.vector()[:] = U