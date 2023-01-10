#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  7 22:12:48 2022

@author: mnv
"""

import numpy as np
import math

import ufl
from dolfinx import fem, io, mesh, plot
from ufl import ds, dx, grad, inner, sin, cos, dot, form

from mpi4py import MPI
from petsc4py.PETSc import ScalarType 
from dolfinx.nls.petsc import NewtonSolver

dtype = ScalarType

L = 2.

msh = mesh.create_interval(comm=MPI.COMM_WORLD, nx=100, points = [-L,L])

#Elv = ufl.VectorElement("Lagrange", msh.ufl_cell(), degree = 1, dim = 3)
Els = ufl.FiniteElement("Lagrange", msh.ufl_cell(), degree = 1)
#El = Elv*Els
#El = ufl.MixedElement([Elv,Els])

Vs = fem.FunctionSpace(msh, Els)
#V1 = fem.FunctionSpace(msh, Elv)
Vv =  fem.VectorFunctionSpace(msh, ("Lagrange", 1), dim = 3)

facets_l = mesh.locate_entities_boundary(msh, dim=0,
                                       marker=lambda x: np.isclose(x[0], -L))

facets_r = mesh.locate_entities_boundary(msh, dim=0,
                                       marker=lambda x: np.isclose(x[0], L))

#f1 = fem.Function(V.sub(0),)

dofs_l = fem.locate_dofs_topological(V=Vv, entity_dim=0, entities=facets_l)
dofs_r = fem.locate_dofs_topological(V=Vv, entity_dim=0, entities=facets_r)

# bc_func = ufl.as_vector((0.0, 0.0, 0.0))

# bc_expr = fem.Expression(bc_func, Vv.element.interpolation_points())

# f = fem.Function(Vv)

# f.interpolate(bc_func)

bc_l = fem.dirichletbc(np.array([0.,0.,0], dtype=dtype), dofs=dofs_l, V=Vv)
bc_r = fem.dirichletbc(np.array([0.,0.,0], dtype=dtype), dofs=dofs_r, V=Vv)

(m, lamb) = ufl.TrialFunction(Vv), ufl.TrialFunction(Vs)
(v, sig) = ufl.TestFunction(Vv), ufl.TestFunction(Vs)

a = fem.Constant(msh, math.pi/4)
b = fem.Constant(msh, 0*math.pi/4)
k = fem.Constant(msh, 1000.)
Ms = fem.Constant(msh, 3.46)
H = fem.Constant(msh, 0.)

m1, m2, m3 = ufl.split(m)

h_add = ufl.as_vector((0., sin(a)*(sin(a)*m2+cos(b)*m3)-2*math.pi*Ms*Ms/k*m2 + H*Ms/2/k, cos(b)*(sin(a)*m2+cos(b)*m3)))

F = -inner(grad(m),grad(v))*dx + dot(h_add,v)*dx - lamb*dot(m,v)*dx + (m1*m1+m2*m2+m3*m3-1)*sig*dx

#prob = fem.petsc.NonlinearProblem(F, m)

# a = form([[-inner(grad(m),grad(v))*dx, dot(h_add,v)*dx],
#           [(m1*m1+m2*m2+m3*m3-1)*sig*dx, None]])
#np.array([0.,0.,0], dtype=dtype)

# In[system]

import numpy as np
import math

import ufl
from dolfinx import fem, io, mesh, plot
from ufl import ds, dx, grad, inner, sin, cos, dot, form

from dolfinx.nls.petsc import NewtonSolver
from mpi4py import MPI
from petsc4py import PETSc

dtype = PETSc.ScalarType

L = 2.

msh = mesh.create_interval(comm=MPI.COMM_WORLD, nx=100, points = [-L,L])

Elv = ufl.VectorElement("Lagrange", msh.ufl_cell(), degree = 1, dim = 3)
Els = ufl.FiniteElement("Lagrange", msh.ufl_cell(), degree = 1)
El = Elv*Els
#El = ufl.MixedElement([Elv,Els])

V = fem.FunctionSpace(msh, El)
#V1 = fem.FunctionSpace(msh, Elv)
#Vv =  fem.VectorFunctionSpace(msh, ("Lagrange", 1), dim = 3)

facets_l = mesh.locate_entities_boundary(msh, dim=0,
                                       marker=lambda x: np.isclose(x[0], -L))

facets_r = mesh.locate_entities_boundary(msh, dim=0,
                                       marker=lambda x: np.isclose(x[0], L))

V1 = V.sub(0)
V11 = V1.sub(0)
V12 = V1.sub(1)
V13 = V1.sub(2)

dofs_l = fem.locate_dofs_topological(V=V11, entity_dim=0, entities=facets_l)
dofs_r = fem.locate_dofs_topological(V=V11, entity_dim=0, entities=facets_r)

c1 = fem.Constant(msh, 0.)
c2 = fem.Constant(msh, 0.)

bc_l_1 = fem.dirichletbc(c1, dofs = dofs_l, V=V11)
bc_r_1 = fem.dirichletbc(c2, dofs = dofs_r, V=V11)

dofs_l = fem.locate_dofs_topological(V=V12, entity_dim=0, entities=facets_l)
dofs_r = fem.locate_dofs_topological(V=V12, entity_dim=0, entities=facets_r)

c1 = fem.Constant(msh, 0.)
c2 = fem.Constant(msh, 0.)

bc_l_2 = fem.dirichletbc(c1, dofs = dofs_l, V=V12)
bc_r_2 = fem.dirichletbc(c2, dofs = dofs_r, V=V12)

dofs_l = fem.locate_dofs_topological(V=V13, entity_dim=0, entities=facets_l)
dofs_r = fem.locate_dofs_topological(V=V13, entity_dim=0, entities=facets_r)

c1 = fem.Constant(msh, -1.)
c2 = fem.Constant(msh, 1.)

bc_l_3 = fem.dirichletbc(c1, dofs = dofs_l, V=V13)
bc_r_3 = fem.dirichletbc(c2, dofs = dofs_r, V=V13)

a = fem.Constant(msh, 0*math.pi/4)
b = fem.Constant(msh, 0*math.pi/4)
k = fem.Constant(msh, 1000.)
Ms = fem.Constant(msh, 3.46)
H = fem.Constant(msh, 0.)

u = fem.Function(V)
w = ufl.TestFunction(V)

m, lamb = ufl.split(u)
v, sig  = ufl.split(w)

m1, m2, m3 = ufl.split(m)

h_add = ufl.as_vector((0., sin(a)*(sin(a)*m2+cos(b)*m3)-2*math.pi*Ms*Ms/k*m2 + H*Ms/2/k, cos(b)*(sin(a)*m2+cos(b)*m3)))

F = (-inner(grad(m),grad(v))*dx + dot(h_add,v)*dx - lamb*dot(m,v)*dx + (m1*m1+m2*m2+m3*m3-1.0)*sig*dx)

f = fem.Function(V)

problem = fem.petsc.NonlinearProblem(F, u, [bc_l_1, bc_r_1, bc_l_2, bc_r_2, bc_l_3, bc_r_3])
solver = NewtonSolver(MPI.COMM_WORLD, problem)
solver.convergence_criterion = "incremental"
solver.rtol = 1e-10
solver.max_it = 10000
solver.report = True

ksp = solver.krylov_solver
opts = PETSc.Options()
option_prefix = ksp.getOptionsPrefix()
opts[f"{option_prefix}ksp_type"] = "preonly"
opts[f"{option_prefix}pc_type"] = "lu"
opts[f"{option_prefix}pc_factor_mat_solver_type"] = "mumps"
ksp.setFromOptions()

r = solver.solve(u)
# In[parameterized]
import numpy as np
import math

import ufl
from dolfinx import fem, io, mesh, plot
from ufl import ds, dx, grad, inner, sin, cos, dot, form

from mpi4py import MPI
from petsc4py.PETSc import ScalarType 

dtype = ScalarType

L = 2.

msh = mesh.create_interval(comm=MPI.COMM_WORLD, nx=100, points = [-L,L])

Els = ufl.FiniteElement("Lagrange", msh.ufl_cell(), degree = 1)
El = Els*Els
#El = ufl.MixedElement([Elv,Els])

V = fem.FunctionSpace(msh, El)

facets_l = mesh.locate_entities_boundary(msh, dim=0,
                                       marker=lambda x: np.isclose(x[0], -L))

facets_r = mesh.locate_entities_boundary(msh, dim=0,
                                       marker=lambda x: np.isclose(x[0], L))

V1 = V.sub(0)
V2 = V.sub(1)

dofs_l = fem.locate_dofs_topological(V=V1, entity_dim=0, entities=facets_l)
dofs_r = fem.locate_dofs_topological(V=V1, entity_dim=0, entities=facets_r)

c1 = fem.Constant(msh, 0.)
c2 = fem.Constant(msh, math.pi)

bc_l_1 = fem.dirichletbc(c1, dofs = dofs_l, V=V1)
bc_r_1 = fem.dirichletbc(c2, dofs = dofs_r, V=V1)

dofs_l = fem.locate_dofs_topological(V=V2, entity_dim=0, entities=facets_l)
dofs_r = fem.locate_dofs_topological(V=V2, entity_dim=0, entities=facets_r)

c1 = fem.Constant(msh, 0.)
c2 = fem.Constant(msh, 0.)

bc_l_2 = fem.dirichletbc(c1, dofs = dofs_l, V=V2)
bc_r_2 = fem.dirichletbc(c2, dofs = dofs_r, V=V2)