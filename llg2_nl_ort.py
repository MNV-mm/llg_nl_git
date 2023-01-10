#!/usr/bin/env python
# coding: utf-8

# In[1]:


from fenics import *
import DD_Hd
import funcs_2
import numpy as np
import math
import sympy as sp
from sympy.printing import print_ccode
#import bempp.api
def norm_sol(m, u, FS):
    #vector().array() is replaced by vector().get_local()
    u_array = u.vector().get_local()
    m_array = m.vector().get_local()
    N = int(np.size(u_array))
    v2d = vertex_to_dof_map(FS)
    d2v = dof_to_vertex_map(FS)
    u_array_2 = u_array[v2d]
    m_array_2 = m_array[v2d]
    mm_array_2 = m_array_2+u_array_2
    i = 0
    while i+2 < N:
        norm = math.sqrt(math.pow(mm_array_2[i],2) + math.pow(mm_array_2[i+1],2) + math.pow(mm_array_2[i+2],2))
        mm_array_2[i] /= norm
        mm_array_2[i+1] /= norm
        mm_array_2[i+2] /= norm
        i += 4
    
    mm_array = mm_array_2[d2v]
    m.vector()[:] = mm_array
    return m

def norm_sol_s(u, FS):
    #vector().array() is replaced by vector().get_local()
    u_array = u.vector().get_local()
    N = int(np.size(u_array))
    v2d = vertex_to_dof_map(FS)
    d2v = dof_to_vertex_map(FS)
    u_array_2 = u_array[v2d]
    i = 0
    while i+2 < N:
        norm = math.sqrt(math.pow(u_array_2[i],2) + math.pow(u_array_2[i+1],2) + math.pow(u_array_2[i+2],2))
        u_array_2[i] /= norm
        u_array_2[i+1] /= norm
        u_array_2[i+2] /= norm
        i += 3
    
    u_array = u_array_2[d2v]
    u.vector()[:] = u_array
    return u

def max_norm(u):
    FS = u.function_space()
    u_array = u.vector().get_local()
    N = int(np.size(u_array))
    v2d = vertex_to_dof_map(FS)
    d2v = dof_to_vertex_map(FS)
    u_array_2 = u_array[v2d]
    i = 0
    norm_prev = 0
    while i+2 < N:
        norm = math.sqrt(math.pow(u_array_2[i],2) + math.pow(u_array_2[i+1],2) + math.pow(u_array_2[i+2],2))
        if norm > norm_prev:
            norm_prev = norm
        i += 3
    return norm_prev

def h_rest(m,p, e_f, dedz, phi, hd_s, kk, kk_o):
    a1 = 1*45/180*math.pi # along -y direction
    a2 = 0*10/180*math.pi
    m1, m2, m3 = split(m)
    e1, e2, e3 = split(e_f)
    dedz_1, dedz_2, dedz_3 = split(dedz)
    ## есть маг поле по y
    vec = as_vector((p*(2*e1*m1.dx(0) + 2*e2*m2.dx(0) + 2*e3*m3.dx(0) + m1*e1.dx(0) + m2*e2.dx(0) + m3*e3.dx(0) + m1*e1.dx(0) + m2*e1.dx(1) + m3*dedz_1) + math.sin(a2)*(math.sin(a2)*m1 + math.sin(a1)*math.cos(a2)*m2 + math.cos(a1)*math.cos(a2)*m3), \
                     p*(2*e1*m1.dx(1) + 2*e2*m2.dx(1) + 2*e3*m3.dx(1) + m1*e1.dx(1) + m2*e2.dx(1) + m3*e3.dx(1) + m1*e2.dx(0) + m2*e2.dx(1) + m3*dedz_2) + math.sin(a1)*math.cos(a2)*(math.sin(a2)*m1 + math.sin(a1)*math.cos(a2)*m2 + math.cos(a1)*math.cos(a2)*m3), \
                          p*(m1*e3.dx(0) + m2*e3.dx(1) + m3*dedz_3 + m1*dedz_1 + m2*dedz_2 + m3*dedz_3) + math.cos(a1)*math.cos(a2)*(math.sin(a2)*m1 + math.sin(a1)*math.cos(a2)*m2 + math.cos(a1)*math.cos(a2)*m3)))
    oo = Constant(0)
    #g_vec = as_vector((grad(dot(m,e_f))[0],grad(dot(m,e_f))[1],oo))
    phi_vec = as_vector((-phi.dx(0), -phi.dx(1), oo))
    return vec + phi_vec #+ hd_s

def hs_rest(m,p,e_f,phi):
    m1, m2, m3 = split(m)
    e1, e2, e3 = split(e_f)
    oo = Constant(0)
    vec = as_vector((oo, oo, m3)) #нет производной по третьей координатe от поля
    #g_vec = as_vector((grad(dot(m,e_f))[0],grad(dot(m,e_f))[1],oo))
    phi_vec = as_vector((-phi.dx(0), -phi.dx(1), oo))
    return vec + phi_vec

def grad3(v):
    oo = Constant(0)
    #vec = as_vector((grad(v)[0],grad(v)[1],oo))
    vec = as_vector((v.dx(0),v.dx(1),oo))
    return vec

def my_Hd_v(phi, m, m_z_bl):
    PI = Constant(math.pi)
    m1, m2, m3 = split(m)
    #oo = Constant(0)
    #vec = as_vector((grad(v)[0],grad(v)[1],oo))
    vec = as_vector((-phi.dx(0), -phi.dx(1), -4*PI*2*(m3-m_z_bl)))
    return vec

def to_2d(v):
    v1, v2, v3 = split(v)
    vec = as_vector((v1,v2))
    return vec

def dot_v(m,mm,w,pp,e_f):
    #m1, m2, m3 = split(m)
    mm1, mm2, mm3 = split(m)
    e1, e2, e3 = split(e_f)
    #w1, w2, w3 = split(w)
    expr = dot(grad(cross(w,m)[0]),grad(mm1) - 2*pp*e1*to_2d(mm)) + \
        dot(grad(cross(w,m)[1]),grad(mm2) - 2*pp*e2*to_2d(mm)) + \
            dot(grad(cross(w,m)[2]),grad(mm3) - 2*pp*e3*to_2d(mm))
    return expr

def dots_v(m,mm,w,pp,e_f):
    #m1, m2, m3 = split(m)
    mm1, mm2, mm3 = split(m)
    e1, e2, e3 = split(e_f)
    #w1, w2, w3 = split(w)
    expr = dot(grad(cross(w,m)[0]),grad(mm1)) + \
        dot(grad(cross(w,m)[1]),grad(mm2)) + \
            dot(grad(cross(w,m)[2]),grad(mm3))
    return expr

def g_c(m,w,i):
    oo = Constant(0)
    expr = as_vector((cross(w,m)[i].dx(0),cross(w,m)[i].dx(1),oo))
    return expr

def mgm(m,pp,e_f,i):
    m1, m2, m3 = split(m)
    e1, e2, e3 = split(e_f)
    mm = [m1,m2,m3]
    E = [e1,e2,e3]
    oo = Constant(0)
    expr = as_vector((mm[i].dx(0),mm[i].dx(1),oo)) +2*pp*E[i]*m
    return expr

def dmdn(m,n):
    m1, m2, m3 = split(m)
    v1 = dot(grad(m1),n)
    v2 = dot(grad(m2),n)
    v3 = dot(grad(m3),n)
    return as_vector((v1,v2,v3))

alpha1 = 1 #0.9 #0.1 #0.0001 
alpha2 = 10   #parameter alpha
UU0 = 0*1.6*10/3 #Voltage (CGS)
AA = 9.5*10**(-8) #4.3e-6 #2*10**(-8) #(erg/cm) - exchange constant
M_s = 3.46
kk = 1000# erg/cm**3 - unaxial anisotropy constant
kk_o = 1054/2 - 2*pi*M_s**2
theta_0 = 0*math.pi/4

rr0 = 0.00003 # cm - effective electrode radius
dd = math.sqrt(AA/kk)# characteristic domain wall width
beta = math.sqrt(1+2*math.pi*M_s**2/kk)
beta_n = math.sqrt(1-(kk_o-2*math.pi*M_s**2)/kk)
dd_n = math.sqrt(AA/(kk+2*math.pi*M_s**2))
g = 10**(-6) # magnetoelectric constant
#p = g*UU0/rr0/(2*math.sqrt(AA*kk))
# p = g*UU0/1e-4/(2*math.sqrt(AA*kk)/6)*0.1
Hy = -30
xx0 = 0
yy0 = 2
#beta = 1.2#parameter beta
#print(parameters.linear_algebra_backend)
#list_linear_solver_methods()

# In[2]:


# Create mesh and define function space
Lx = 60 # 60 150 80
Ly = 40 # 30 80 40
#DD_Hd.pe_EF(5,30,1,Lx,Ly,'/home/mnv/Documents/python_doc/llg_nl/E_series')

#FS_1, FS_3, FS_3_1, FS, e_v = DD_Hd.pe_EF(5,30,1,Lx,Ly)
#mesh = FS.mesh()

mesh = Mesh("/home/llg_nl/MESH.xml")
SL_mesh = RectangleMesh(Point(-Lx/2,-Ly/2),Point(Lx/2,Ly/2),int(2*Lx),int(2*Ly))

z_max = 0.5
p1 = Point(-Lx/2,-Ly/2,-z_max)
p2 = Point(Lx/2,Ly/2,z_max)
nx = 300
ny = 200
mesh_3d = BoxMesh(p1,p2,nx,ny,2)

#SL_space, FS_1, FS_3, FS_3_1, FS

El = VectorElement('CG', triangle, 1, dim=3)
FS = FunctionSpace(mesh, El)

El_1 = FiniteElement('CG', triangle, 1)
FS_1 = FunctionSpace(mesh, El_1)

SL_El = FiniteElement('CG', triangle, 1)
SL_space = FunctionSpace(SL_mesh, SL_El)

El_3 = FiniteElement('CG', tetrahedron, 2)
FS_3 = FunctionSpace(mesh_3d, El_3)

El_3_1 = FiniteElement('CG', tetrahedron, 1)
FS_3_1 = FunctionSpace(mesh_3d, El_3_1)

e_v = Function(FS)
dedz_v = Function(FS)

E_series = TimeSeries('/home/llg_nl/E_mid_20')
dEdz_series = TimeSeries('/home/llg_nl/E_mid_20_dEdz')

E_series.retrieve(e_v.vector(),0)
dEdz_series.retrieve(dedz_v.vector(),0)

E_array = e_v.vector().get_local()
E_max = max_norm(e_v)
dEdz_array = dedz_v.vector().get_local()

e_v.vector()[:] = E_array/E_max
dedz_v.vector()[:] = dEdz_array/E_max

p = g*UU0/1e-4/(2*math.sqrt(AA*kk))*E_max

dedz_1, dedz_2, dedz_3 = split(dedz_v)

#El = VectorElement('CG', triangle, 1, dim=3)
#El1 = FiniteElement('CG', triangle, 1)
#FS = FunctionSpace(mesh, El)
#FS_1 = FunctionSpace(mesh,El1)
# dy = 5
# R0 = 10
# s_s = 0.05 #0.1
# s_L = 0.1 #0.9
# lax = 10 #adsorbtion layer thickness
# lay = 10
# nx = 330 #300
# ny = 150#400
# p1 = Point(-Lx/2,-Ly/2)
# p2 = Point(Lx/2,Ly/2)
# mesh = RectangleMesh(p1,p2,nx,ny)
#mesh = DD_Hd.wall_mesh(Lx,Ly,dy,R0,s_s,s_L)
v = Function(FS) #TrialFunction
w = TestFunction(FS)
#K = FunctionSpace(mesh,El2)
##########################hd_side = DD_Hd.side_pot(FS_1, FS_3_1, FS_3_1, FS, 50, z_max, Lx, Ly, 240)

# In[] # Symbolic expressions
# x, y, z = sp.symbols('x y z')
# xx, yy = sp.symbols('x[0] x[1]')
# x0, y0 = sp.symbols('x0 y0')
# d, r0, U0 = sp.symbols('d r0 U0')

# f_expr = U0*r0/sp.sqrt((r0-z)**2+((x-x0)**2 + (y-y0)**2))
# E1 = -sp.diff(f_expr,x)
# E1 = sp.simplify(E1.subs([(x,d*x),(y,d*y),(z,d*z),(x0,d*x0),(y0,d*y0),(z,0),(x,xx),(y,yy)])/U0*r0)
# E2 = -sp.diff(f_expr,y)
# E2 = sp.simplify(E2.subs([(x,d*x),(y,d*y),(z,d*z),(x0,d*x0),(y0,d*y0),(z,0),(x,xx),(y,yy)])/U0*r0)
# E3 = -sp.diff(f_expr,z)
# E3 = sp.simplify(E3.subs([(x,d*x),(y,d*y),(z,d*z),(x0,d*x0),(y0,d*y0),(z,0),(x,xx),(y,yy)])/U0*r0)
# E1_c=sp.ccode(E1)
# E2_c=sp.ccode(E2)
# E3_c=sp.ccode(E3)
# #print(E3_c)

# pe_p_str = DD_Hd.pe_p(1,5,20,1)
# pe_p_expr = Expression(pe_p_str, degree = 2)
# pe_p= project(pe_p_expr,FS_1)
# vtkfile_pe_p = File('graphs/pe_p.pvd')
# vtkfile_pe_p << pe_p

# u_0 = UU0
# aa = 5
# bb = 4*Ly
# cc = 1
# pe_ef_str = DD_Hd.pe_ef(aa,bb,cc)
# E_pe_expr = Expression((pe_ef_str[0],pe_ef_str[1],pe_ef_str[2]), degree = 4, a = aa, b = bb, c = cc, z = -20)
# #Ex = Expression(pe_ef_str[0],degree = 2, z = -1.1*c)
# E_pe = project(E_pe_expr,FS) #interpolate
# #E_pe_x = project(Ex,FS_1) 
# vtkfile_pe_ef = File('graphs/pe_ef.pvd')
# vtkfile_pe_ef << E_pe

# In[3]:

m_z_bloch = project(Expression(("cos(2*atan(exp(x[1]/d)))"), degree = 4, d=1), FS_1)
wall_type = 'neel'# 'bloch'  'neel'
# Define boundary condition
if wall_type =='neel':
    ub = Expression(("0", "-sin(2*atan(exp(x[1]/d)))", "cos(2*atan(exp(x[1]/d)))"), degree = 4, d=1/beta)
    #ub = Expression(("0", "-sin(2*atan(exp(x[1]/d)))*cos(a) + cos(2*atan(exp(x[1]/d)))*sin(a)", "sin(a)*sin(2*atan(exp(x[1]/d))) + cos(2*atan(exp(x[1]/d)))*cos(a)"), degree = 4, d=1/beta, a = theta_0)
    #ub_n = Expression(("0", "sin(2*atan(exp(x[1]/d))+a)", "cos(2*atan(exp(x[1]/d))+a)"), degree = 4, d=1/beta, a = theta_0)
    #ub = Expression(("0", "sqrt(1-(tanh(x[1]/d)*tanh(x[1]/d)))", "tanh(x[1]/d)", "0"), degree = 4, d=1/beta)
    
if wall_type =='bloch':
    ub = Expression(("-sin(2*atan(exp(x[1]/d)))", "cos(2*atan(exp(x[1]/d)))*sin(a)", "cos(2*atan(exp(x[1]/d)))*cos(a)"), degree = 4, d=1, a = -theta_0)
    #m_bloch = project(Expression(("sin(2*atan(exp(x[1]/d)))", "0", "cos(2*atan(exp(x[1]/d)))"), degree = 4, d=1),FS)
    #ub = Expression(("sin(-2*atan(exp((x[1]-5)/d)) - 2*atan(exp((x[1]+5)/d)))", "0", "cos(2*atan(exp((x[1]-5)/d)) - 2*atan(exp((x[1]+5)/d)))"), degree = 4, d=1)
    #ub = Expression(("sin(3*x[1]/30)", "0", "cos(3*x[1]/30)"), degree = 2)
    #ub = Expression(("sqrt(1-(tanh(x[1]/d)*tanh(x[1]/d)))", "0", "tanh(x[1]/d)"), degree = 5, d=1)
#ub = Expression(("cos(x[1])", "0", "sin(x[1])", "0"), degree = 4)
phi_nl = Expression("4*p*(2*b*atan(exp(x[1]/b))*cos(t0) + x[1]*sin(t0) - b*asinh(1/2*(2*exp(2*x[1]/b) + exp(4*x[1]/b))/(1+exp(2*x[1]/b)))*sin(t0))", degree=4, p = math.pi, b = beta, t0 = theta_0) # Expression("4*p*2*b*atan(exp(x[1]/b))", degree=4, p = math.pi, b = beta)
Hy_expr = Expression("-(5.5 + 0.00000002*(pow(x[1],6) + 300000*pow(x[1],2)))", degree = 4)

# Define electric field
electrode_type = 'plane' # 'plane'
if electrode_type == 'circle':
    e1 = Expression((E1_c),degree = 2, U0 = UU0, d = dd, r0 = rr0, x0 = xx0, y0 = yy0)   
    e2 = Expression((E2_c),degree = 2, U0 = UU0, d = dd, r0 = rr0, x0 = xx0, y0 = yy0)
    e3 = Expression((E3_c),degree = 2, U0 = UU0, d = dd, r0 = rr0, x0 = xx0, y0 = yy0)
    e_v = Expression((E1_c, E2_c, E3_c), degree = 2, U0 = UU0, d = dd, r0 = rr0, x0 = xx0, y0 = yy0)
if electrode_type == 'plane':
    print('plane_electrode')
    #e_v = Expression((pe_ef_str[0],pe_ef_str[1],pe_ef_str[2]), degree = 4, a = a, b = b, c = c, z = -1.1*c)

def boundary(x, on_boundary):
    return on_boundary

def my_boundary(x, on_boundary):
    tol = 1E-16
    return (on_boundary and near(x[1],Ly/2,tol)) or (on_boundary and near(x[1],-Ly/2,tol))

BC = DirichletBC(FS, ub, boundary)
# Define initial value
time_old = TimeSeries('/home/llg_nl/series_old/m')
time_new = TimeSeries('/home/llg_nl/series_new/m')

in_type = 'old'
if in_type == 'old':
    t = 1424
    m = Function(FS)
    time_old.retrieve(m.vector(), t)
if in_type == 'new':
    m = project(ub,FS)
if in_type == 'rand':
    m = project(ub,FS)
    m = DD_Hd.rand_vec(m, 0.001)
    m = norm_sol_s(m,FS)
    
m_b = project(ub,FS)
m = norm_sol_s(m, FS)
m1, m2, m3 = m.split()

h = 0.001 #cm
l = math.sqrt(4*math.sqrt(AA*kk)/(M_s**2)*h)/dd
Z = h/dd/2

idx, space_top, slp_pot, trace_space, trace_matrix = DD_Hd.s_chg_prep(SL_space, FS_1, FS_3_1, FS_3_1, FS, Z)
hd_s = DD_Hd.s_chg(m3, SL_space, FS_1, FS_3_1, FS_3_1, FS, idx, space_top, slp_pot, trace_space, trace_matrix)
# vtkfile_hd_s = File('/home/llg_nl/graphs/hd_s.pvd')
# vtkfile_hd_s << hd_s

hd_ext_expr = funcs_2.n_pair(Ly, l, Z, 0, 4)
hd_ext = project(hd_ext_expr, FS)
vtkfile_Hd_ext = File('/home/llg_nl/graphs/Hd_ext.pvd')
vtkfile_Hd_ext << hd_ext
H_st = project(Expression(('0', '0', '-10/20*x[1]'), degree = 4),FS)
# vtkfile_hd_ext = File('/home/llg_nl/graphs/hd.pvd')
# vtkfile_hd_ext << hd_ext

e_f = e_v # project(e_v,FS)
m1, m2, m3 = split(m)
e1, e2, e3 = split(e_f)#e_f.split()
v1, v2, v3 = split(v) #split(v)
w1, w2, w3 = split(w)
# e3_values = e3.compute_vertex_values()
# m3_values = m3.compute_vertex_values()
al = Constant(alpha1)
# class A(Expression):
#     def set_al_values(self, Lx, Ly, lax, lay, al_0, al_1):
#         self.Lx, self.Ly = Lx, Ly
#         self.lax, self.lay = lax, lay
#         self.al_0, self.al_1 = al_0, al_1
        
#     def eval(self, value, x):
#         "Set value[0] to value at point x"
#         tol = 1E-14
#         if x[0] <= 1: #-Lx/2 + lax + tol:
#             value[0] = self.al_0
#         # elif x[0] >= Lx/2-lax +tol:
#         #     value[0] = self.al_0
#         # elif x[1] <= -Ly/2 + lay +tol:
#         #     value[0] = self.al_0
#         # elif x[1] >= Ly/2 - lay +tol:
#         #     value[0] = self.al_0
#         else:
#             value[0] = self.al_1

# al = A(degree = 0)
# al.set_al_values(Lx, Ly, lax, lay, alpha1, alpha2)
# tol = 1E-14
# al = Expression('(x[0] <= -65 + tol) || (x[0]>=65+tol) || (x[1]<=-30+tol) || (x[1]>=30+tol)? alpha2:alpha1', degree = 0, tol = tol, alpha2 = alpha2, alpha1 = alpha1)
pp = Constant(p)#p
k = Constant(kk)
k_o = Constant(kk_o)
Ms = Constant(M_s)
hy = project(Hy_expr,FS_1)

#u_n = interpolate(ub, V)
#u_n = Function(V)
#u_n1, u_n2, u_n3 = split(u_n)
#/media/mnv/A2E41E9EE41E74AF/
vtkfile_m = File('/home/llg_nl/graphs/m.pvd')
vtkfile_cr = File('/home/llg_nl/graphs/cross.pvd')
vtkfile_diff = File('/home/llg_nl/graphs/diff.pvd')
vtkfile_hd_v = File('/home/llg_nl/graphs/hd_v.pvd')
vtkfile_hd_s = File('/home/llg_nl/graphs/hd_s.pvd')

vtkfile_e = File('/home/llg_nl/graphs/e.pvd')
vtkfile_P = File('/home/llg_nl/graphs/P.pvd')
# vtkfile_l = File('graphs/l.pvd')
#vtkfile_m << m
vtkfile_e << e_f
# vtkfile_m2 << m2
# vtkfile_m3 << m3
#vtkfile_m_3_in << m3
# In[4]:
# In[5]
mx, my, mz = m.split()
m_b_1, m_b_2, m_b_3 = split(m_b)
m_b_2d = as_vector((m_b_1,m_b_2))
phi_0 = Function(FS_1)
if wall_type == 'bloch':
    phi_0.vector()[:] = np.zeros(mx.compute_vertex_values().shape)
if wall_type == 'neel':
    phi_0 = project(-phi_nl, FS_1)
phi = DD_Hd.pot(m, wall_type, beta, phi_0, m_b_2d)
i = 0
j = 0
count = 0
dt = 0.555 #0.025 ## 0.01
Dt = Constant(dt)
T =  0
tol = 5E-8
theta = 1
E_old = 0
th = Constant(theta)
N_f = 100
n = FacetNormal(mesh)
oo = Constant(0)
PI = Constant(math.pi)
Hd_v_y = as_vector((oo, Constant(0.), oo)) #Constant(-26/2) on y axis
#hd_s+hd_ext
F = dot(w,(v-m)/Dt-al*cross(m,(v-m)/Dt))*dx \
+ (1-th)**2*dot(w,cross(m,h_rest(m,pp,e_f,dedz_v,M_s*M_s/2/k*phi,M_s*M_s/2/k*(hd_ext + Hd_v_y), kk, kk_o)))*dx  + (1-th)*th*dot(w,cross(v,h_rest(m,pp,e_f,dedz_v,M_s*M_s/2/k*phi,M_s*M_s/2/k*(hd_ext + Hd_v_y), kk, kk_o)))*dx + (1-th)*th*dot(w,cross(m,h_rest(v,pp,e_f,dedz_v,M_s*M_s/2/k*phi,M_s*M_s/2/k*(hd_ext + Hd_v_y), kk, kk_o)))*dx + th**2*dot(w,cross(v,h_rest(v,pp,e_f,dedz_v,M_s*M_s/2/k*phi,M_s*M_s/2/k*(hd_ext + Hd_v_y), kk, kk_o)))*dx \
    - (1-th)**2*dot_v(m,m,w,pp,e_f)*dx - (1-th)*th*dot_v(m,v,w,pp,e_f)*dx - (1-th)*th*dot_v(v,m,w,pp,e_f)*dx - th**2*dot_v(v,v,w,pp,e_f)*dx \
        + dot(w,cross(m_b,dmdn(m_b,n)))*ds + 2*pp*dot(w,cross(m_b,e_f))*dot(to_2d(m_b),n)*ds
Jac = derivative(F,v)
diffr = Function(FS)
Hd = Function(FS)

title = 't' + ', '  + 'w_ex' + ', '  + 'w_a' + ', '  + 'w_hd_1' + ', '  + 'w_hd_2' +  ', '  + 'w_me' +  ', ' + 'diff\n'
file_txt = open('/home/llg_nl/graphs/avg_table.txt','w')
file_txt.write(title)
file_txt.close()
while j <= 10:
    if i>=N_f:
        print(N_f, ' iterations reached')
        break
    #phi = DD_Hd.pot(m, wall_type, beta)
    #- M_s/2/k*Hy*w2*dx
    #- M_s/2/k*M_s*hy*w2*dx
    #- k_o/k*(v1-m1)*w1*dx
    #+ M_s*M_s/2/k*(w1*phi.dx(0) + w2*phi.dx(1))*dx
        # F = dot(w,(v-m)/Dt-al*cross(m,(v-m)/Dt))*dx \
        # + (1-th)**2*dot(w,cross(m,h_rest(m,pp,e_f)))*dx  + (1-th)*th*dot(w,cross(v,h_rest(m,pp,e_f)))*dx + (1-th)*th*dot(w,cross(m,h_rest(v,pp,e_f)))*dx + th**2*dot(w,cross(v,h_rest(v,pp,e_f)))*dx \
        #     - (1-th)**2*dot_v(m,m,w,pp,e_f)*dx - (1-th)*th*dot_v(m,v,w,pp,e_f)*dx - (1-th)*th*dot_v(v,m,w,pp,e_f)*dx - th**2*dot_v(v,v,w,pp,e_f)*dx \
        #         + 2*pp*dot(w,cross(m_b,e_f))*dot(to_2d(m_b),n)*ds
    solve(F==0,v,J=Jac) # BC!!!
    
    v = norm_sol_s(v, FS)
    V = v.vector()
    M = m.vector()
    Diffr = V - M
    diffr.vector()[:] = Diffr/(Lx*Ly*dt)
    #Hd_v = project(-grad3(phi),FS)
    #cr = project(cross(m,dmdn(m,n)),FS)
    #Hd.vector()[:] = Hd_v.vector() + hd_s.vector() + hd_ext.vector()
    error = (m-v)**2*dx
    E = sqrt(abs(assemble(error)))/(Lx*Ly)/dt
    
    w_ex = assemble((dot(grad(m1),grad(m1)) + dot(grad(m2),grad(m2)) + dot(grad(m3),grad(m3)))*dx)/(Lx*Ly)
    w_a = assemble(-m3*m3*dx)/(Lx*Ly)
    w_hd_1 = assemble(-dot(to_2d(m),-grad(phi))*dx)/(Lx*Ly)
    w_hd_2 = assemble(-dot(m,hd_ext+Hd_v_y)*dx)/(Lx*Ly)
    w_me = assemble(pp*dot(e_f,m*div(to_2d(m)) - grad(m)*m)*dx)/(Lx*Ly)
    data_ex = str(w_ex)
    data_a = str(w_a)
    data_hd_1 = str(w_hd_1)
    data_hd_2 = str(w_hd_2)
    data_me = str(w_me)
    data = str(round(T,5)) + ', ' + data_ex + ', ' + data_a + ', ' + data_hd_1 + ', ' + data_hd_2 + ', ' + data_me + ', ' + str(E) + '\n'
    if i%5 == 0:
        vtkfile_m << (m, T)
        vtkfile_hd_v << (phi, T)
        #vtkfile_hd_s << hd_s
        vtkfile_diff << (diffr, T)
        file_txt = open('/home/llg_nl/avg_table.txt','a')
        file_txt.write(data)
        file_txt.close()
        #vtkfile_cr << cr
        
    # vtkfile_m2 << m2
    # vtkfile_m3 << m3
    # vtkfile_l << u_l
    #plot(u3)
    
    v1, v2, v3 = v.split()
    # P = project(m*(m1.dx(0) + m2.dx(1)) - as_vector((m1*m1.dx(0)+m2*m1.dx(1), m1*m2.dx(0)+m2*m2.dx(1), m1*m3.dx(0)+m2*m3.dx(1))), FS_3)
    # vtkfile_P << P
    # error = (m-v)**2*dx
    # E = sqrt(abs(assemble(error)))/(Lx*Ly)/dt
    delta_E = E-E_old
    E_old = E
    print('delta = ', E, ', ', 'i = ', i)
    if E <= tol:
        j += 1
    i += 1
    
    if (abs(delta_E/E) <= 5E-3) and (delta_E < 0):
        count += 1
    else:
        count = 0
    if count >= 50:
        count = 0
        dt = round(dt + 0.01, 4) #0.05
        Dt.assign(dt)
        print('NEW Time Step:', dt)
    
    m.assign(v)
    phi_n = DD_Hd.pot(m, wall_type, beta, phi, m_b_2d)
    #hd_s_n = DD_Hd.s_chg(m3, SL_space, FS_1, FS_3_1, FS_3_1, FS, idx, space_top, slp_pot, trace_space, trace_matrix)
    phi.assign(phi_n)
    #hd_s.assign(hd_s_n)
    # U = u.vector()
    # m.vector()[:] = U
    m1, m2, m3 = m.split()
    T = T + dt


plot(v3)
# vtkfile_m << m
# vtkfile_phi << phi
# P = project(m*(m1.dx(0) + m2.dx(1)) - as_vector((m1*m1.dx(0)+m2*m1.dx(1), m1*m2.dx(0)+m2*m2.dx(1), m1*m3.dx(0)+m2*m3.dx(1))), FS)
# vtkfile_P << P
time_new.store(m.vector(),i)
print(i)
# In[ ]:


