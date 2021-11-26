#sudo docker run --rm -ti -m=8g -v $(pwd):/home/fenics/shared:z multiphenics/multiphenics
from __future__ import print_function
import numpy as np
from dolfin import *
import sympy
import matplotlib.pyplot as plt
parameters['allow_extrapolation'] = True
parameters["form_compiler"]["representation"] = 'uflacs'
from mshr import *
from multiphenics import *
parameters["ghost_mode"] = "shared_facet" 


# Number of rellienement
Iter_standard = 5
Iter_phi = 5


# parameter of the ghost penalty
sigma = 20.0


#parameter in the construction of the exact mesh
param_mesh_exact = 10


# Polynome Pk
polV = 2
#parameters["form_compiler"]["quadrature_degree"]=2*(polV+1)
polphi=polV+1

# Ghost penalty
ghost = True


# plot the solution
Plot = False


# Compute the conditioning number
conditioning = False


# parameters
R=0.21 # radius of the solid
rho_f = 1.0
rho_s = 2.0 # density of the solid
m=rho_s*np.pi*R**2 # mass of the solid
G=-10.0 # gravity


# parameters of chi
r0 = 0.25 # chi(r0)=1
r1 = 0.45 # chi(r1)=0


# construction of the function chi : chi(r0)=1 and chi(r1)=0
class ChiExpr(UserExpression):
    def eval(self, value, x):
        r = ((x[0]-0.5)**2+(x[1]-0.5)**2)**(0.5)
        if r<=r0:
            value[0] = 1.0
        if r>=r1:
            value[0] = 0.0
        if r<r1 and r>r0:
            value[0] = 1.0+(-6.0*r**5+15.0*(r0+r1)*r**4-10.0*(r0**2+4.0*r0*r1+r1**2)*r**3+30.0*r0*r1*(r0+r1)*r**2-30.0*r0**2*r1**2*r + r0**3*(r0**2-5.0*r1*r0+10.0*r1**2))/(r1-r0)**5
    def value_shape(self):
        return (2,)


# Function used to write in the outputs files
def output_latex(f,A,B):
	for i in range(len(A)):
		f.write('(')
		f.write(str(A[i]))
		f.write(',')
		f.write(str(B[i]))
		f.write(')\n')
	f.write('\n')


# point (x,y) dans Omega
def Omega(x,y):
    return ((x-0.5)**2+(y-0.5)**2)**0.5>R or near(((x-0.5)**2+(y-0.5)**2)**0.5,R)


############################################
##### construction of the exact solution ###
############################################

# construction of the mesh
domain_rec = Rectangle(Point(0.0,0.0),Point(1.0,1.0))
domain_circle = Circle(Point(0.5,0.5),R)
mesh_exact = generate_mesh(domain_rec-domain_circle,param_mesh_exact)
print("num cells mesh exact:",mesh_exact.num_cells())


# Restriction on Gamma (boundary of the solide)
class OnGamma(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and ( (pow(x[0]-0.5,2)+pow(x[1]-0.5,2))**0.5<=(0.5+R)/2.0 or near((pow(x[0]-0.5,2)+pow(x[1]-0.5,2))**0.5,(0.5+R)/2.0))
on_Gamma=OnGamma()
gamma_restriction = MeshRestriction(mesh_exact,on_Gamma)
File("data/Gamma_Restriction.rtc.xml") << gamma_restriction
XDMFFile("data/Gamma_Restriction.rtc.xdmf").write(gamma_restriction)
gamma_rest = MeshRestriction(mesh_exact, "data/Gamma_Restriction.rtc.xml")


# Construction of the space
V_u_exact = VectorFunctionSpace(mesh_exact, 'CG',polV)
V_p_exact = FunctionSpace(mesh_exact, 'CG',polV-1)
V_real_exact = VectorFunctionSpace(mesh_exact, 'R',0,4)
V_mult_exact = VectorFunctionSpace(mesh_exact, 'CG',polV)
V_exact = BlockFunctionSpace([V_u_exact,V_p_exact,V_real_exact,V_mult_exact], restrict=[None,None,gamma_rest,gamma_rest])
print('dim V',V_exact.dim())


# Dirichlet
class DirichletBoundary(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and ((pow(x[0]-0.5,2)+pow(x[1]-0.5,2))**0.5>=(0.5+R)/2.0 or near((pow(x[0]-0.5,2)+pow(x[1]-0.5,2))**0.5,(0.5+R)/2.0))
boundary_exact = MeshFunction("size_t", mesh_exact, mesh_exact.topology().dim()-1)
boundary_exact.set_all(0)
dirichlet_boundary = DirichletBoundary()
dirichlet_boundary.mark(boundary_exact, 1)
bc = DirichletBC(V_exact.sub(0), Constant((0.,0.)), boundary_exact, 1)
bcs = BlockDirichletBC([bc,None,None,None])


# Computation of the source term
g_exact = Expression(('0.0','G'),degree=polV,domain=mesh_exact,G=G)

# definition of r
xr = Expression(('x[1]-0.5','0.5-x[0]'),degree=2,domain=mesh_exact)

# dx ds
dx_exact = Measure("dx")(domain = mesh_exact)
ds_exact = Measure("ds")(domain = mesh_exact,subdomain_data = boundary_exact)


# Resolution
trial = BlockTrialFunction(V_exact)
u, p, real, l2  = block_split(trial)
U = as_vector([real[0],real[1]])
psi = real[2]
l1=real[3]
test = BlockTestFunction(V_exact)
v, q, realt, mu2 = block_split(test)
Vt = as_vector([realt[0],realt[1]])
xi = realt[2]
mu1 = realt[3]

# construction of the bilinear form
a_u_v = 2.0*inner(sym(grad(u)),sym(grad(v)))*dx_exact 
a_p_v = - p*div(v)*dx_exact 
a_real_v = 0
a_l2_v = inner(l2,v)*ds_exact(0)

a_u_q = - q*div(u)*dx_exact
a_p_q = 0
a_real_q =l1*q*dx_exact
a_l2_q =0

a_u_realtest =  0
a_p_realtest =  mu1*p*dx_exact
a_real_realtest = 0
a_l2_realtest = inner(l2,-Vt-xi*xr)*ds_exact(0)

a_u_mu2 = inner(mu2,u)*ds_exact(0)
a_p_mu2 = 0
a_real_mu2 = inner(mu2,-U-psi*xr)*ds_exact(0)
a_l2_mu2 = 0


# Construction of the linear form
l_v = inner(v,rho_f*g_exact)*dx_exact
l_q = 0
l_realtest = (1.0/(2.0*np.pi*R))*inner(Vt,m*g_exact)*ds_exact(0)
l_mu2 = 0

# assemble the block
a_exact = [[ a_u_v,  a_p_v, a_real_v, a_l2_v],
       [ a_u_q,  a_p_q , a_real_q, a_l2_q],
       [a_u_realtest,  a_p_realtest, a_real_realtest, a_l2_realtest],
       [ a_u_mu2,  a_p_mu2, a_real_mu2, a_l2_mu2]]
l_exact =  [l_v,l_q,l_realtest,l_mu2]

# SOLVE 
A_exact = block_assemble(a_exact)
L_exact = block_assemble(l_exact)
bcs.apply(A_exact)
bcs.apply(L_exact)
sol_exact = BlockFunction(V_exact)
print("Assemble the matrices: ok")
block_solve(A_exact, sol_exact.block_vector(), L_exact,"mumps")
print("Solve the problem: ok")
u_exact, p_exact, real_exact, l2 =  block_split(sol_exact)
U_x_exact, U_y_exact, psi_exact = real_exact[0], real_exact[1], real_exact[2]


print('L2-norm of the exact sol: ',assemble(inner(u_exact,u_exact)*dx_exact)**0.5)
print('h :',mesh_exact.hmax())
print("U psi exact:",real_exact(0,0))


# plot
plt.clf()
plot_sol = plot(u_exact)
#plot(mesh_exact)
plt.savefig('output/stokes.png')
#file = File('stokes.pvd')
#file << u_exact

plt.clf()
plot_mesh = plot(mesh_exact)
file = File('output/stokes_mesh_exact.pvd')
file << mesh_exact
plt.savefig('output/stokes_mesh_exact.png')


#############################################
##### construction of the standard solution ###
#############################################
size_mesh_standard_vec = np.zeros(Iter_standard)
error_u_L2_standard_vec = np.zeros(Iter_standard)
error_u_H1_standard_vec = np.zeros(Iter_standard)
error_p_L2_standard_vec = np.zeros(Iter_standard)
for i in range(Iter_standard):
	print('##################')
	print('## Iteration classical FEM',i+1,'##')
	print('##################')
	# construction of the mesh
	domain_rec = Rectangle(Point(0.0,0.0),Point(1.0,1.0))
	domain_circle = Circle(Point(0.5,0.5),R)
	mesh = generate_mesh(domain_rec-domain_circle,10*2**i)
	print("num cells mesh exact:",mesh.num_cells())



	# Restriction on Gamma (boundary of the solide)
	gamma_restriction = MeshRestriction(mesh,on_Gamma)
	File("data/Gamma_Restriction.rtc.xml") << gamma_restriction
	XDMFFile("data/Gamma_Restriction.rtc.xdmf").write(gamma_restriction)
	gamma_rest = MeshRestriction(mesh, "data/Gamma_Restriction.rtc.xml")


	# Construction of the space
	V_u = VectorFunctionSpace(mesh, 'CG',polV)
	V_p = FunctionSpace(mesh, 'CG',polV-1)
	V_real = VectorFunctionSpace(mesh, 'R',0,4)
	V_mult = VectorFunctionSpace(mesh, 'CG',polV)
	V = BlockFunctionSpace([V_u,V_p,V_real,V_mult], restrict=[None,None,gamma_rest,gamma_rest])
	print('dim V',V.dim())


	# Dirichlet
	boundary = MeshFunction("size_t", mesh, mesh.topology().dim()-1)
	boundary.set_all(0)
	dirichlet_boundary = DirichletBoundary()
	dirichlet_boundary.mark(boundary, 1)
	bc = DirichletBC(V.sub(0), Constant((0.,0.)), boundary, 1)
	bcs = BlockDirichletBC([bc,None,None,None])


	# Computation of the source term
	g = Expression(('0.0','G'),degree=polV,domain=mesh,G=G)

	# definition of r
	xr = Expression(('x[1]-0.5','0.5-x[0]'),degree=2,domain=mesh)

	# dx ds
	dx = Measure("dx")(domain = mesh)
	ds = Measure("ds")(domain = mesh,subdomain_data = boundary)


	# Resolution
	trial = BlockTrialFunction(V)
	u, p, real, l2  = block_split(trial)
	U = as_vector([real[0],real[1]])
	psi = real[2]
	l1=real[3]
	test = BlockTestFunction(V)
	v, q, realt, mu2 = block_split(test)
	Vt = as_vector([realt[0],realt[1]])
	xi = realt[2]
	mu1 = realt[3]

	# construction of the bilinear form
	a_u_v = 2.0*inner(sym(grad(u)),sym(grad(v)))*dx 
	a_p_v = - p*div(v)*dx 
	a_real_v = 0
	a_l2_v = inner(l2,v)*ds(0)

	a_u_q = - q*div(u)*dx
	a_p_q = 0
	a_real_q =l1*q*dx
	a_l2_q =0

	a_u_realtest =  0
	a_p_realtest =  mu1*p*dx
	a_real_realtest = 0
	a_l2_realtest = inner(l2,-Vt-xi*xr)*ds(0)

	a_u_mu2 = inner(mu2,u)*ds(0)
	a_p_mu2 = 0
	a_real_mu2 = inner(mu2,-U-psi*xr)*ds(0)
	a_l2_mu2 = 0


	# Construction of the linear form
	l_v = inner(v,rho_f*g)*dx
	l_q = 0
	l_realtest = (1.0/(2.0*np.pi*R))*inner(Vt,m*g)*ds(0)
	l_mu2 = 0

	# assemble the block
	a = [[ a_u_v,  a_p_v, a_real_v, a_l2_v],
		   [ a_u_q,  a_p_q , a_real_q, a_l2_q],
		   [a_u_realtest,  a_p_realtest, a_real_realtest, a_l2_realtest],
		   [ a_u_mu2,  a_p_mu2, a_real_mu2, a_l2_mu2]]
	l =  [l_v,l_q,l_realtest,l_mu2]

	# SOLVE 
	A = block_assemble(a)
	L = block_assemble(l)
	bcs.apply(A)
	bcs.apply(L)
	sol = BlockFunction(V)
	print("Assemble the matrices: ok")
	block_solve(A, sol.block_vector(), L,"mumps")
	print("Solve the problem: ok")
	u, p, real, l2 =  block_split(sol)
	U_x, U_y, psi = real[0], real[1], real[2]


	# Computation of the error
	#u_exact = interpolate(u_exact,V_u)
	norm_L2_u_exact =assemble(inner(u_exact,u_exact)*dx)**0.5
	norm_H1_u_exact =assemble(grad(u_exact)**2*dx)**0.5
	norm_L2_p_exact =assemble(p_exact**2*dx)**0.5
	err_L2 = assemble((u-u_exact)**2*dx)**0.5/norm_L2_u_exact
	err_H1 = assemble((grad(u-u_exact))**2*dx)**0.5/norm_H1_u_exact
	err_p = assemble((p-p_exact)**2*dx)**0.5/norm_L2_p_exact
	size_mesh_standard_vec[i] = mesh.hmax()
	error_u_L2_standard_vec[i] = err_L2
	error_u_H1_standard_vec[i] = err_H1
	error_p_L2_standard_vec[i] = err_p
	print('h :',mesh.hmax())
	print('relative L2 error u : ',err_L2)
	print('relative H1 error u : ',err_H1)
	print('relative L2 error p : ',err_p)	




#############################################
##### construction of the phiFEM solution ###
#############################################

# Initialistion of the output
size_mesh_phi_vec = np.zeros(Iter_phi)
error_u_L2_phi_vec = np.zeros(Iter_phi)
error_u_H1_phi_vec = np.zeros(Iter_phi)
error_p_L2_phi_vec = np.zeros(Iter_phi)
cond_phi_vec = np.zeros(Iter_phi)
for i in range(Iter_phi):
	print('###########################')
	print('## Iteration phi FEM',i+1,'##')
	print('###########################')


	# Construction of the mesh
	N = int(10*2**((i)))
	mesh_macro = RectangleMesh(Point(0.0, 0.0), Point(1.0, 1.0), N, N)
	domains = MeshFunction("size_t", mesh_macro, mesh_macro.topology().dim())
	domains.set_all(0)
	for ind in range(mesh_macro.num_cells()):
		mycell = Cell(mesh_macro,ind)
		v1x,v1y,v2x,v2y,v3x,v3y = mycell.get_vertex_coordinates()
		if Omega(v1x,v1y) or Omega(v2x,v2y) or Omega(v3x,v3y):
			domains[ind] = 1
	mesh = SubMesh(mesh_macro, domains, 1)
	mesh_compl = SubMesh(mesh_macro, domains, 0) # to compute the integral on S
	print("num cells mesh :",mesh.num_cells())

	plt.clf()
	plot_mesh = plot(mesh)
	file = File('output/stokes_mesh_phi.pvd')
	file << mesh
	plt.savefig('output/stokes_mesh_phi.png')


	# Construction of phi
	V_phi = FunctionSpace(mesh,'CG',polphi)
	phi = Expression('-(x[0]-0.5)*(x[0]-0.5)-(x[1]-0.5)*(x[1]-0.5)+R*R',R=R,degree=polphi,domain=mesh)
	phi = interpolate(phi,V_phi)


	# Computation of the source term
	g_expr = Expression(('0.0','G'),degree=polV,domain=mesh,G=G)


	# Facets and cells where we apply the ghost penalty
	mesh.init(1,2)
	vertex_ghost = MeshFunction("size_t", mesh, mesh.topology().dim()-2)
	facet_ghost = MeshFunction("size_t", mesh, mesh.topology().dim()-1)
	cell_ghost = MeshFunction("size_t", mesh, mesh.topology().dim())
	vertex_ghost.set_all(0)
	facet_ghost.set_all(0)
	cell_ghost.set_all(0)
	for mycell in cells(mesh):
		for myfacet in facets(mycell):
			v1, v2 = vertices(myfacet)
			if phi(v1.point().x(),v1.point().y())*phi(v2.point().x(),v2.point().y())<0 or near(phi(v1.point().x(),v1.point().y())*phi(v2.point().x(),v2.point().y()),0):
				cell_ghost[mycell] = 1
				for myfacet2 in facets(mycell):
					facet_ghost[myfacet2] = 1
				for myvertex in vertices(mycell):
					vertex_ghost[myvertex] = 1


	# for the dirichlet condition
	for myfacet in facets(mesh):
		if facet_ghost[myfacet]==0 and myfacet.exterior():
			facet_ghost[myfacet]=3

	#####################################
	###  construction of chi  ###########
	#####################################
	#  construction of chi  
	V_chi = FunctionSpace(mesh,'CG',5)
	chi = ChiExpr(element=V_chi.ufl_element())
	chi = interpolate(chi,V_chi)
	plot(chi)
	plt.savefig('chi')

	"""V_chi = FunctionSpace(mesh,'CG',1)

	# for the construction of chi
	for mycell in cells(mesh):
		if cell_ghost[mycell] ==0:
			for myfacet in facets(mycell):
				v1, v2 = vertices(myfacet)
				if (vertex_ghost[v1] == 1 and vertex_ghost[v2] == 0) or (vertex_ghost[v2] == 1 and vertex_ghost[v1] == 0):
					facet_ghost[myfacet] = 2
					cell_ghost[mycell] == 2


	# Initialize cell function for domains
	dx = Measure("dx")(domain = mesh,subdomain_data = cell_ghost)

	# Resolution to construct chi
	u = TrialFunction(V_chi)
	v = TestFunction(V_chi)

	# construction of the bilinear form to construct chi
	a = u*v*dx(0) + u*v*dx(1)
	L = v*Constant(1.0)*dx(1)

	# Solve the linear system
	chi = Function(V_chi)
	solve(a == L, chi)

	# plot	
	plt.clf()
	#plot_sol = plot(mesh)
	#chi = interpolate(Expression('1.0',degree=polV,domain=mesh),V_phi)
	plot_sol = plot(chi)
	plt.savefig('output/chi_{0}.png'.format(i))"""

	#####################################
	###  end construction of chi  ###########
	#####################################


	# compute the intrale on O\Omega_h
	g_expr_compl = 	Expression(('0.0','G'),degree=polV,domain=mesh_compl,G=G)
	dx = Measure("dx")(domain = mesh_compl)
	xr_compl = Expression(('x[1]-0.5','0.5-x[0]'),degree=2,domain=mesh_compl)
	int_compl_chi_r = assemble(rho_f*inner(g_expr_compl,xr_compl)*dx)
	int_compl_gx = assemble(rho_f*g_expr_compl[0]*dx)
	int_compl_gy = assemble(rho_f*g_expr_compl[1]*dx)
	int_compl = assemble(Expression('1.0',degree=polV,domain=mesh_compl)*dx)


	# Construction of the space
	V_u = VectorFunctionSpace(mesh, 'CG',polV)
	V_p = FunctionSpace(mesh, 'CG',polV-1)
	V_real = VectorFunctionSpace(mesh, 'R',0,4)
	V = BlockFunctionSpace([V_u,V_p,V_real], restrict=[None,None,None])
	print('dim V',V.dim())


	# Boundary condition
	bc = DirichletBC(V.sub(0), Constant((0.,0.)),facet_ghost, 3)
	bcs = BlockDirichletBC([bc,None,None])


	# dx ds
	dx = Measure("dx")(domain = mesh,subdomain_data = cell_ghost)
	ds = Measure("ds")(domain = mesh,subdomain_data = facet_ghost)


	# Resolution
	trial = BlockTrialFunction(V)
	w, p, real = block_split(trial)
	U = as_vector([real[0],real[1]])
	psi = real[2]
	l=real[3]
	test = BlockTestFunction(V)
	s, q, realt = block_split(test)
	Vt = as_vector([realt[0],realt[1]])
	xi = realt[2]
	mu = realt[3]


	# construction of the bilinear form
	n = FacetNormal(mesh)
	h = CellDiameter(mesh)
	xr = Expression(('x[1]-0.5','0.5-x[0]'),degree=2,domain=mesh)

	a_w_s = 2.0*inner(sym(grad(w*phi)),sym(grad(s*phi)))*dx -2.0*inner(dot(sym(grad(w*phi)),n),phi*s)*ds(1)
	a_p_s = inner(dot(p*Identity(2),n),phi*s)*ds(1) - p*div(phi*s)*dx 
	a_real_s = 2.0*inner(sym(grad(chi*(U+psi*xr))),sym(grad(s*phi)))*dx -2.0*inner(dot(sym(grad(chi*(U+psi*xr))),n),phi*s)*ds(1)
	a_w_q = - q*div(phi*w)*dx
	a_p_q = 0
	a_real_q =- q*div(chi*(U+psi*xr))*dx + l*q*dx
	a_w_realtest =  2.0*inner(sym(grad(w*phi)),sym(grad(chi*(Vt+xi*xr))))*dx
	a_p_realtest =  - p*div(chi*(Vt+xi*xr))*dx  + mu*p*dx
	a_real_realtest = 2.0*inner(sym(grad(chi*(U+psi*xr))),sym(grad(chi*(Vt+xi*xr))))*dx


	# add or not the ghost penalty to the bilinear form
	if ghost==True:
		a_w_s +=  sigma*h**2*inner(div(grad(phi*w)),div(grad(phi*s)))*dx(1) + sigma*div(phi*w)*div(phi*s)*dx(1)+sigma*avg(h)*inner(jump(grad(phi*w),n),jump(grad(phi*s),n))*dS(1) +sigma*avg(h)**3*inner(jump(grad(grad(phi*w)),n),jump(grad(grad(phi*s)),n))*dS(1)
		a_p_s += -sigma*h**2*inner(div(grad(phi*s)),grad(p))*dx(1)
		a_w_q += sigma*h**2*inner(div(grad(phi*w)),grad(q))*dx(1)
		a_p_q += -sigma*h**2*inner(grad(q),grad(p))*dx(1) 

		###  U and psi in ghost penaly
		a_real_s +=  sigma*div(chi*(U+psi*xr))*div(phi*s)*dx(1) + sigma*h**2*inner(div(grad(chi*(U+psi*xr))),div(grad(phi*s)))*dx(1)+sigma*avg(h)*inner(jump(grad(chi*(U+psi*xr)),n),jump(grad(phi*s),n))*dS(1) +sigma*avg(h)**3*inner(jump(grad(grad(chi*(U+psi*xr))),n),jump(grad(grad(phi*s)),n))*dS(1)
		a_real_q += sigma*h**2*inner(div(grad(chi*(U+psi*xr))),grad(q))*dx(1)
		a_w_realtest +=  sigma*div(phi*w)*div(chi*(Vt+xi*xr))*dx(1) + sigma*h**2*inner(div(grad(phi*w)),div(grad(chi*(Vt+xi*xr))))*dx(1)+sigma*avg(h)*inner(jump(grad(phi*w),n),jump(grad(chi*(Vt+xi*xr)),n))*dS(1) +sigma*avg(h)**3*inner(jump(grad(grad(phi*w)),n),jump(grad(grad(chi*(Vt+xi*xr))),n))*dS(1)
		a_p_realtest +=  -sigma*h**2*inner(div(grad(chi*(Vt+xi*xr))),grad(p))*dx(1)
		a_real_realtest += sigma*div(chi*(U+psi*xr))*div(chi*(Vt+xi*xr))*dx(1) + sigma*h**2*inner(div(grad(chi*(U+psi*xr))),div(grad(chi*(Vt+xi*xr))))*dx(1)+sigma*avg(h)*inner(jump(grad(chi*(U+psi*xr)),n),jump(grad(chi*(Vt+xi*xr)),n))*dS(1) +sigma*avg(h)**3*inner(jump(grad(grad(chi*(U+psi*xr))),n),jump(grad(grad(chi*(Vt+xi*xr))),n))*dS(1)


	# Construction of the linear form
	l_s = inner(phi*s,rho_f*g_expr)*dx
	l_q = 0
	l_realtest = inner(rho_f*g_expr,chi*(Vt+xi*xr))*dx+(1.0/(1.0-int_compl))*(int_compl_gx*Vt[0]+int_compl_gy*Vt[1]+int_compl_chi_r*xi+ (1.0-rho_f/rho_s)*m*inner(g_expr,Vt))*dx


	# add or not the ghost penalty to the linear form
	if ghost==True:
		l_s += -sigma*h**2*rho_f*inner(g_expr,div(grad(phi*s)))*dx(1) 
		l_q += -sigma*h**2*rho_f*inner(g_expr,grad(q))*dx(1)

		###  U and psi in ghost penaly
		l_realtest += -sigma*h**2*rho_f*inner(g_expr,div(grad(chi*(Vt+xi*xr))))*dx(1) 


	# assemble the block
	a = [[ a_w_s,  a_p_s, a_real_s],
		   [ a_w_q,  a_p_q , a_real_q],
		   [a_w_realtest,  a_p_realtest, a_real_realtest]]
	l =  [l_s,l_q,l_realtest]


	# assemble and solve
	A = block_assemble(a)
	L = block_assemble(l)
	bcs.apply(A)
	bcs.apply(L)
	sol = BlockFunction(V)
	print("Assemble the matrices: ok")
	block_solve(A, sol.block_vector(), L,"mumps")
	print("Solve the problem: ok")
	u, p, real =  block_split(sol)
	U_x, U_y, psi = real[0], real[1], real[2]
	U = as_vector([U_x,U_y])
	u = phi*u + chi*(U + psi*xr)


	p_exact = p_exact - assemble(p_exact*dx_exact)

	# Computation of the error
	Iu_h = project(u,V_u_exact)
	Ip_h = project(p,V_p_exact)
	Ip_h = Ip_h - assemble(Ip_h*dx_exact)
	norm_L2_u_h =assemble((Iu_h)**2*dx_exact)**0.5
	norm_L2_u_h =assemble((Iu_h)**2*dx_exact)**0.5
	err_L2 = assemble((Iu_h-u_exact)**2*dx_exact)**0.5/assemble((u_exact)**2*dx_exact)**0.5
	err_H1 = assemble((grad(Iu_h-u_exact))**2*dx_exact)**0.5/assemble((grad(u_exact))**2*dx_exact)**0.5
	err_p = assemble(((Ip_h-p_exact))**2*dx_exact)**0.5/assemble(((p_exact))**2*dx_exact)**0.5
	size_mesh_phi_vec[i] = mesh_macro.hmax()
	error_u_L2_phi_vec[i] = err_L2
	error_u_H1_phi_vec[i] = err_H1
	error_p_L2_phi_vec[i] = err_p
	print('h :',mesh_macro.hmax())
	print('norm L2 of u_h : ',norm_L2_u_h)
	print('relative L2 error u : ',err_L2)
	print('relative H1 error u : ',err_H1)
	print('relative L2 error p : ',err_p)	
	print("U psi: ",real(0,0))
	#print('L2 norm : ',assemble((u_expr)**2*dx(0))**0.5)	
	if conditioning == True:
		A = np.matrix(assemble(a).array())
		ev, eV = np.linalg.eig(A)
		ev = abs(ev)
		#cond = mesh.hmax()**2*np.max(ev)/np.min(ev)
		cond = np.max(ev)/np.min(ev)
		cond_phi_vec[i] = cond
		print("conditioning number x h^2",cond)
	print("num cells mesh",mesh.num_cells())
	print('')

	"""plt.clf()
	plot_p = plot(p)
	plt.savefig('p_{0}.png'.format(i))"""


# Print the output vectors
print('##################################')
print('######## clasical FEM ############')
print('##################################')
print('Vector h :',size_mesh_standard_vec)
print('Vector relative L2 error u standard : ',error_u_L2_standard_vec)
print('Vector relative H1 error u standard : ',error_u_H1_standard_vec)
print('Vector relative L2 error p standard : ',error_p_L2_standard_vec)
print('##################################')
print('########    phi FEM   ############')
print('##################################')
print('Vector h :',size_mesh_phi_vec)
print('Vector relative L2 error u phiFEM : ',error_u_L2_phi_vec)
print('Vector relative H1 error u phiFEM : ',error_u_H1_phi_vec)
print('Vector relative L2 error p phiFEM : ',error_p_L2_phi_vec)
if conditioning == True:
    print("conditioning number",cond_phi_vec)


# plot error
plt.clf()
plot_sol = plt.loglog(size_mesh_standard_vec,error_u_L2_standard_vec,"o-",label="L2 Error u standard")
plot_sol = plt.loglog(size_mesh_phi_vec,error_u_L2_phi_vec,"o-",label="L2 Error u phi FEM")
x = [size_mesh_standard_vec[0],size_mesh_standard_vec[-1]]
y = [error_u_L2_standard_vec[0],error_u_L2_standard_vec[0]*size_mesh_standard_vec[-1]**2*size_mesh_standard_vec[0]**(-2)]
plot_sol = plt.loglog(x,y,label="h^2")
x = [size_mesh_phi_vec[0],size_mesh_phi_vec[-1]]
y = [error_u_L2_phi_vec[0],error_u_L2_phi_vec[0]*size_mesh_phi_vec[-1]**3*size_mesh_phi_vec[0]**(-3)]
plot_sol = plt.loglog(x,y,label="h^3")
plt.legend()
plt.savefig('output/stokes_error_u_L2.png')
plt.clf()
plot_sol = plt.loglog(size_mesh_standard_vec,error_u_H1_standard_vec,"o-",label="H1 Error u standard")
plot_sol = plt.loglog(size_mesh_phi_vec,error_u_H1_phi_vec,"o-",label="H1 Error u phi FEM")
x = [size_mesh_standard_vec[0],size_mesh_standard_vec[-1]]
y = [error_u_H1_standard_vec[0],error_u_H1_standard_vec[0]*size_mesh_standard_vec[-1]**2*size_mesh_standard_vec[0]**(-2)]
plot_sol = plt.loglog(x,y,label="h^2")
x = [size_mesh_standard_vec[0],size_mesh_standard_vec[-1]]
y = [error_u_H1_standard_vec[0],error_u_H1_standard_vec[0]*size_mesh_standard_vec[-1]*size_mesh_standard_vec[0]**(-1)]
plot_sol = plt.loglog(x,y,label="h")
plt.legend()
plt.savefig('output/stokes_error_u_H1.png')
plt.clf()
plot_sol = plt.loglog(size_mesh_standard_vec,error_p_L2_standard_vec,"o-",label="L2 Error p standard")
plot_sol = plt.loglog(size_mesh_phi_vec,error_p_L2_phi_vec,"o-",label="L2 Error p phi FEM")
x = [size_mesh_standard_vec[0],size_mesh_standard_vec[-1]]
y = [error_u_H1_standard_vec[0],error_u_H1_standard_vec[0]*size_mesh_standard_vec[-1]**2*size_mesh_standard_vec[0]**(-2)]
plot_sol = plt.loglog(x,y,label="h^2")
plt.legend()
plt.savefig('output/stokes_error_p.png')


#  Write the output file for latex
if ghost == False:
	f = open('output/output_stokes_no_ghost.txt','w')
if ghost == True:
	f = open('output/output_stokes_ghost.txt','w')
f.write('relative L2 norm u standard : \n')	
output_latex(f,size_mesh_standard_vec,error_u_L2_standard_vec)
f.write('relative H1 norm u standard : \n')	
output_latex(f,size_mesh_standard_vec,error_u_H1_standard_vec)
f.write('relative L2 norm p standard : \n')	
output_latex(f,size_mesh_standard_vec,error_p_L2_standard_vec)
f.write('relative L2 norm u phifem : \n')	
output_latex(f,size_mesh_phi_vec,error_u_L2_phi_vec)
f.write('relative H1 norm u phifem : \n')	
output_latex(f,size_mesh_phi_vec,error_u_H1_phi_vec)
f.write('relative L2 norm p phifem : \n')	
output_latex(f,size_mesh_phi_vec,error_p_L2_phi_vec)
f.write('conditioning number phifem : \n')	
output_latex(f,size_mesh_phi_vec,cond_phi_vec)
f.close()
