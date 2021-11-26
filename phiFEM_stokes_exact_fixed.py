#docker pull multiphenics/multiphenics
##docker run --rm -ti -m=8g -v $(pwd):/home/fenics/shared:z multiphenics/multiphenics
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

# extrapolation of phi
extrapol = False


# Polynome Pk
polV = 2
#parameters["form_compiler"]["quadrature_degree"]=2*(polV+1)
polPhi =polV+1

# Ghost penalty
ghost = True


# plot the solution
Plot = False


# Compute the conditioning number
conditioning = False


# parameters
R=0.21 # radius of the solid


# parameters of chi
r0 = 0.25 # chi(r0)=1
r1 = 0.45 # chi(r1)=1


# construction of the function chi
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

# Dirichlet
class DirichletBoundary_ext(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and (pow(x[0]-0.5,2)+pow(x[1]-0.5,2))**0.5>=(0.5+R)/2.0 
class DirichletBoundary_int(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and (pow(x[0]-0.5,2)+pow(x[1]-0.5,2))**0.5<=(0.5+R)/2.0 



## FEM standard
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


	# Construction of the space
	V_u = VectorFunctionSpace(mesh, 'CG',polV)
	V_p = FunctionSpace(mesh, 'CG',polV-1)
	V_real = FunctionSpace(mesh, 'R',0)
	V = BlockFunctionSpace([V_u,V_p,V_real], restrict=[None,None,None])
	print('dim V',V.dim())


	# computation of the exact solution
	u_exact = Expression(('cos(pi*x[0])*sin(pi*x[1])','-sin(pi*x[0])*cos(pi*x[1])'),domain=mesh,degree=polV+6)
	p_exact = Expression('(x[1]-0.5)*cos(2.0*pi*x[0]) + (x[0]-0.5)*sin(2.0*pi*x[1])',domain=mesh,degree=polV+6)


	# Computation of the source term
	# Computation of the source term
	x, y = sympy.symbols('xx yy')
	u1 = sympy.cos(sympy.pi*x)*sympy.sin(sympy.pi*y)
	u2 = -sympy.sin(sympy.pi*x)*sympy.cos(sympy.pi*y)
	p_sympy = (y-0.5)*sympy.cos(2.0*sympy.pi*x) + (x-0.5)*sympy.sin(2.0*sympy.pi*y)
	f1 = -sympy.diff(sympy.diff(u1, x),x)-sympy.diff(sympy.diff(u1, y),y)+sympy.diff(p_sympy, x)
	f2 = -sympy.diff(sympy.diff(u2, x),x)-sympy.diff(sympy.diff(u2, y),y)+sympy.diff(p_sympy, y)
	f_exact = Expression((sympy.ccode(f1).replace('xx', 'x[0]').replace('yy', 'x[1]'),sympy.ccode(f2).replace('xx', 'x[0]').replace('yy', 'x[1]')),degree=polV,domain=mesh)
	#f_exact = -div(grad(u_exact))+grad(p_exact)
	phi = Expression('-pow((x[0]-0.5)*(x[0]-0.5)+(x[1]-0.5)*(x[1]-0.5),0.5)+R',R=R,degree=polV+6,domain=mesh)
	uD_int = u_exact*(1.0+phi)


	# Dirichlet
	boundary = MeshFunction("size_t", mesh, mesh.topology().dim()-1)
	boundary.set_all(0)
	dirichlet_boundary_ext = DirichletBoundary_ext()
	dirichlet_boundary_int = DirichletBoundary_int()
	dirichlet_boundary_ext.mark(boundary, 1)
	dirichlet_boundary_int.mark(boundary, 2)
	bc1 = DirichletBC(V.sub(0), u_exact, boundary, 1)
	bc2 = DirichletBC(V.sub(0), uD_int, boundary, 2)
	bcs = BlockDirichletBC([bc1,bc2])



	# dx ds
	dx = Measure("dx")(domain = mesh)
	ds = Measure("ds")(domain = mesh,subdomain_data = boundary)


	# Resolution
	trial = BlockTrialFunction(V)
	u, p, l1  = block_split(trial)
	test = BlockTestFunction(V)
	v, q, mu1 = block_split(test)


	# construction of the bilinear form
	a_u_v = 2.0*inner(sym(grad(u)),sym(grad(v)))*dx
	a_p_v = - p*div(v)*dx
	a_real_v = 0
	a_u_q = - q*div(u)*dx
	a_p_q = 0
	a_real_q =l1*q*dx
	a_u_realtest =  0
	a_p_realtest =  mu1*p*dx
	a_real_realtest = 0
	a_u_mu2 = 0
	a_p_mu2 = 0
	a_real_mu2 = 0


	# Construction of the linear form
	l_v = inner(v,f_exact)*dx
	l_q = 0
	l_realtest = 0.0


	# assemble the block
	a = [[ a_u_v,  a_p_v, a_real_v],
	       [ a_u_q,  a_p_q , a_real_q],
	       [a_u_realtest,  a_p_realtest, a_real_realtest]]
	l =  [l_v,l_q,l_realtest]


	# SOLVE 
	A = block_assemble(a)
	L = block_assemble(l)
	bcs.apply(A)
	bcs.apply(L)
	sol = BlockFunction(V)
	print("Assemble the matrices: ok")
	block_solve(A, sol.block_vector(), L,"mumps")
	print("Solve the problem: ok")
	u, p, real =  block_split(sol)


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





# Initialistion of the output
size_mesh_phi_vec = np.zeros(Iter_phi)
error_u_L2_phi_vec = np.zeros(Iter_phi)
error_u_H1_phi_vec = np.zeros(Iter_phi)
error_p_L2_phi_vec = np.zeros(Iter_phi)
cond_phi_vec = np.zeros(Iter_phi)
for i in range(Iter_phi):
	print('##################')
	print('## Iteration phi FEM',i+1,'##')
	print('##################')


	# Construction of the mesh
	N = int(10*2**((i)))
	mesh_macro = RectangleMesh(Point(0.0, 0.0), Point(1.0, 1.0), N, N)
	mesh_macro.init(1,2)
	domains = MeshFunction("size_t", mesh_macro, mesh_macro.topology().dim())
	domains.set_all(0)
	for ind in range(mesh_macro.num_cells()):
		mycell = Cell(mesh_macro,ind)
		v1x,v1y,v2x,v2y,v3x,v3y = mycell.get_vertex_coordinates()
		if Omega(v1x,v1y) or Omega(v2x,v2y) or Omega(v3x,v3y):
			domains[ind] = 1
	mesh = SubMesh(mesh_macro, domains, 1)
	print("num cells mesh :",mesh.num_cells())

	# Construction of phi
	phi_exact = Expression('-pow((x[0]-0.5)*(x[0]-0.5)+(x[1]-0.5)*(x[1]-0.5),0.5)+R',R=R,degree=polV+4,domain=mesh)

	if extrapol == True:
		V_phi = FunctionSpace(mesh,'CG',polPhi)
		V_phi_extra = FunctionSpace(mesh,'CG',polPhi+1)
		phi_temp = interpolate(phi_exact,V_phi)
		phi = Function(V_phi_extra)
		phi.extrapolate(phi_temp)
	else:
		V_phi = FunctionSpace(mesh,'CG',polPhi)
		phi = interpolate(phi_exact,V_phi)

	# computation of the exact solution
	u_exact = Expression(('cos(pi*x[0])*sin(pi*x[1])','-sin(pi*x[0])*cos(pi*x[1])'),domain=mesh,degree=polV+4)
	p_exact = Expression('(x[1]-0.5)*cos(2.0*pi*x[0]) + (x[0]-0.5)*sin(2.0*pi*x[1])',domain=mesh,degree=polV+4)


	# Computation of the source term
	#f_exact = -div(grad(u_exact))+grad(p_exact)
	x, y = sympy.symbols('xx yy')
	u1 = sympy.cos(sympy.pi*x)*sympy.sin(sympy.pi*y)
	u2 = -sympy.sin(sympy.pi*x)*sympy.cos(sympy.pi*y)
	p_sympy = (y-0.5)*sympy.cos(2.0*sympy.pi*x) + (x-0.5)*sympy.sin(2.0*sympy.pi*y)
	f1 = -sympy.diff(sympy.diff(u1, x),x)-sympy.diff(sympy.diff(u1, y),y)+sympy.diff(p_sympy, x)
	f2 = -sympy.diff(sympy.diff(u2, x),x)-sympy.diff(sympy.diff(u2, y),y)+sympy.diff(p_sympy, y)
	f_exact = Expression((sympy.ccode(f1).replace('xx', 'x[0]').replace('yy', 'x[1]'),sympy.ccode(f2).replace('xx', 'x[0]').replace('yy', 'x[1]')),degree=5,domain=mesh)


	# Facets and cells where we apply the ghost penalty
	mesh.init(1,2)
	#vertex_ghost = MeshFunction("size_t", mesh, mesh.topology().dim()-2)
	facet_ghost = MeshFunction("size_t", mesh, mesh.topology().dim()-1)
	cell_ghost = MeshFunction("size_t", mesh, mesh.topology().dim())
	#vertex_ghost.set_all(0)
	facet_ghost.set_all(0)
	cell_ghost.set_all(0)
	for mycell in cells(mesh):
		for myfacet in facets(mycell):
			v1, v2 = vertices(myfacet)
			if phi(v1.point().x(),v1.point().y())*phi(v2.point().x(),v2.point().y())<0 or near(phi(v1.point().x(),v1.point().y())*phi(v2.point().x(),v2.point().y()),0):
				cell_ghost[mycell] = 1
				for myfacet2 in facets(mycell):
					facet_ghost[myfacet2] = 1
				#for myvertex in vertices(mycell):
				#	vertex_ghost[myvertex] = 1


	# for the dirichlet condition
	for myfacet in facets(mesh):
		if facet_ghost[myfacet]==0 and myfacet.exterior():
			facet_ghost[myfacet]=3

	"""count = 0
	for myfacet in facets(mesh):
		if facet_ghost[myfacet]==3 and myfacet.exterior():
			count +=1
	print("count facet",count)

	count = 0
	for mycell in cells(mesh):
		if cell_ghost[mycell]==1:
			count +=1
	print("count cell",count)"""


	plot_mesh = plot(mesh)
	file = File('output/stokes_fixed_mesh.pvd')
	file << mesh
	plt.savefig('output/stokes_fixed_mesh.png')

	#  construction of chi  et uD
	V_chi = FunctionSpace(mesh,'CG',5)
	chi = ChiExpr(element=V_chi.ufl_element())
	chi = interpolate(chi,V_chi)
	uD = u_exact*(1.0+phi)

	"""uD1 = project(uD[0],V_phi)
	u1 = project(u_exact[0],V_phi)
	print("uD",uD1(R+0.5,0.5),u1(R+0.5,0.5))"""

	# Construction of the space
	V_u = VectorFunctionSpace(mesh, 'CG',polV)
	V_p = FunctionSpace(mesh, 'CG',polV-1)
	V_real = FunctionSpace(mesh, 'R',0)
	V = BlockFunctionSpace([V_u,V_p,V_real], restrict=[None,None,None])
	print('dim V',V.dim())


	# Boundary condition
	bc = DirichletBC(V.sub(0), -u_exact,facet_ghost, 3)
	bcs = BlockDirichletBC([bc])


	# dx ds
	dx = Measure("dx")(domain = mesh,subdomain_data = cell_ghost)
	ds = Measure("ds")(domain = mesh,subdomain_data = facet_ghost)


	# Resolution
	trial = BlockTrialFunction(V)
	w, p, l = block_split(trial)
	test = BlockTestFunction(V)
	s, q, mu = block_split(test)


	# construction of the bilinear form
	n = FacetNormal(mesh)
	h = CellDiameter(mesh)

	a_w_s = 2.0*inner(sym(grad(w*phi)),sym(grad(s*phi)))*dx - 2.0*inner(dot(sym(grad(w*phi)),n),phi*s)*ds(1)
	a_p_s = inner(dot(p*Identity(2),n),phi*s)*ds(1) - p*div(phi*s)*dx 
	a_real_s = 0

	a_w_q = - q*div(phi*w)*dx
	a_p_q = 0
	a_real_q = l*q*dx

	a_w_realtest = 0
	a_p_realtest =  mu*p*dx
	a_real_realtest = 0


	# add or not the ghost penalty to the bilinear form
	if ghost==True:
		a_w_s += sigma*avg(h)*inner(jump(grad(phi*w),n),jump(grad(phi*s),n))*dS(1) +sigma*avg(h)**3*inner(jump(grad(grad(phi*w)),n),jump(grad(grad(phi*s)),n))*dS(1) + sigma*h**2*inner(div(grad(phi*w)),div(grad(phi*s)))*dx(1) + sigma*div(phi*w)*div(phi*s)*dx(1)
		a_p_s += -sigma*h**2*inner(div(grad(phi*s)),grad(p))*dx(1)
		a_w_q += sigma*h**2*inner(div(grad(phi*w)),grad(q))*dx(1)
		a_p_q += -sigma*h**2*inner(grad(q),grad(p))*dx(1)


	# Construction of the linear form
	l_s = inner(phi*s,f_exact)*dx - 2.0*inner(sym(grad(uD)),sym(grad(s*phi)))*dx+ 2.0*inner(dot(sym(grad(uD)),n),phi*s)*ds(1)
	l_q = q*div(uD)*dx
	l_realtest = 0


	# add or not the ghost penalty to the linear form
	if ghost==True:
		l_s += - sigma*avg(h)*inner(jump(grad(uD),n),jump(grad(phi*s),n))*dS(1) - sigma*h**2*inner(f_exact,div(grad(phi*s)))*dx(1)-sigma*h**2*inner(div(grad(uD)),div(grad(phi*s)))*dx(1) - sigma*div(uD)*div(phi*s)*dx(1)- sigma*avg(h)*inner(jump(grad(grad(uD)),n),jump(grad(grad(phi*s)),n))*dS(1)
		l_q += -sigma*h**2*inner(f_exact,grad(q))*dx(1)-sigma*h**2*inner(div(grad(uD)),grad(q))*dx(1)


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
	u = phi*u + uD


	"""mesh2 = refine(mesh)
	dx2 = Measure("dx")(domain = mesh2)
	V_u_2 = VectorFunctionSpace(mesh2, 'CG',polV+1)
	V_p_2 = FunctionSpace(mesh2, 'CG',polV)"""

	p_exact = p_exact - assemble(p_exact*dx)
	
	# Computation of the error
	norm_L2_u_exact =assemble(inner(u_exact,u_exact)*dx)**0.5
	norm_H1_u_exact =assemble(grad(u_exact)**2*dx)**0.5
	norm_L2_p_exact =assemble(p_exact**2*dx)**0.5
	err_L2 = assemble((u-u_exact)**2*dx)**0.5/norm_L2_u_exact
	err_H1 = assemble((grad(u-u_exact))**2*dx)**0.5/norm_H1_u_exact
	err_p = assemble(((p-p_exact))**2*dx)**0.5/norm_L2_p_exact
	size_mesh_phi_vec[i] = mesh_macro.hmax()
	error_u_L2_phi_vec[i] = err_L2
	error_u_H1_phi_vec[i] = err_H1
	error_p_L2_phi_vec[i] = err_p
	print('h :',mesh_macro.hmax())
	print('relative L2 error u : ',err_L2)
	print('relative H1 error u : ',err_H1)
	print('relative L2 error p : ',err_p)	
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
plt.savefig('output/stokes_fixed_error_u_L2.png')
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
plt.savefig('output/stokes_fixed_error_u_H1.png')
plt.clf()
plot_sol = plt.loglog(size_mesh_standard_vec,error_p_L2_standard_vec,"o-",label="L2 Error p standard")
plot_sol = plt.loglog(size_mesh_phi_vec,error_p_L2_phi_vec,"o-",label="L2 Error p phi FEM")
x = [size_mesh_standard_vec[0],size_mesh_standard_vec[-1]]
y = [error_u_H1_standard_vec[0],error_u_H1_standard_vec[0]*size_mesh_standard_vec[-1]**2*size_mesh_standard_vec[0]**(-2)]
plot_sol = plt.loglog(x,y,label="h^2")
plt.legend()
plt.savefig('output/stokes_fixed_error_p.png')


#  Write the output file for latex
if ghost == False:
	f = open('output/output_stokes_fixed_no_ghost.txt','w')
if ghost == True:
	f = open('output/output_stokes_fixed_ghost.txt','w')
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


# Plot and save
if Plot == True:
	sol = project(u_h*phi,V)
	plot_sol = plot(sol)
	file = File('output/stokes_fixed_solution.pvd')
	file << sol
	plt.savefig('output/stokes_fixed_solution.png')
