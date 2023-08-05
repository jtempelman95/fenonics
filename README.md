# fenonics
Codes for dispersion computation wtih fenicsx


Solving Bloch diserpersion problems required several steps


This software provides a free open-souce platform for solving 2D dispersion problems. 

The gmsh api is used to create a square mesh with an inclusion. The geometry of the inclusion is defined by a 'design vector'. This is an set of radii corresopnding to radially even spaced coordinates in a sub-quadrant of the domain. 

These points are copied across quadrants of the unit cell and a periodic spline curve is fit to them. This maps the geometry of the inclusion.


The complex Bloch eigenproblem is constructed by inserting a Bloch ansatz into the variational form and then utilizeing the dolfinx-mpc module to apply stard periodic boundary conditions.

As of now, the code only supports the wave equation,

$$
\rho\ddot{u} = T\nabla^2u
$$

with Bloch-periodic BCs

$$ \begin{align} &u(0,y) = u(a,y)e^{i k_xa} \ \ \text{on} \ 0 < y < a \\
&u(x,0) = u(x,a)e^{i k_ya} \ \ \text{on} \ 0 < x < a\end{align}
$$

## Brief summary of workflow

1. The `gmsh-api` module generates a mesh parameterized by an internal void geometry
2. `dolfinx` builds a function space on the mesh based on a Bloch-ansatz weak form
4. `dolfinx-mpc`applies the necessary periodc BCs and returns the mass and stiffness matrices
5. `scipy` solves the complex eigenproblem at a given wavenumber
6. `fenonics` packages all the operations to conviently solve over an IBZ, post-process solutions, and visualization results
