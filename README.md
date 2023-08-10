# fenonics
## *Codes for dispersion computation wtih `fenicsx`*

This software provides a free open-souce platform for solving 2D dispersion problems.  As of now, the code only supports the wave equation,

$$
\rho\ddot{u} = T\nabla^2u
$$

with Bloch-periodic BCs

$$ \begin{align} &u(0,y) = u(a,y)e^{i k_xa} \ \ \text{on} \ 0 < y < a \\
&u(x,0) = u(x,a)e^{i k_ya} \ \ \text{on} \ 0 < x < a\end{align}
$$

Future builds will include elasticity problems.

The `gmsh api` is used to create a square mesh with an inclusion paramterized by a spline curve. The geometry of the inclusion is created using a `gmsh` module provided herein. Next, the `gmsh` module produces a mesh of the geometry, which is then passed to `dolfinx` and `dolfinx-mpc` to construct the appropriate boundary conditions and weak forms for solving the dispersion problem. In the real build version, the Bloch-ansatz is insereted directly into the weak form, and the weak form is then split into real and imagainary components before being assembled. The complex eigenproblem is then solved using either `scipy` or `Petsc`, depending on what build of `dolfinx` you have installed.

## Brief summary of workflow

1. The `gmsh-api` module generates a mesh parameterized by an internal void geometry
2. `dolfinx` builds a function space on the mesh based on a Bloch-ansatz weak form
4. `dolfinx-mpc`applies the necessary periodc BCs and returns the mass and stiffness matrices
5. `scipy` solves the complex eigenproblem at a given wavenumber
6. `fenonics` packages all the operations to conviently solve over an IBZ, post-process solutions, and visualization results

![image](https://github.com/jtempelman95/fenonics/assets/107128967/5592a872-b564-48a9-a8ae-97352910895c)


## Instructions to download

1. Clone the repo `git@github.com:jtempelman95/fenonics.git`
2. `cd` to the root directory of the repo on your file path
3. run `pip install .`

#### Enviorments
You can run the codes using either the real or comlpex builds of `dolfinx`
We've provided a `enviornment_real.yml` and `enviornment_cmpx.yml` for either enviornment. 

## Instructions to use

  - To see the step-by-step processes for computing the band strucutre per steps 1-5 of the worflow, open `demos/Tuorial_SolveBands`
  - To see a tutorial on how to use the `fenonics` module to solve disperion problems, open `demos/Tutorial_fenonics`
