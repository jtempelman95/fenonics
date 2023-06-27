


"""
    
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                
                    MESH GENERATION WITH GMSH
                
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                
                
    Josh Tempelman
    Universrity of Illinois
    jrt7@illinois.edu
    
    Originated      MARCH 10, 2023
    Last Modified   MARCH 24, 2023
    
    ! ! Do not distribute ! !
    
    
    About:
        This program defines meshing functions that take inputs
        in the form of design vectors and constructs a meshed
        domain based on its arguments.
    
    
        
"""



#%%
# ------------------
# Dependencies
# ------------------
import  gmsh
import  sys
import  numpy as np
import  matplotlib.pyplot as plt
import  timeit
import  matplotlib.pyplot as plt
from    mpi4py import MPI
from    dolfinx.io import gmshio
from    mpi4py import MPI
import  pyvista
pyvista.start_xvfb()
from dolfinx.plot import create_vtk_mesh

#%% function


def get_mesh_SquareSpline(a,da,r,Nquads,offset,meshalg,refinement_level,refinement_dist, isrefined = True ,cut = True):
    """
    arguments
        isrefined           -   decide to refine around interior edges
        cut                 -   Decide of interor geometry is cut-out or inclusion
        Nquad               - number of quadrants to repeat geometry about
        a                   - unit cell lenght
        da                  - nominal mesh spacing
        r                   - vector of radii
        offset              - ofset angle
        meshalg             - wich algorithm to use in gmsh
        refinement_level    - how much denser refined geomtety is
    """
    gmsh.initialize()
    gmsh.clear()
    # gmsh.option.setNumber("Geometry.Tolerance", 1e-6/4.48401)
    # a - > Charecteristic Length of the unit cell
    # r - > vector of inputs to parameterize the mesh (spline)
    # Nquads - > Number of quadrants to rotate about

    p1 = gmsh.model.occ.addPoint(0,  0,   0,tag = 1)
    p2 = gmsh.model.occ.addPoint(a,  0,   0,tag = 2)
    p3 = gmsh.model.occ.addPoint(a,  a ,  0,tag = 3)
    p4 = gmsh.model.occ.addPoint(0,  a,   0,tag = 4)
    
    # Adding curves defining outter perimeter
    l1 = gmsh.model.occ.addLine(1, 2, tag = 1) # Bottom wall
    l2 = gmsh.model.occ.addLine(2, 3, tag = 2) # Right wall
    l3 = gmsh.model.occ.addLine(3, 4, tag = 3) # Top wall
    l4 = gmsh.model.occ.addLine(4, 1, tag = 4) # Left wall
    gmsh.model.occ.synchronize()

    # Create the outter box
    cell_boundary = gmsh.model.occ.addCurveLoop([l1, l2, l3, l4],  tag = 1)
    gmsh.model.occ.synchronize()
    # Create a surface from the outter box
    cell_surface  = gmsh.model.occ.addPlaneSurface([cell_boundary], tag = 1) 
    surf = [(2, cell_surface)]
    gmsh.model.occ.synchronize()

    # Add the spline curve
    xspl = []
    yspl = []
    idx = 0
    if Nquads % 2 == 0:
        shift = 2*np.pi/len(r)/Nquads/2
    else:
        shift = 0
    for quad in range(Nquads):
        for k in range(len(r)):
            xc = a/2 + r[k]/2   * np.cos( 2*k*np.pi/len(r)/Nquads + 2*quad*np.pi/Nquads+shift       + offset)
            yc = a/2+ r[k]/2    * np.sin( 2*k*np.pi/len(r)/Nquads + 2*quad*np.pi/Nquads+shift         + offset)
            yspl.append(yc)
            xspl.append(xc)
            idx+=1
            # print(2*k*np.pi/len(r)/Nquads + 2*quad*np.pi/Nquads+shift)
        if Nquads % 2 == 0:
            r =  np.flip(r)
    
    # xspl.append(xspl[0])
    # yspl.append(yspl[0])
    splinepoints = [ gmsh.model.occ.addPoint(xspl[v], yspl[v],0 ) for v in range(len(xspl))]
    splinepoints.append(splinepoints[0])
    spline1 = gmsh.model.occ.add_bspline(splinepoints ,tag = 125)
   
    
    # Create a surface from the spline
    spline_boundary = gmsh.model.occ.addCurveLoop([spline1],  tag = 100)
    spline_surf = gmsh.model.occ.addPlaneSurface( [spline_boundary] , tag = 100) 
    gmsh.model.occ.synchronize()
    
    
    

    
    if cut == True:
        # Fuse the spline surface to the unit cell
        all_surfaces = [(2,spline_surf)]
        out, whole_domain = gmsh.model.occ.cut([(2, cell_surface)], all_surfaces)
        gmsh.model.occ.synchronize()
        
        other_surfaces = []
        tag = 1;
        for domain in whole_domain[0]:
            com     = gmsh.model.occ.getCenterOfMass(domain[0], domain[1])
            mass    = gmsh.model.occ.getMass(domain[0], domain[1])
            disk1   = gmsh.model.addPhysicalGroup(domain[0], [domain[1]], tag=tag)
            tag+=1
        other_surfaces.append(whole_domain[0][0])
    
    else:
        # Fuse the spline surface to the unit cell
        all_surfaces = [(2,spline_surf)]
        out, whole_domain = gmsh.model.occ.fragment([(2, cell_surface)], all_surfaces)
        gmsh.model.occ.synchronize()
        
        other_surfaces = []
        tag = 1;
        for domain in whole_domain[0]:
            com     = gmsh.model.occ.getCenterOfMass(domain[0], domain[1])
            mass    = gmsh.model.occ.getMass(domain[0], domain[1])
            disk1   = gmsh.model.addPhysicalGroup(domain[0], [domain[1]], tag=tag)
            tag+=1
        other_surfaces.append(whole_domain[1][0])

    
     # set mesh size
    # First remove anythng not in bounding box
    eps = 1e-3
    gmsh.option.setNumber("Geometry.OCCBoundsUseStl", 1)
    vin = gmsh.model.getEntitiesInBoundingBox(  -eps, -eps,  0,  
                                            a+eps, a+eps, 0,
                                            2)
    
    p = gmsh.model.getBoundary(vin, False, False, True)  # Get all points
    gmsh.model.mesh.setSize(p, da)
    p = gmsh.model.getEntitiesInBoundingBox(2 - eps, -eps, -eps, 2 + eps, eps, eps,1)
    gmsh.model.mesh.setSize(p, da)
    
    
    # Mesh refnement around the inclusion
    if isrefined == True:
        if cut == True:
            gmsh.model.mesh.field.add("Distance", 1)
            edges = gmsh.model.getBoundary(other_surfaces, oriented=True)
            gmsh.model.mesh.field.setNumbers(1, "CurvesList", [5])
            gmsh.model.mesh.field.setNumber(1, "Sampling", 500)
            gmsh.model.mesh.field.add("Threshold", 2)
            gmsh.model.mesh.field.setNumber(2, "IField", 1)
            gmsh.model.mesh.field.setNumber(2, "LcMin", da/refinement_level)
            gmsh.model.mesh.field.setNumber(2, "LcMax", da*1)
            gmsh.model.mesh.field.setNumber(2, "DistMin", .025 * a)
            gmsh.model.mesh.field.setNumber(2, "DistMax", refinement_dist)
            gmsh.model.mesh.field.add("Min", 5)
            gmsh.model.mesh.field.setNumbers(5, "FieldsList", [2])
            gmsh.model.mesh.field.setAsBackgroundMesh(5)
        else:
            gmsh.model.mesh.field.add("Distance", 1)
            edges = gmsh.model.getBoundary(other_surfaces, oriented=True)
            gmsh.model.mesh.field.setNumbers(1, "EdgesList", [e[1] for e in edges])
            gmsh.model.mesh.field.add("Threshold", 2)
            gmsh.model.mesh.field.setNumber(2, "IField", 1)
            gmsh.model.mesh.field.setNumber(2, "LcMin", da/refinement_level)
            gmsh.model.mesh.field.setNumber(2, "LcMax", da*1)
            gmsh.model.mesh.field.setNumber(2, "DistMin", .025 * a)
            gmsh.model.mesh.field.setNumber(2, "DistMax", refinement_dist)
            gmsh.model.mesh.field.add("Min", 5)
            gmsh.model.mesh.field.setNumbers(5, "FieldsList", [2])
            gmsh.model.mesh.field.setAsBackgroundMesh(5)

    

    
    # Add in periodic constraint
    print(gmsh.model.getEntities(dim=1))
    left  = []
    right = []
    kj = 0
    for line in gmsh.model.getEntities(dim=1):
            
            com = gmsh.model.occ.getCenterOfMass(line[0], line[1])
            
            # Add the exterior and interior edges as a physical groups
            gmsh.model.addPhysicalGroup(1,[line[1]],line[1])
            
            if np.isclose(com[0], 0):
                left.append(line[1])
                
            if np.isclose(com[0], a):
                right.append(line[1])


    # gmsh.model.mesh.setSize([(1,5)], da/10)
    
    # print('found left lines is:'+str(left))
    # print('defined left lines is:'+str(left))
    print(' --------- ERROR HERE --------------')
    # wall_ents = gmsh.model.getEntitiesForPhysicalGroup(2, walls)
    translation = [1, 0, 0, a,   0, 1, 0, 0, 0,   0, 1, 0, 0,  0, 0, 1]
    translationy = [1, 0, 0, 0,   0, 1, 0, a, 0,   0, 1, 0, 0,  0, 0, 1]
    # gmsh.model.mesh.setPeriodic(1, left, right, translation)
    gmsh.model.mesh.setPeriodic(1, [l4],[l2], translation)
    gmsh.model.mesh.setPeriodic(1, [l1],[l3], translationy)
    gmsh.model.occ.synchronize()
    print(' --------- ERROR HERE --------------')
    
    mesh_dim = 2
    gmsh.option.setNumber("Mesh.Algorithm", meshalg)
    gmsh.model.mesh.generate(mesh_dim) 
        
    print('to here')
    return gmsh.model, xspl, yspl


    


##################################################################################################
def get_mesh_SquareMultiSpline(a,da,r,Nquads,offset,meshalg,refinement_level,refinement_dist, isrefined = True ,cut = True):
    gmsh.initialize()
    gmsh.clear()
    # gmsh.option.setNumber("Geometry.Tolerance", 1e-6/4.48401)
    # a - > Charecteristic Length of the unit cell
    # r - > vector of inputs to parameterize the mesh (spline)
    # Nquads - > Number of quadrants to rotate about
        
    p1 = gmsh.model.occ.addPoint(0,  0,   0,tag = 1)
    p2 = gmsh.model.occ.addPoint(a,  0,   0,tag = 2)
    p3 = gmsh.model.occ.addPoint(a,  a ,  0,tag = 3)
    p4 = gmsh.model.occ.addPoint(0,  a,   0,tag = 4)
    
    # Adding curves defining outter perimeter
    l1 = gmsh.model.occ.addLine(1, 2, tag = 1) # Bottom wall
    l2 = gmsh.model.occ.addLine(2, 3, tag = 2) # Right wall
    l3 = gmsh.model.occ.addLine(3, 4, tag = 3) # Top wall
    l4 = gmsh.model.occ.addLine(4, 1, tag = 4) # Left wall
    gmsh.model.occ.synchronize()

    # Create the outter box
    cell_boundary = gmsh.model.occ.addCurveLoop([l1, l2, l3, l4],  tag = 1)
    gmsh.model.occ.synchronize()
    # Create a surface from the outter box
    cell_surface  = gmsh.model.occ.addPlaneSurface([cell_boundary], tag = 1) 
    surf = [(2, cell_surface)]
    gmsh.model.occ.synchronize()
    

    # Add the spline curve

    tagsurf = 100
    sft= 0
    inclusion_idx = 0
    for ii in np.array([     .25,    .75]):
        for jj in np.array([ .25,    .75]):
            xspl = []
            yspl = []
            for quad in range(Nquads):
                for k in range(len(r[0])):
                    rd = np.random.rand()*0
                    xc = ii*a  + (r[inclusion_idx][k])  * np.cos( 2*k*np.pi/len(r)/Nquads + 2*quad*np.pi/Nquads + offset + sft*np.pi/8 )
                    yc = jj*a  + (r[inclusion_idx][k])  * np.sin( 2*k*np.pi/len(r)/Nquads + 2*quad*np.pi/Nquads + offset + sft*np.pi/8 )
                    yspl.append(yc)
                    xspl.append(xc)
            sft += 1
            splinepoints = [ gmsh.model.occ.addPoint(xspl[v], yspl[v],0 ) for v in range(len(xspl))]
            splinepoints.append(splinepoints[0])
            gmsh.model.occ.add_bspline(splinepoints ,tag = tagsurf)
            print(tagsurf)
            tagsurf+=1
            inclusion_idx += 1
    
    # Create a surface from the spline
    spline_boundary1 = gmsh.model.occ.addCurveLoop([100],  tag = 100)
    spline_boundary2 = gmsh.model.occ.addCurveLoop([101],  tag = 101)
    spline_boundary3 = gmsh.model.occ.addCurveLoop([102],  tag = 102)
    spline_boundary4 = gmsh.model.occ.addCurveLoop([103],  tag = 103)
    spline_surf1     = gmsh.model.occ.addPlaneSurface( [spline_boundary1], tag = 100) 
    spline_surf2     = gmsh.model.occ.addPlaneSurface( [spline_boundary2], tag = 101) 
    spline_surf3     = gmsh.model.occ.addPlaneSurface( [spline_boundary3], tag = 102) 
    spline_surf4     = gmsh.model.occ.addPlaneSurface( [spline_boundary4], tag = 103) 
    gmsh.model.occ.synchronize()
    
    
    if cut == True:
        # Fuse the spline surface to the unit cell
        all_surfaces = [(2,spline_surf1)]
        all_surfaces.append([2,spline_surf2])
        all_surfaces.append([2,spline_surf3])
        all_surfaces.append([2,spline_surf4])
        out, whole_domain = gmsh.model.occ.cut([(2, cell_surface)], all_surfaces)
        gmsh.model.occ.synchronize()
        
        other_surfaces = []
        tag = 1;
        for domain in whole_domain[0]:
            com     = gmsh.model.occ.getCenterOfMass(domain[0], domain[1])
            mass    = gmsh.model.occ.getMass(domain[0], domain[1])
            disk1   = gmsh.model.addPhysicalGroup(domain[0], [domain[1]], tag=tag)
            tag+=1
        other_surfaces.append(whole_domain[0][0])
    
    else:
        # Fuse the spline surface to the unit cell
        all_surfaces = [(2,spline_surf1)]
        all_surfaces.append([2,spline_surf2])
        all_surfaces.append([2,spline_surf3])
        all_surfaces.append([2,spline_surf4])
        out, whole_domain = gmsh.model.occ.fragment([(2, cell_surface)], all_surfaces)
        gmsh.model.occ.synchronize()
        
        other_surfaces = []
        tag = 1;
        for domain in whole_domain[0]:
            com     = gmsh.model.occ.getCenterOfMass(domain[0], domain[1])
            mass    = gmsh.model.occ.getMass(domain[0], domain[1])
            disk1   = gmsh.model.addPhysicalGroup(domain[0], [domain[1]], tag=tag)
            tag+=1
        # other_surfaces.append(whole_domain[0][1])
        other_surfaces.append(whole_domain[1][0])
        other_surfaces.append(whole_domain[2][0])
        other_surfaces.append(whole_domain[3][0])
        other_surfaces.append(whole_domain[4][0])

    
     # set mesh size
    # First remove anythng not in bounding box
    eps = 1e-3
    gmsh.option.setNumber("Geometry.OCCBoundsUseStl", 1)
    vin = gmsh.model.getEntitiesInBoundingBox(  -eps, -eps,  0,  
                                            a+eps, a+eps, 0,
                                            2)
    
    p = gmsh.model.getBoundary(vin, False, False, True)  # Get all points
    gmsh.model.mesh.setSize(p, da)
    p = gmsh.model.getEntitiesInBoundingBox(2 - eps, -eps, -eps, 2 + eps, eps, eps,1)
    gmsh.model.mesh.setSize(p, da)
    
    
    # Mesh refnement around the inclusion
    if isrefined == True:
        if cut == True:
            gmsh.model.mesh.field.add("Distance", 1)
            edges = gmsh.model.getBoundary(other_surfaces, oriented=True)
            gmsh.model.mesh.field.setNumbers(1, "CurvesList", [5,6,7,8])
            gmsh.model.mesh.field.add("Threshold", 2)
            gmsh.model.mesh.field.setNumber(2, "IField", 1)
            gmsh.model.mesh.field.setNumber(2, "LcMin", da/refinement_level)
            gmsh.model.mesh.field.setNumber(2, "LcMax", da*1)
            gmsh.model.mesh.field.setNumber(2, "DistMin", .025 * a)
            gmsh.model.mesh.field.setNumber(2, "DistMax", refinement_dist)
            gmsh.model.mesh.field.add("Min", 5)
            gmsh.model.mesh.field.setNumbers(5, "FieldsList", [2])
            gmsh.model.mesh.field.setAsBackgroundMesh(5)
        else:
            gmsh.model.mesh.field.add("Distance", 1)
            edges = gmsh.model.getBoundary(other_surfaces, oriented=True)
            gmsh.model.mesh.field.setNumbers(1, "EdgesList", [e[1] for e in edges])
            gmsh.model.mesh.field.add("Threshold", 2)
            gmsh.model.mesh.field.setNumber(2, "IField", 1)
            gmsh.model.mesh.field.setNumber(2, "LcMin", da/refinement_level)
            gmsh.model.mesh.field.setNumber(2, "LcMax", da*1)
            gmsh.model.mesh.field.setNumber(2, "DistMin", .025 * a)
            gmsh.model.mesh.field.setNumber(2, "DistMax", refinement_dist)
            gmsh.model.mesh.field.add("Min", 5)
            gmsh.model.mesh.field.setNumbers(5, "FieldsList", [2])
            gmsh.model.mesh.field.setAsBackgroundMesh(5)


  
    # Add in periodic constraint
    print(gmsh.model.getEntities(dim=1))
    left  = []
    right = []
    for line in gmsh.model.getEntities(dim=1):
            com = gmsh.model.occ.getCenterOfMass(line[0], line[1])
            if np.isclose(com[0], 0):
                left.append(line[1])
                
            if np.isclose(com[0], a):
                right.append(line[1])
        
    print(' --------- ERROR HERE --------------')
    # wall_ents = gmsh.model.getEntitiesForPhysicalGroup(2, walls)
    translation = [1, 0, 0, a,   0, 1, 0, 0, 0,   0, 1, 0, 0,  0, 0, 1]
    translationy = [1, 0, 0, 0,   0, 1, 0, a, 0,   0, 1, 0, 0,  0, 0, 1]
    # gmsh.model.mesh.setPeriodic(1, left, right, translation)
    # gmsh.model.mesh.setPeriodic(1, [l2],[l4], translation)
    # gmsh.model.mesh.setPeriodic(1, [l1],[l3], translationy)
    gmsh.model.occ.synchronize()
    print(' --------- ERROR HERE --------------')
    
            

    

    mesh_dim = 2
    gmsh.option.setNumber("Mesh.Algorithm", meshalg)
    gmsh.model.mesh.generate(mesh_dim) 
        
    print('Gmsh model created: Multi-spline class')
    return gmsh.model



    # Test
"""
Test the funtoin
"""

if __name__ == '__main__':


    # ################################################## #
    # Imports for finite element modeling                #
    # ################################################## #
    from mpi4py import MPI
    import dolfinx_mpc
    import dolfinx
    from dolfinx.fem    import Function, FunctionSpace, VectorFunctionSpace
    from dolfinx.mesh   import create_unit_square 
    from dolfinx.mesh   import create_rectangle
    from dolfinx.fem    import form
    from dolfinx.fem.petsc import assemble_matrix
    from dolfinx    import plot
    from dolfinx.io import gmshio
    from mpi4py     import MPI
    from dolfinx    import fem
    from ufl        import TrialFunction, TestFunction, grad, dx, dot, nabla_div, Identity, nabla_grad, inner, sym
    from petsc4py.PETSc import ScalarType
    from slepc4py   import SLEPc
    from petsc4py   import PETSc
    from typing     import List
    import scipy.sparse
    from scipy.sparse.linalg import eigsh
    from scipy import sparse
    # ################################################## #
    # Imports for plotting                               #
    # ################################################## #
    import matplotlib.pyplot as plt
    from dolfinx.io import gmshio
    import pyvista
    pyvista.start_xvfb()
    # ################################################## #
    # Imports for the meshing                            #
    # ################################################## #
    import gmsh
    from dolfinx.plot       import create_vtk_mesh
    from dolfinx.io         import XDMFFile
    from dolfinx.io.gmshio  import model_to_mesh




    # =================================
    # Get latin hypercube sampling
    # =================================
    from scipy.stats import qmc
    LHS_Seed    = 1
    sampler     = qmc.LatinHypercube(d=3,seed =  LHS_Seed)
    Nsamp       = 250
    sample      = sampler.random(n=Nsamp)
    sidx        = 6

    ######################################################################
    #                      Domain  Inputs                                #
    ######################################################################
    #   a_len               Charecterstic unit cell length (primitive basis)
    #   r                   vector of points to fit spline to
    #   Nquads              Number of quadrants to repeat spline through (radially)
    #   offset              Offset angle for inclusion geometry
    #   iscut               Choose if inclusion is void or an added domain
    #   dvar                Design variable to probe 
    ######################################################################


    a_len       =   .1
    dvar        =   1/2       
    r           =   np.array([1,dvar,.2,.8,.3])*a_len*.95
    r           =   np.array([1,np.random.rand(1)[0],.3])*a_len*.95
    r           =   sample[sidx,:]*a_len*.9
    # r           = np.hstack((r,np.flip(r)))
    # r           = np.hstack((r,r[0]))
    # r           = np.array([1,.5,.4,.5])*a_len*.95
    dvec        =   r
    offset      =   0*np.pi
    design_vec  =   np.concatenate( (r/a_len, [offset] ))
    Nquads      =   8

    # Set file name to save figures based on input vector
    name = 'dvec'
    for k in range(len(design_vec)):
        name += '_' + str(int(100*np.round(design_vec[k],2)))


    ######################################################################
    #                  Phsyical  Params                                  #
    ######################################################################
    #   c                   Speed of sound in media
    #   rho                 Desnity
    ######################################################################
    c           = [1500,5100]   # if solid inclusion (mutlple materail model)
    rho         = [1e3,7e3]     # if solid inclusion (mutlple materail model) 
    c           = [30]          # if void inclusion  (if iscut)
    rho         = [1.2]         # if void inclusion  (if iscut)

    ######################################################################
    #                      Mesh Inputs                                   #
    ######################################################################
    #   refinement_level    Choose ratio of how dense mesh gets around refinement field 
    #   refinement_dist     Maximum distance of refinement field from refined edges
    #   isrefined           Choose whether or not to refine mesh around internal edges
    #   meshalg             Meshing algorithm for gmsh to use
    #   da                  Nominal Mesh Density
    ######################################################################
    da                  =   a_len/13
    meshalg             =   6
    refinement_level    =   4
    refinement_dist     =   a_len/10

    ######################################################################
    #                        Solver inputs                               #
    ######################################################################
    # npi       Number of points to loop through i-th cut of IBZ
    # nvec      Number of eigenvectrs to solve for in each step
    # fspace    Function space to use
    ######################################################################
    np1     = 20
    np2     = 20
    np3     = 20
    nvec    = 20
    fspace  = 'CG'

    ######################################################################
    #                  Generate a mesh with one inclsuion                #
    ######################################################################
    meshalg                 = 6
    gmsh.model, xpt, ypt    = get_mesh_SquareSpline(a_len ,da,r,Nquads,offset,meshalg,
                                                    refinement_level,refinement_dist,
                                                    isrefined = True, cut = True)

    #################################################################
    #            Import to dolfinx and save as xdmf                 #
    #################################################################
    mesh_comm       = MPI.COMM_WORLD
    model_rank      = 0
    mesh, ct, _     = model_to_mesh(gmsh.model, mesh_comm, model_rank, gdim=2)
    with XDMFFile(MPI.COMM_WORLD, "mt.xdmf", "w") as xdmf:
        xdmf.write_mesh(mesh)
        xdmf.write_meshtags(ct)


    #################################################################
    #              Plot the mesh                                    #
    #################################################################
    V = FunctionSpace(mesh,(fspace,1))
    v = Function(V)
    plotter = pyvista.Plotter()
    grid = pyvista.UnstructuredGrid(*create_vtk_mesh(mesh, mesh.topology.dim))
    num_local_cells = mesh.topology.index_map(mesh.topology.dim).size_local
    grid.cell_data["Marker"] = 1-ct.values[ct.indices<num_local_cells]
    grid.set_active_scalars("Marker")
    actor = plotter.add_mesh(grid, show_edges=True, line_width= 3, edge_color= 'k', style='wireframe')
    plotter.set_background('white', top='white')
    plotter.view_xy()
    plotter.camera.tight(padding=0.1)
    plotter.add_title('CELLS:'+str(ct.values.shape[0])+ '  | NODES:'+str(v.vector[:].shape[0]),color = 'r')
    plotter.show()



    # ###########################################################
    # # Get spline fit from the mesh 
    # ###########################################################
    node_interior = gmsh.model.mesh.getNodesForPhysicalGroup(1,5)[1]
    x_int = node_interior[0::3]
    y_int = node_interior[1::3]
    x_int = np.concatenate([x_int,[x_int[0]]])
    y_int = np.concatenate([y_int,[y_int[0]]])
    # plt.plot(x_int,y_int,'-')
    xi = x_int - a_len/2
    yi = y_int - a_len/2

    plt.plot(np.array(xpt)- a_len/2, np.array(ypt)- a_len/2,'.')
    plt.plot(xi,yi)
    plt.grid()
    plt.show()

    '''
    # Testing sensitivity to mesh options
    '''

    np.random.seed(24)
    r           =   np.array([1,np.random.rand(1)[0],.3])*a_len*.95
    r = np.random.rand(4)*a_len*.95


    gmsh.model, xpt, ypt    = get_mesh_SquareSpline(a_len ,a_len/100,r,Nquads,offset,meshalg,
                                                    refinement_level,refinement_dist,
                                                    isrefined = True, cut = True)
    # ###########################################################
    # # Get spline fit from the mesh 
    # ###########################################################
    node_interior = gmsh.model.mesh.getNodesForPhysicalGroup(1,5)[1]
    x_int = node_interior[0::3]
    y_int = node_interior[1::3]
    x_int = np.concatenate([x_int,[x_int[0]]])
    y_int = np.concatenate([y_int,[y_int[0]]])
    # plt.plot(x_int,y_int,'-')
    xi = x_int - a_len/2
    yi = y_int - a_len/2
    Xi = xi
    Yi = yi
    # plt.plot(np.array(xpt)- a_len/2, np.array(ypt)- a_len/2,'.')
    Nquads = 16


    plt.figure(figsize = (10,10))
    plt.plot(xi,yi,color = 'k',linewidth = 2.5, label = str(gmsh.model.mesh.getNodes()[0].shape[0]))
        


    da                  =   a_len/15
    meshalg             =   6
    refinement_level    =   4
    refinement_dist     =   a_len/25
    meshalg                 = 6
    for refinement_level in np.linspace(2,6,3):
        gmsh.model, xpt, ypt    = get_mesh_SquareSpline(a_len ,da,r,Nquads,offset,meshalg,
                                                        refinement_level,refinement_dist,
                                                        isrefined = True, cut = True)
        # ###########################################################
        # # Get spline fit from the mesh 
        # ###########################################################
        node_interior = gmsh.model.mesh.getNodesForPhysicalGroup(1,5)[1]
        x_int = node_interior[0::3]
        y_int = node_interior[1::3]
        x_int = np.concatenate([x_int,[x_int[0]]])
        y_int = np.concatenate([y_int,[y_int[0]]])
        # plt.plot(x_int,y_int,'-')
        xi = x_int - a_len/2
        yi = y_int - a_len/2

        # plt.plot(np.array(xpt)- a_len/2, np.array(ypt)- a_len/2,'.')
        plt.plot(xi,yi,'-',linewidth=1.5, label = str(gmsh.model.mesh.getNodes()[0].shape[0]))
    plt.grid()
    plt.legend()


    ################################################################
    #        Import to dolfinx and save as xdmf                 #
    ################################################################
    mesh_comm       = MPI.COMM_WORLD
    model_rank      = 0
    mesh, ct, _     = model_to_mesh(gmsh.model, mesh_comm, model_rank, gdim=2)
    with XDMFFile(MPI.COMM_WORLD, "mt.xdmf", "w") as xdmf:
        xdmf.write_mesh(mesh)
        xdmf.write_meshtags(ct)


    #################################################################
    #              Plot the mesh                                    #
    #################################################################
    V = FunctionSpace(mesh,(fspace,1))
    v = Function(V)
    plotter = pyvista.Plotter()
    grid = pyvista.UnstructuredGrid(*create_vtk_mesh(mesh, mesh.topology.dim))
    num_local_cells = mesh.topology.index_map(mesh.topology.dim).size_local
    grid.cell_data["Marker"] = 1-ct.values[ct.indices<num_local_cells]
    grid.set_active_scalars("Marker")
    actor = plotter.add_mesh(grid, show_edges=True, line_width= 3, edge_color= 'k', style='wireframe')
    plotter.set_background('white', top='white')
    plotter.view_xy()
    plotter.camera.tight(padding=0.1)
    plotter.add_title('CELLS:'+str(ct.values.shape[0])+ '  | NODES:'+str(v.vector[:].shape[0]),color = 'r')
    plotter.show()