'''
Crystal Finder

"Crystal Growth in Polyethylene by Molecular Dynamics: The Crystal Edge and Lame
llar Thickness" Macromolecules 51, 13, 2018
https://doi.org/10.1021/acs.macromol.8b00857

See for more info:

Ryan Davidson
https://scrunts23.medium.com/dbscan-algorithm-from-scratch-in-python-475b82e0571c

Erik Lindernoren
https://github.com/eriklindernoren/ML-From-Scratch

Jake VanderPlas
https://gist.github.com/jakevdp/5216193

Tuukka Verho
https://doi.org/10.1021/acs.macromol.8b00857
'''


class CrystalFinder(object):
    def __init__(self, filename, polymers, monomers,
                 verho_coeff=7, verho_cutoff=0.9, min_length=8,
                 method='BallTree', eps=1.1, lam=0.9, leaf_size=30, min_pts=8,
                 second_filt=False, crystalmin=100,
                 meshing=False, mesh_type='gauss', cntpresent=False):

        ####################################
        # Validate File and basic data info
        ####################################
        if isinstance(filename, str):
            self.filename = filename
        else:
            raise ValueError("Filename is not a string")

        # remove .dump
        self.prefix = filename[0:len(filename)-5]
        # names for all outputs
        self.cntname = self.prefix+'_cnt_end_xyz.dump'
        self.totaloutname = self.prefix+'_PointCloud.dump'
        self.meshname = self.prefix+'_mesh'
        self.vectorname = self.prefix+'_Cry.directions'

        if polymers < 0 or polymers == 0:
            raise ValueError(
                "Polymer chain amount cannot be negative or equal to zero")
        elif not isinstance(polymers, int):
            raise ValueError("Polymer chain amount is not an integer")
        self.polymers = polymers

        if monomers < 0 or monomers == 0:
            raise ValueError(
                "Monomer chain amount cannot be negative or equal to zero")
        elif not isinstance(monomers, int):
            raise ValueError("Monomer chain amount is not an integer")
        self.monomers = monomers

        ####################################
        # Verho
        ####################################
        if verho_coeff < 0 or verho_coeff == 0:
            raise ValueError(
                "Verho coefficient cannot be negative or equal to zero")
        elif not isinstance(verho_coeff, int):
            raise ValueError("Verho coefficient is not an integer")
        self.verho_coeff = verho_coeff

        if not 0.0 < verho_cutoff < 1.0:
            raise ValueError("Verho Cutoff must be between 0 and 1")
        self.verho_cutoff = verho_cutoff

        if min_length < 1:
            raise ValueError("Minimum samples must be greater than zero")
        elif not isinstance(min_length, int):
            raise ValueError("Minimum length filter is not an integer")
        self.min_length = min_length

        ####################################
        # Nearest neighbors and Clustering
        ####################################

        if method == 'BallTree' or method == 'Brute':
            self.method = method
        else:
            raise TypeError('Unsupported method type')

        if eps < 0.0 or eps == 0.0:
            raise ValueError("Epsilon cannot be negative or equal to zero")
        self.eps = eps

        if not 0.0 < lam < 1.0:
            raise ValueError("Lambda Cutoff must be between 0 and 1")
        self.lam = lam

        if leaf_size < 0 or leaf_size == 0:
            raise ValueError("Leaf size cannot be negative or equal to zero")
        elif not isinstance(leaf_size, int):
            raise ValueError("Leaf size is not an integer")
        self.leaf_size = leaf_size

        if min_pts < 1:
            raise ValueError("Minimum samples must be greater than zero")
        self.min_pts = min_pts

        ####################################
        # Post Processing
        ####################################

        if second_filt > 1:
            raise ValueError("Second filter application is not a boolean")
        elif not isinstance(second_filt, int):
            raise ValueError("Second filter application is not a boolean ")
        else:
            self.second_filt = second_filt

        if crystalmin < 0 or crystalmin == 0:
            raise ValueError(
                "Crystal filter cutoff cannot be negative or equal to zero")
        elif not isinstance(crystalmin, int):
            raise ValueError("Crystal filter cutoff is not an integer")
        self.crystalmin = crystalmin

        if meshing > 1:
            raise ValueError("Meshing boolean is not a boolean")
        elif not isinstance(meshing, int):
            raise ValueError("Meshing boolean is not a boolean ")
        else:
            self.meshing = meshing

        if mesh_type == 'gauss' or mesh_type == 'alpha':
            self.mesh_type = mesh_type
        else:
            raise TypeError('Unsupported meshing type')

        if cntpresent > 1:
            raise ValueError("CNT presence boolean is not a boolean")
        elif not isinstance(cntpresent, int):
            raise ValueError("CNT presence boolean is not a boolean ")
        else:
            self.cntpresent = cntpresent

    def avg_vectors(self, vectorlist, vectorarray):
        '''
        Finding each passing section of chain and average
        '''
        # for check if desired
        chain_id = []
        n = 0

        chain_vector = []

        cc_x = []
        cc_y = []
        cc_z = []

        from numpy import sqrt, array
        from scipy.signal import hann, tukey, parzen, nuttall

        for i in range(0, len(vectorlist)):
            if vectorlist[i]:
                cc_x.append(vectorarray[i, 0])
                cc_y.append(vectorarray[i, 1])
                cc_z.append(vectorarray[i, 2])

            else:
                if len(cc_x) > 0:
                    sum_x = 0
                    sum_y = 0
                    sum_z = 0

                    wn = nuttall(len(cc_x))

                    for j in range(len(cc_x)):
                        sum_x = sum_x + cc_x[j]*wn[j]
                        sum_y = sum_y + cc_y[j]*wn[j]
                        sum_z = sum_z + cc_z[j]*wn[j]

                    chain_x = sum_x/sqrt(sum_x**2+sum_y**2+sum_z**2)
                    chain_y = sum_y/sqrt(sum_x**2+sum_y**2+sum_z**2)
                    chain_z = sum_z/sqrt(sum_x**2+sum_y**2+sum_z**2)

                    for j in range(len(cc_x)+1):
                        chain_vector.append([chain_x, chain_y, chain_z])
                        chain_id.append(n)

                    cc_x = []
                    cc_y = []
                    cc_z = []
                    n = n+1
                else:
                    chain_vector.append([0, 0, 0])
                    chain_id.append(-1)

        chain_vector = array(chain_vector)
        chain_id = array(chain_id)

        locations = []
        chords = []
        for i in range(len(vectorlist)):
            if vectorlist[i]:
                locations.append([self.x[i], self.y[i], self.z[i]])
                chords.append([chain_vector[i, 0], chain_vector[i, 1], chain_vector[i, 2]])

        locations = array(locations)
        chords = array(chords)

        return locations, chords

    def centroid_id(self, avgvector):
        from numpy import abs, array, min, max
        # crystal info
        c_loc = []
        for i in range(len(self.moltype)):
            if int(abs(self.moltype[i])) > 1:  # != 1 or 0 or -1:
                c_loc.append(
                    [self.moltype[i], self.x[i], self.y[i], self.z[i]])
        c_loc = array(c_loc)

        # find centroids
        centroid = []
        centroid_dist = []
        exist_list = []
        # skip -1, the noise
        for i in range(min(self.labels), max(self.labels)+2):
            cx = 0
            cy = 0
            cz = 0
            Exists = False
            n = 1

            for j in range(len(c_loc)):
                if i == c_loc[j, 0]:
                    cx = cx + c_loc[j, 1]
                    cy = cy + c_loc[j, 2]
                    cz = cz + c_loc[j, 3]
                    n = n + 1
                    Exists = True

            cx = cx/n
            cy = cy/n
            cz = cz/n

            centroid.append([i, cx, cy, cz])
            exist_list.append(Exists)

        centroid = array(centroid)
        exist_list = array(exist_list)

        # Centroid and Vector of crystals for Abaqus
        cent_vect = []
        for i in range(len(exist_list)):
            if exist_list[i]:
                cent_vect.append([centroid[i, 0],
                                  centroid[i, 1], centroid[i,2], centroid[i, 3],
                                  avgvector[i, 1], avgvector[i, 2], avgvector[i, 3]])

        cent_vect = array(cent_vect, dtype='object')

        seq = open(self.vectorname, 'w+')
        f = open(self.filename, 'r')
        n = 0
        for line in f.readlines()[0:8]:
            if n == 3:
                seq.write(str(len(c_loc))+'\n')
            else:
                seq.write(line)
            n = n+1
        seq.write('# Crystal Centroids, Sizes, and Avg Vectors \n')
        seq.write('Crystal, cx, cy, cz, vx, vy, vz \n')
        for i in range(len(cent_vect)):
            string = str(cent_vect[i, 0])+' '+ str(cent_vect[i, 1])+' '+str(cent_vect[i, 2])+' '+str(cent_vect[i, 3])+' '+ str(cent_vect[i, 4])+' '+str(cent_vect[i, 5])+' '+str(cent_vect[i, 6])+'\n'
            seq.write(string)
        seq.close()

        return exist_list

    def crystal_dump(self):
        #########################################
        # Full Crystalline Dump
        #########################################
        # crystal info
        from numpy import array
        c_loc = []
        for i in range(len(self.moltype)):
            if self.moltype[i] > 1:
                c_loc.append(
                    [self.atom[i], self.id[i], self.moltype[i], self.x[i], self.y[i], self.z[i]])
        c_loc = array(c_loc)

        seq = open(self.totaloutname, 'w+')
        f = open(self.filename, 'r')
        n = 0
        for line in f.readlines()[0:9]:
            if n == 3:
                seq.write(str(len(c_loc))+'\n')
            else:
                seq.write(line)
            n = n+1

        for i in range(len(c_loc)):
            string = str(c_loc[i, 0])+' '+str(c_loc[i, 1])+' '+str(c_loc[i, 2])\
            +' '+str(c_loc[i, 3])+' '+str(c_loc[i, 4]) +' '+str(c_loc[i, 5])\
            +' '+str(c_loc[i, 3])+' '+str(c_loc[i, 4]) +' '+str(c_loc[i, 5])+'\n'
            seq.write(string)
        seq.close()

    def length_filter(self, vectorlist, moltype=None):
        #########################################
        # Filtering chains to make sure they have a minimum length
        #########################################
        from numpy import array
        # Enforce Min length of N RUs
        newvl = [False]*len(vectorlist)
        n = 0
        for i in range(len(vectorlist)):
            if vectorlist[i]:
                # check if true
                n = n+1
            else:
                if n > self.min_length:
                    # Set true if passes
                    for j in range(0, n):
                        newvl[i-j] = True
                # if False
                n = 0

        del vectorlist
        vectorlist = array(newvl)

        if self.second_filt:
            checktype = [0]*len(vectorlist)
            for i in range(len(vectorlist)):
                if vectorlist[i] and moltype[i] > 0:
                    checktype[i] = moltype[i]
                else:
                    vectorlist[i] = False
                    checktype[i] = -1

            moltype = checktype

            return vectorlist, moltype

        else:
            return vectorlist

    def type_filter(self, vectorlist, labels):
        #########################################
        # Filter out based on type
        #########################################
        moltype = [0]*len(vectorlist)
        n = 0
        for i in range(len(vectorlist)):
            if vectorlist[i]:
                moltype[i] = int(labels[n])
                n = n+1

        checktype = [0]*len(vectorlist)
        for i in range(len(vectorlist)):
            if vectorlist[i] and moltype[i] > 0:
                checktype[i] = moltype[i]
            else:
                vectorlist[i] = False
                checktype[i] = -1

        moltype = checktype

        print('Found '+str(int(max(labels)))+' clusters')

        return moltype

    def ovito_meshing(self, exist_list, mesh_type, cntpresent):
        #########################################
        # Ovito Meshing -- Crystal / Amorphous
        #########################################
        import os
        import sys
        from numpy import abs, array

        from ovito.io import import_file, export_file
        from ovito.modifiers import ConstructSurfaceModifier, AffineTransformationModifier, CombineDatasetsModifier
        from ovito.vis import SurfaceMeshVis

        import meshio

        FREECADPATH = '/usr/lib/freecad/lib'

        sys.path.append(FREECADPATH)
        import FreeCAD
        import Mesh
        import Part

        #########################################
        # Pointclouds for meshing
        #########################################
        for i in range(len(exist_list)):
            if exist_list[i]:
                actual = i-1
                c_loc = []
                for j in range(len(self.moltype)):
                    if int(abs(self.moltype[j])) == actual:
                        c_loc.append(
                            [self.atom[j], self.id[j], self.moltype[j], self.x[j], self.y[j], self.z[j]])
                c_loc = array(c_loc)

                outname = self.prefix+'CryCloud'+str(actual)+'.dump'
                seq = open(outname, 'w+')
                f = open(self.filename, 'r')
                n = 0
                for line in f.readlines()[0:9]:
                    if n == 3:
                        seq.write(str(len(c_loc))+'\n')
                    else:
                        seq.write(line)
                    n = n+1

                for j in range(len(c_loc)):
                    string = str(c_loc[j, 0])+' '+str(c_loc[j, 1])+' '+str(c_loc[j, 2])+' '+str(c_loc[j, 3])+' '+str(
                        c_loc[j, 4])+' '+str(c_loc[j, 5])+' '+str(c_loc[j, 3])+' '+str(c_loc[j, 4])+' '+str(c_loc[j, 5])+' '+'\n'
                    seq.write(string)
                seq.close()

        #########################################
        # Ovito Meshing -- Crystal / Amorphous
        #########################################
        f = open(self.filename, 'r')
        n = 0
        box = f.readlines()[5:8]
        f.close()

        xlo, xhi = box[0].split()
        ylo, yhi = box[1].split()
        zlo, zhi = box[2].split()

        xlo = float(xlo)
        xhi = float(xhi)
        ylo = float(ylo)
        yhi = float(yhi)
        zlo = float(zlo)
        zhi = float(zhi)

        # For Affine Transform
        xlen = xhi-xlo + 4
        ylen = yhi-ylo + 4
        zlen = zhi-zlo + 4

        xorig = xlo-2
        yorig = ylo-2
        zorig = zlo-2

        for i in range(len(exist_list)):
            if exist_list[i]:
                actual = i-1
                outname = self.prefix+'CryCloud'+str(actual)+'.dump'

                pipeline = import_file(outname)

                pipeline.modifiers.append(AffineTransformationModifier(
                    # Transform box only to remove periodic surface mesh
                    operate_on={'cell'},
                    relative_mode=False,
                    target_cell=[[xlen, 0, 0, xorig],
                                 [0, ylen, 0, yorig],
                                 [0, 0, zlen, zorig]]))

                if mesh_type == 'gauss':
                    pipeline.modifiers.append(ConstructSurfaceModifier(
                        method=ConstructSurfaceModifier.Method.GaussianDensity,
                        grid_resolution=25,
                        isolevel=1.0,
                        radius_scaling=0.5))
                else:
                    pipeline.modifiers.append(ConstructSurfaceModifier(
                        method=ConstructSurfaceModifier.Method.AlphaShape,
                        radius=3,
                        smoothing_level=2))  # int

                surface_mesh = pipeline.compute().surfaces['surface']
                export_file(surface_mesh, self.meshname+"_cry" +
                            str(actual)+".vtk", "vtk/trimesh")

                mesh = meshio.read(self.meshname+"_cry"+str(actual)+".vtk")
                mesh.write(self.meshname+"_cry"+str(actual)+".obj")

                os.remove(self.meshname+"_cry"+str(actual)+".vtk")
                del mesh

                # https://forum.freecadweb.org/viewtopic.php?t=16295
                doc = FreeCAD.newDocument("ImportExport")
                mesh = Mesh.Mesh(self.meshname+"_cry"+str(actual)+".obj")
                shape = Part.Shape()
                shape.makeShapeFromMesh(mesh.Topology, 0.1)
                shape.exportStep(self.meshname+"_cry"+str(actual)+".step")

                del mesh, shape, doc, pipeline

                os.remove(self.meshname+"_cry"+str(actual)+".obj")
                os.remove(outname)

            if cntpresent:
                del mesh, pipeline
                #########################################
                # Ovito Meshing -- CNT
                #########################################
                pipeline = import_file(self.cntname)

                pipeline.modifiers.append(AffineTransformationModifier(
                    # Transform box only to remove periodic surface mesh
                    operate_on={'cell'},
                    relative_mode=False,
                    target_cell=[[xlen, 0, 0, xorig],
                                 [0, ylen, 0, yorig],
                                 [0, 0, zlen, zorig]]))

                if mesh_type == 'gauss':
                    pipeline.modifiers.append(ConstructSurfaceModifier(
                        method=ConstructSurfaceModifier.Method.GaussianDensity,
                        grid_resolution=25,
                        isolevel=0.5,
                        radius_scaling=1.2))
                else:
                    pipeline.modifiers.append(ConstructSurfaceModifier(
                        method=ConstructSurfaceModifier.Method.AlphaShape,
                        radius=3,
                        smoothing_level=2))  # int

                surface_mesh = pipeline.compute().surfaces['surface']
                export_file(surface_mesh, meshname+"_cnt.vtk", "vtk/trimesh")

                mesh = meshio.read(meshname+"_cry"+str(actual)+".vtk")
                mesh.write(meshname+"_cry"+str(actual)+".ply")

                os.remove(meshname+"_cry"+str(actual)+".vtk")
                del mesh

                # https://forum.freecadweb.org/viewtopic.php?t=16295
                doc = FreeCAD.newDocument("ImportExport")
                mesh = Mesh.Mesh(meshname+"_cry"+str(actual)+".ply")
                shape = Part.Shape()
                shape.makeShapeFromMesh(mesh.Topology, 0.1)
                shape.exportStep(meshname+"_cry"+str(actual)+".step")

                del mesh, shape, doc, pipeline

                os.remove(meshname+"_cry"+str(actual)+".ply")
                os.remove(outname)

    def quantity_filter(self, vectorlist, vectorarray, moltype, labels):
        #########################################
        # Filter out based on quantity of type
        #########################################
        from numpy import array, sqrt, min, max

        # https://www.geeksforgeeks.org/python-count-occurrences-element-list/
        def countX(lst, x):
            count = 0
            for ele in lst:
                if (ele == x):
                    count = count+1

            return count

        def aveVector(lst, vect, type):
            x = []
            y = []
            z = []
            for i in range(len(lst)):
                if (lst[i] == type):
                    x = x + vect[i, 0]
                    y = y + vect[i, 1]
                    z = z + vect[i, 2]
            if len(x) == 0:
                out_x = 0
                out_y = 0
                out_z = 0
            else:
                out_x = sum(x)/sqrt(x**2+y**2+z**2)
                out_y = sum(y)/sqrt(x**2+y**2+z**2)
                out_z = sum(z)/sqrt(x**2+y**2+z**2)
            return moltype, out_x, out_y, out_z

        pertype = []
        for i in range(min(labels), max(labels)+2):
            pertype.append([i, countX(moltype, i)])

        pertype = array(pertype)

        for i in range(len(pertype)):
            if pertype[i, 1] < self.crystalmin:
                for j in range(len(vectorlist)):
                    if moltype[j] == pertype[i, 0]:
                        moltype[j] = 1

        for i in moltype:
            if int(i) != 1 or -1 or 0:
                vectorlist[i] = True
            else:
                vectorlist[i] = False

        #########################################
        # Vector Array Averaging
        #########################################

        avgvector = []
        for i in range(min(labels), max(labels)+2):
            avgvector.append(aveVector(moltype, vectorarray, i))

        avgvector = array(avgvector, dtype='object')

        return vectorlist, moltype, avgvector

    def verho_equation(self, polymers, monomers):
        '''
        Verho search function to determine alignment of segments of poylmer chain
        '''
        from pandas import read_csv
        from numpy import array, sqrt

        # nan and nan1 to take place of ITEM: ATOM
        try:
            data = read_csv(self.filename, sep=' ', header=8, names=[
                            'atom', 'id', 'type', 'x', 'y', 'z', 'xu', 'yu', 'zu', 'nan', 'nan1'])
            data = data.sort_values(['id', 'atom'], ascending=[True, True])
        except:
            raise ValueError("Data is an empty array")

        # display data
        print(data)

        self.atom = array(data.atom)
        self.id = array(data.id)
        self.moltype = array(data.type)
        self.x = array(data.x)
        self.y = array(data.y)
        self.z = array(data.z)
        self.xu = array(data.xu)
        self.yu = array(data.yu)
        self.zu = array(data.zu)

        del data

        vectorlist = []
        vectorarray = []
        anglelist = []
        len_coeff = self.verho_coeff

        # Tuukka search function
        for i in range(polymers):
            for j in range(monomers):
                # in j above beginning of buffer or below end
                if j > (len_coeff-1) and j < (monomers-len_coeff-1):
                    # calculate average vector
                    dx = 0
                    dy = 0
                    dz = 0

                    # to iterate from -k to k, needs to be -k to k+1
                    for k in range(-len_coeff, len_coeff+1):
                        # vector
                        vx = self.xu[(j+1)+i*monomers+k] - \
                            self.xu[(j-1)+i*monomers+k]
                        vy = self.yu[(j+1)+i*monomers+k] - \
                            self.yu[(j-1)+i*monomers+k]
                        vz = self.zu[(j+1)+i*monomers+k] - \
                            self.zu[(j-1)+i*monomers+k]
                        # unit vector
                        ux = vx/sqrt(vx**2+vy**2+vz**2)
                        uy = vy/sqrt(vx**2+vy**2+vz**2)
                        uz = vz/sqrt(vx**2+vy**2+vz**2)
                        # add to d
                        dx = dx+ux
                        dy = dy+uy
                        dz = dz+uz
                    # avg d
                    lam = 1.0/(2*len_coeff+1)*sqrt(dx**2+dy**2+dz**2)

                    # final unit vector
                    ux = dx/sqrt(dx**2+dy**2+dz**2)
                    uy = dy/sqrt(dx**2+dy**2+dz**2)
                    uz = dz/sqrt(dx**2+dy**2+dz**2)

                    if ux < 0:
                        ux = -1*ux
                    if uy < 0:
                        uy = -1*uy
                    if uz < 0:
                        uz = -1*uz

                    if lam > self.verho_cutoff:
                        vectorlist.append(True)
                    else:
                        vectorlist.append(False)
                    vectorarray.append([ux, uy, uz])
                else:
                    vectorlist.append(False)
                    vectorarray.append([0, 0, 0])

        vectorlist  = array(vectorlist)
        vectorarray = array(vectorarray)

        return vectorlist, vectorarray

    def find(self):
        print('Reading Data and Verho Equation Application')
        # Import data and Verho Equation
        vectorlist, vectorarray = self.verho_equation(
            self.polymers, self.monomers)

        print('Minimum Length Enforcement \n')
        # Enforce Minimum Length
        vectorlist = self.length_filter(vectorlist)

        print('Chain averaging \n')
        # Average vectors for chain segment
        data, chords = self.avg_vectors(
            vectorlist, vectorarray)

        print('Clustering')
        # VDBSCAN
        from vdbscan import VDBSCAN

        VDB = VDBSCAN(data, chords, 'BallTree', self.eps, self.lam, self.leaf_size, self.min_pts)
        self.labels, core_loc = VDB.fit()

        print('Applying type filter for final pointcloud \n')
        # type filter

        moltype = self.type_filter(vectorlist, self.labels)

        # optional second length filter
        if self.second_filt:
            print('Applying secondary length filter \n')
            moltype, vectorlist = self.length_filter(
                vectorlist, self.moltype)

        print('Applying minimum crystal size filter \n')
        # quantity filter
        self.vectorlist, self.moltype, avgvector = self.quantity_filter(
            vectorlist, vectorarray, moltype, self.labels)

        #########################################
        # Centroid data printing
        #########################################
        print('Writing crystal point cloud file \n')
        self.crystal_dump()

        print('Determining centroids, vectors, and printing for use in FEM \n')
        exist_list = self.centroid_id(avgvector)

        if self.meshing:
            print('Writing meshes \n')
            self.ovito_meshing(exist_list, self.mesh_type, self.cntpresent)
