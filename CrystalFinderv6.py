from CrystalFinder import CrystalFinder

filename = 'MullerPlatheReference_polymer.dump'

CF = CrystalFinder(  filename,
                # Simulation info
                polymers= 100,
                monomers= 1000,
                # Verho
                verho_coeff=2,
                verho_cutoff=0.92,
                min_length=7,
                # Clustering
                method='BallTree',
                eps=1.2,
                lam=0.95,
                leaf_size=30,
                min_pts=8,
                # Post Processing
                second_filt=False,
                crystalmin=100,
                # Meshing
                meshing=True,
                mesh_type='gauss',
                cntpresent=False)

CF.find()
