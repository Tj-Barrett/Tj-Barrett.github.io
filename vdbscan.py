'''
VDBSCAN is a normal DBSCAN algorithm plus a check on a precomputed orientation
vector associated with each point, in order to better cluster molecular dynamics
simulation data. The vector check method is based on the paper by Tuukka Verho
et al. :

"Crystal Growth in Polyethylene by Molecular Dynamics: The Crystal Edge and Lame
llar Thickness" Macromolecules 51, 13, 2018
https://doi.org/10.1021/acs.macromol.8b00857

DBSCAN algorithim is based on Ryan Davidson's medium article,
as well as Erik Lindernoren's github library.

This method precomputes the neighbor list making it compatible with the balltree
method of Jake VanderPlas. BallTree is significantly faster than the brute
method

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


class VDBSCAN(object):
    def __init__(self, data, chords, method='BallTree', eps=1.1, lam=0.9, leaf_size=30, min_pts=8):

        # validate data
        if data.size == 0:
            raise ValueError("Data is an empty array")
        self.data = data
        self.labels = [0]*len(data)
        self.cores = [True]*len(data)

        if chords is not None:
            if chords.size == 0:
                raise ValueError("Chord data is an empty array")
            else:
                self.chords = chords

        else:
            # if none make an array of all 1s
            self.chords = [1]*len(data)

        if method == 'BallTree' or method == 'Brute':
            self.method = method
        else:
            raise TypeError('Unsupported method type')

        if eps < 0 or eps == 0:
            raise ValueError("Epsilon cannot be negative or equal to zero")
        self.eps = eps

        if lam < 0 or lam == 0:
            raise ValueError("Lambda cannot be negative or equal to zero")
        self.lam = lam

        if leaf_size < 0 or leaf_size == 0:
            raise ValueError("Leaf size cannot be negative or equal to zero")
        self.leaf_size = leaf_size

        if min_pts < 1:
            raise ValueError("Minimum samples must be greater than zero")
        self.min_pts = min_pts

    def expand(self, ind, cluster):
        self.labels[ind] = cluster

        neighborhood = self.neighbors_dict[ind]

        i = 0
        while i < len(neighborhood):
            query = neighborhood[i]

            # if labeled noise by not being an initial core point, claim
            if self.labels[query] == -1:
                self.labels[query] = cluster

            # if in another cluster or unclaimed, claim it and neighbors
            else:
                # elif self.labels[query] == 0:
                self.labels[query] = cluster
                leaves = self.neighbors_dict[query]

                # if it is a core point as well, add to list to look through
                if self.count[query] >= self.min_pts:

                    for j in range(len(leaves)):
                        if leaves[j] not in neighborhood:
                            neighborhood.append(leaves[j])
            # else:

            i = i+1

    def get_neighbors(self):
        '''
        Modified version of distance formula used in DBSCAN. Looks for distances
        and the alignment of the average chord vector of the segment to count
        as a neighbor

        Both methods will return:
            count - total # neighbors passing
            neighbors - array of indexes of neighbors passing

        Does not assign core points
        '''
        from numpy import dot

        if self.method == 'BallTree':
            # nearest neighbor run
            try:
                from nn_ball_tree import BallTree

                balltree = BallTree(self.data, self.leaf_size)
                dist, ints = balltree.query(self.data, self.leaf_size)

                # chord vector sort
                neighbors_dict = {}
                count = []
                for i in range(len(self.data)):
                    # total neighborhood
                    neighborhood = ints[i]
                    # neighbors that pass
                    neighbors = []
                    n = 0
                    for j in range(len(dist[i])):
                        # distances of neighborhood
                        array = dist[i]

                        # neighbor in question
                        query = neighborhood[j]

                        # check if above length, not parallel, out of range, or same place
                        if (array[j] >= self.eps) \
                                or (dot(self.chords[i], self.chords[query]) <= self.lam)\
                                or (i == query):
                            continue

                        neighbors.append(query)
                        n = n+1
                    neighbors_dict[i] = neighbors
                    count.append(n)
            except IndexError:
                raise IndexError(
                    'Bounds error in BallTree. Try modifying leaf_size.')

        else:
            from cnn_brute import CBrute

            brute = CBrute(self.data, self.chords, self.eps, self.lam)
            count, neighbors_dict = brute.brute_total()

        return count, neighbors_dict

    def get_avg_vectors(self):
        '''
        Finding each passing section of chain and average
        '''
        # for check if desired
        # chain_id = []
        # n = 0

        avgchain = []

        cc_x = []
        cc_y = []
        cc_z = []

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

        			chain_x = sum_x/np.sqrt(sum_x**2+sum_y**2+sum_z**2)
        			chain_y = sum_y/np.sqrt(sum_x**2+sum_y**2+sum_z**2)
        			chain_z = sum_z/np.sqrt(sum_x**2+sum_y**2+sum_z**2)

        			for j in range(len(cc_x)+1):
        				avgchain.append([chain_x,chain_y,chain_z])
        				charge.append(n)

        			cc_x = []
        			cc_y = []
        			cc_z = []
        			# n = n+1
        		else:
        			avgchain.append([0,0,0])
        			charge.append(-1)

        avgchain = np.array(avgchain)
        ccharge = np.array(charge)

        print(len(avgchain))

    def fit(self):
        cluster = 2

        # get neighbors from distance and vector
        self.count, self.neighbors_dict = self.get_neighbors()

        # remove leaves
        for i in range(len(self.count)):
            if self.count[i] < self.min_pts:
                self.labels[i] = -1
                self.cores[i] = False
                # consider it a leaf
                self.neighbors_dict[i] = []

        # expand clusters from core nodes
        for i in range(len(self.count)):
            if self.labels[i] == 0:
                self.expand(i, cluster)
                cluster = cluster+1

                # print(str(float(i)/float(len(self.count))*100.0)[:4]+'%')

        return self.labels, self.cores
