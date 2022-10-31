import numpy as np
#from fitsnap3lib.io.input import Config

try:
    import torch
    import math

    class Gaussian3Body():
        """
        Class to calculate Bessel function descriptors
        """
        def __init__(self, num_descriptors, cutoff):
            self.num_descriptors = num_descriptors
            self.cutoff = cutoff
            #self.eta = eta
            #self.mu = mu

            # hard code values for now
            #self.num_descriptors = 11
            #self.coeffs = torch.zeros((1,1,self.num_descriptors))
            #self.eta = 2
            #self.mu = torch.linspace(-1,1,num_descriptors)

            self.pi = torch.tensor(math.pi)
            self.eta = 2.
            self.mu = torch.linspace(-1,1,self.num_descriptors)
            #self.mu = torch.linspace(-1,1,self.num_descriptors).unsqueeze(1).unsqueeze(2)
            #self.mu = self.mu.transpose(0,2) # 1 x 1 x self.num_3body

        def calculate(self, x, unique_i, unique_j, xneigh, numpy_bool = False):
            """
            Calculate 3body descriptors for all pairs.

            Attributes
            ----------

            x: torch.tensor
                Positions of atoms in this batch, size (num_atoms, 3)

            unique_i:
                Indices of atoms i in this batch, size (num_neighbors) where `num_neighbors` is 
                number of neighbors for all atoms in this batch

            unique_j:
                Indices of atoms j in this batch, size (num_neighbors)

            xneigh:
                LAMMPS neighlist transformed positions of all neighbors in the batch, size (num_neighobrs, 3)

            numpy_bool: bool
                Default False; use True if wanting to convert from numpy to torch tensor
            """

            # TODO: Look into using generators to reduce list overhead
            #       https://stackoverflow.com/questions/51105841/faster-python-list-comprehension
            #       scroll down to see awesome generator solutions

            if (numpy_bool):
                x = torch.from_numpy(x)
                unique_i = torch.from_numpy(unique_i)
                unique_j = torch.from_numpy(unique_j)
                xneigh = torch.from_numpy(xneigh)

            assert(unique_i.size()[0]==xneigh.size()[0])
            #print(unique_i.size())
            #print(unique_j.size())
            #print(xneigh.size())


            # normalized displacements between atoms

            diff_norm = torch.nn.functional.normalize(x[unique_i] - xneigh, dim=1)

            # cutoff function for all pairs, size (num_neigh)

            fcrik = self.cutoff_function(x, unique_i, xneigh) #.flatten()
            #print(fcrik.size())
            #assert(False)
            #fcrik = self.cutoff_function(x, unique_i, xneigh).transpose(0,1)
            #print(fcrik.size())

            # unique indices of unique_i for this batch help resolve which atoms i are involved

            ui = unique_i.unique()

            # list of displacements for all atoms i
            # list_of_rij[i] size is (numneigh[i], 3)
            # we do this in a list because each atom i may have different numneigh, and we want 
            # outer product of these pairs for a single atom i, not between all possible pairs in sytem

            list_of_rij = [diff_norm[unique_i==i] for i in ui]
            len_ui = len(list_of_rij) # number of unique atoms i in this batch
            #print(len_ui)

            # list of cutoff functions for all pairs ik, will be repeated along rows to multiply 
            # with outer product of displacements
            # list_of_fcrik[i] is (numneigh[i] x numneigh[i]), where rows are repeated, and cutoff 
            # function is distinct among columns corresponding to different (i,k) pairs
            # TODO: torch.repeat() is known for poor performance; try repeat_interleave, expand, or 
            #       the broadcast indexing trick for multiplying vectors to tensors row-wise.
            # NOTE: if m = torch.tensor([[1,2,3],[4,5,6],[7,8,9]]) and d=torch.tensor([1,2,3]) we 
            #       can simply do m * d to multiply column vector d to rows of m.
            #       Ditto if d=torch.tensor([[1,2,3]])
            #list_of_fcrik = [fcrik[:,unique_i==i] for i in ui]

            #for i in ui:
            #    print(i)
            #for i in range(len_ui):
            #    print(i)
            #print(ui)
            #assert(False)

            list_of_fcrik = [fcrik[unique_i==i] for i in ui]
            #print(list_of_fcrik[0].size())
            #print(list_of_fcrik[0])
            #print(self.mu)

            #assert(False)
            #list_of_fcrik = [list_of_fcrik[idx].unsqueeze(2).repeat(list_of_rij[idx].size()[0],1,self.num_descriptors) 
            #                 for idx in range(len_ui)]
            #assert(False)

            # outer product of displacements gives rij \dot rik for all pairs, for a given atom i
            # this tensor is size  (numneigh[i] x numneigh[i]), is diagonal with ones because rij \dot rij,
            # and symmetric
            # list_of_mm[i] is the outer product for atom i
            # TODO: torch.repeat() is known for poor performance; try repeat_interleave, expand, or 
            #       the broadcast indexing trick for multiplying vectors to tensors row-wise.

            #list_of_mm = [torch.mm(list_of_rij[idx], 
            #              torch.transpose(list_of_rij[idx],0,1)).unsqueeze(2).repeat(1,1,self.num_descriptors) 
            #              for idx in range(len_ui)]

            """
            list_of_mm = [torch.mm(list_of_rij[idx], 
                          torch.transpose(list_of_rij[idx],0,1))
                          for idx in range(len_ui)] 
            """
            #list_of_mm = [torch.mm(list_of_rij[idx], 
            #              torch.transpose(list_of_rij[idx],0,1))
            #              for idx in range(len_ui)]                      

            #assert(False)

            # exponentiate the outer products while applying eta, mu, and cutoff function
            # list_of_exp[i] is the exponentiated gaussian with applied cutoff function for atom i
            # and has size (numneigh[i] x numneigh[i]) since we're still dealing with all pairs

            #list_of_exp = [torch.exp(-1.0*self.eta*(list_of_mm[i]-self.mu)**2)*list_of_fcrik[i] 
            #               for i in range(len_ui)]

            #list_of_exp = [torch.exp(-1.0*self.eta*(list_of_mm[i][:,:,None]-self.mu)**2) 
            #               * list_of_fcrik[i][:,None]
            #               for i in range(len_ui)]

            #print(list_of_exp[0].size())
            #assert(False)

            # Subtract diagonal elements (j,k) where j=k, for all descriptors.
            # TODO: Better way to do this? These for loops are slow... Adds ~4 seconds per epoch
            #       in Ta example.

            """"
            for i in range(len_ui):
              for d in range(self.num_descriptors):
                  list_of_exp[i][:,:,d].fill_diagonal_(0)
            """

            #print(list_of_exp[0][:,:,0])
            #assert(False)

            # calculate 3body descriptors for each pair by summing over all atoms k, along the 
            # columns
            # descriptors_3body has size (num_pairs x num_descriptors) since we concatenate all pairs 
            # for each atom i

            #list_of_dij = [torch.sum(list_of_exp[i],dim=1) for i in range(len_ui)]

            # trying to reduce number of loops:

            """
            list_of_dij = [torch.sum(
                           torch.exp(-1.0*self.eta*(list_of_mm[i][:,:,None]-self.mu)**2) 
                           * list_of_fcrik[i][:,None], 
                           dim=1)
                           for i in range(len_ui)]
            """

            # Cram a bunch of calculations into a single list comprehension to reduce overhead.
            # torch.mm() calculates a matrix of dot products for all pairs, then we fill the diagonals
            # with zeros to ignore dot products of the same pair.

            list_of_dij = [torch.sum(
                              torch.exp(-1.0*self.eta
                                  * (torch.mm(list_of_rij[i], torch.transpose(list_of_rij[i],0,1)).fill_diagonal_(0)[:,:,None]
                                  -self.mu)**2) 
                              * list_of_fcrik[i][:,None], 
                           dim=1)
                           for i in ui] #range(len_ui)]

            #print(list_of_dij[0].size())
            #assert(False)

            descriptors_3body = torch.cat(list_of_dij, dim=0)
            #print(descriptors_3body.size())

            return descriptors_3body


        def calculate_rij(self, x, unique_i, xneigh):
            """
            Calculate radial distance between all pairs

            Attributes
            ----------

            x: torch.Tensor.float
                Array of positions for this batch

            unique_i: atoms i for all atoms in this batch indexed starting from 0 to (natoms_batch-1)

            xneigh: torch.Tensor.float
                Array of neighboring positions (ghost atoms) for this batch, 
                lined up with unique_i for each atom i

            Returns
            -------

            rij: torch.Tensor.float
                Pairwise distance tensor with size (number_neigh, 1)
            """

            diff = x[unique_i] - xneigh
            rij = torch.linalg.norm(diff, dim=1)

            rij = rij.unsqueeze(1)        

            return rij

        def calculate_g3b(self, rij, n):
            """
            Calculate Gaussian 3 body descriptor for all pairs 

            Attributes
            ----------

            rij: torch.Tensor.float
                Pairwise distance tensor with size (number_neigh, 1)

            n: torch.Tensor.float
                Integer in float form representing descriptor index

            Returns
            -------

            g3bn: torch.Tensor.float
                Gaussian 3body descriptor with index n with size ??
                Radial Bessel function for base n with size (number_neigh, 1)
            """

            c = self.cutoff
            pi = torch.tensor(math.pi)
            two_over_c = torch.tensor(2./c)
            rbf = torch.div(torch.sqrt(two_over_c)*torch.sin(((n*pi)/c)*rij), rij)     

            return rbf

        def radial_bessel_basis(self, x, neighlist, unique_i, xneigh):
            """
            Calculate radial Bessel basis functions.

            Attributes
            ----------

            x: torch.Tensor.float
                Array of positions for this batch
            
            neighlist: torch.Tensor.long
                Sparse neighlist for this batch

            unique_i: atoms i for all atoms in this batch indexed starting from 0 to (natoms_batch-1)

            xneigh: torch.Tensor.float
                positions of neighbors corresponding to indices j in neighlist
            """

            num_rbf = self.num_descriptors # number of radial basis functions
                                           # e.g. 3 includes n = 1,2,3

            rij = self.calculate_rij(x, neighlist, unique_i, xneigh)

            basis = torch.cat([self.calculate_bessel(rij, n) for n in range(1,num_rbf+1)], dim=1)

            return basis

        def cutoff_function(self, x, unique_i, xneigh):
            """
            Calculate cutoff function for all rij

            Attributes
            ----------

            x: torch.Tensor.float
                Array of positions for this batch

            unique_i: atoms i for all atoms in this batch indexed starting from 0 to (natoms_batch-1)

            xneigh: torch.Tensor.float
                positions of neighbors corresponding to indices j in neighlist
            """

            rij = self.calculate_rij(x, unique_i, xneigh)

            c = self.cutoff
            #pi = torch.tensor(math.pi)

            #function = 0.5 - 0.5*torch.sin(pi_over_two*((rij-R)/D))
            function = 0.5 + 0.5*torch.cos(self.pi*(rij-0)/(c-0))

            return function[:,0]

        def numpy_calculate_rij(self, x, neighlist, xneigh):
            """
            Calculate radial distance between all pairs

            Attributes
            ----------

            x: np.array
                Array of positions for this batch
            
            neighlist: int
                Sparse neighlist for this batch, columns are i, j

            xneigh: np.array
                Array of neighboring positions (ghost atoms) for this batch, 
                lined up with neighlist[:,1]

            Returns
            -------

            rij: np.array
                Pairwise distance tensor with size (number_neigh, 1)
            """

            diff = x[neighlist[:,0]] - xneigh
            rij = np.linalg.norm(diff, axis=1)    

            rij = np.expand_dims(rij, axis=1)

            return rij

        def numpy_calculate_bessel(self, rij, n):
            """
            Calculate a specific radial bessel function "n" for all pairs

            Attributes
            ----------

            rij: np.array
                Pairwise distance tensor with size (number_neigh, 1)

            n: float
                Integer in float form representing Bessel radial parameter n

            Returns
            -------

            rbf: np.array
                Radial Bessel function for base n with size (number_neigh, 1)
            """

            c = self.cutoff
            pi = math.pi
            two_over_c = 2./c
            rbf = np.divide(np.sqrt(two_over_c)*np.sin(((n*pi)/c)*rij), rij)    

            return rbf

        def numpy_radial_bessel_basis(self, x, neighlist, xneigh):
            """
            Calculate radial Bessel basis functions.

            Attributes
            ----------

            x: np.array
                Array of positions for this batch
            
            neighlist: int
                Sparse neighlist for this batch

            xneigh: np.array
                positions of neighbors corresponding to indices j in neighlist
            """

            num_rbf = self.num_descriptors # number of radial basis functions
                                           # e.g. 3 includes n = 1,2,3

            rij = self.numpy_calculate_rij(x,neighlist,xneigh)

            basis = np.concatenate([self.numpy_calculate_bessel(rij, n) for n in range(1,num_rbf+1)], axis=1)
            #basis = np.vstack([self.numpy_calculate_bessel(rij, n) for n in range(1,num_rbf+1)])

            return basis
            
except ModuleNotFoundError:

    class Bessel():
        """
        Dummy class for factory to read if torch is not available for import.
        """
        def __init__(self):
            raise ModuleNotFoundError("No module named 'torch'")
