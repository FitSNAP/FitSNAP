import numpy as np
#from fitsnap3lib.io.input import Config

try:
    import torch
    import math

    class Bessel():
        """
        Class to calculate Bessel function descriptors
        """
        def __init__(self, num_descriptors, cutoff):
            #self.config = Config()
            #self.num_descriptors = self.config.sections['CUSTOM'].num_descriptors
            #self.cutoff = self.config.sections['CUSTOM'].cutoff
            self.num_descriptors = num_descriptors
            self.cutoff = cutoff

        def calculate_rij(self, x, neighlist, unique_i, xneigh):
            """
            Calculate radial distance between all pairs

            Attributes
            ----------

            x: torch.Tensor.float
                Array of positions for this batch
            
            neighlist: torch.Tensor.long
                Sparse neighlist for this batch

            unique_i: atoms i for all atoms in this batch indexed starting from 0 to (natoms_batch-1)

            xneigh: torch.Tensor.float
                Array of neighboring positions (ghost atoms) for this batch, 
                lined up with neighlist[:,1]

            Returns
            -------

            rij: torch.Tensor.float
                Pairwise distance tensor with size (number_neigh, 1)
            """

            diff = x[unique_i] - xneigh
            rij = torch.linalg.norm(diff, dim=1)

            rij = rij.unsqueeze(1)        

            return rij

        def calculate_bessel(self, rij, n):
            """
            Calculate a specific radial bessel function "n" for all pairs

            Attributes
            ----------

            rij: torch.Tensor.float
                Pairwise distance tensor with size (number_neigh, 1)

            n: torch.Tensor.float
                Integer in float form representing Bessel radial parameter n

            Returns
            -------

            rbf: torch.Tensor.float
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

        def cutoff_function(self, x, neighlist, unique_i, xneigh):
            """
            Calculate cutoff function for all rij

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

            c = self.cutoff #3.0 # cutoff
            pi = torch.tensor(math.pi)

            #function = 0.5 - 0.5*torch.sin(pi_over_two*((rij-R)/D))
            function = 0.5 + 0.5*torch.cos(pi*(rij-0)/(c-0))

            return function



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
