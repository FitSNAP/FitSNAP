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

        def calculate_bessel(self, rij, cutoff_functions, n):
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

            #cutoff_functions = self.cutoff_function(rij)
            rbf = torch.div(torch.sqrt(two_over_c)*torch.sin(((n*pi)/c)*rij), rij)*cutoff_functions     

            return rbf

        #def radial_bessel_basis(self, x, neighlist, unique_i, xneigh):
        def radial_bessel_basis(self, rij, cutoff_functions, numpy_bool = False):
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

            #num_rbf = self.num_descriptors # number of radial basis functions
                                           # e.g. 3 includes n = 1,2,3

            #rij = self.calculate_rij(x, neighlist, unique_i, xneigh)

            if (numpy_bool):
                rij = torch.from_numpy(rij)

            basis = torch.cat([self.calculate_bessel(rij, cutoff_functions, n) for n in range(1,self.num_descriptors+1)], dim=1)

            return basis

        #def cutoff_function(self, x, neighlist, unique_i, xneigh):
        def cutoff_function(self, rij, numpy_bool=False):
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

            # TODO: Don't need to calculate rij here if we do it once in the beginning...
            #       This bloats the computational graph

            #rij = self.calculate_rij(x, neighlist, unique_i, xneigh)

            if (numpy_bool):
                rij = torch.from_numpy(rij)

            rmin = 3.5

            mask = rij > rmin
            #print(mask)
            if (rij.dtype==torch.float64):
                function = torch.empty(rij.size()).double()
            else:
                function = torch.empty(rij.size()) #.double() # need to use double if doing FD test

            c = self.cutoff #3.0 # cutoff
            pi = torch.tensor(math.pi)

            #function = 0.5 - 0.5*torch.sin(pi_over_two*((rij-R)/D))

            #print(type(rij))
            #print(rij.dtype)
            function[mask] = 0.5 + 0.5*torch.cos(pi*(rij[mask]-rmin)/(c-rmin))
            function[~mask] = 1.0

            return function
            
except ModuleNotFoundError:

    class Bessel():
        """
        Dummy class for factory to read if torch is not available for import.
        """
        def __init__(self):
            raise ModuleNotFoundError("No module named 'torch'")
