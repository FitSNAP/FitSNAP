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
            self.num_descriptors = num_descriptors
            self.cutoff = cutoff

        def calculate_bessel(self, rij, cutoff_functions, n):
            """
            Calculate a specific radial bessel function `n` for all pairs.

            Args:
                rij (torch.Tensor.float): Pairwise distance tensor with size (num_neigh, 1).
                n (torch.Tensor.float): Integer in float form representing Bessel radial parameter n.

            Returns:
                rbf (torch.Tensor.float): Radial Bessel function for base n with size (num_neigh, 1).
            """

            c = self.cutoff
            pi = torch.tensor(math.pi)
            two_over_c = torch.tensor(2./c)
            rbf = torch.div(torch.sqrt(two_over_c)*torch.sin(((n*pi)/c)*rij), rij)*cutoff_functions     

            return rbf

        def radial_bessel_basis(self, rij, cutoff_functions, numpy_bool = False):
            """
            Calculate radial Bessel basis functions.

            Args:
                x (torch.Tensor.float): Array of positions for this batch.
                neighlist (torch.Tensor.long): Sparse neighlist for this batch.
                unique_i (torch.Tensor.long): Atoms i for all atoms in this batch indexed starting 
                    from 0 to (natoms_batch-1).
                xneigh (torch.Tensor.float): Positions of neighbors corresponding to indices j in 
                    the neighbor list.

            Returns:
                basis (torch.Tensor.float): Concatenated tensor of Bessel functions for all pairs 
                    with size (num_neigh, num_descriptors)
            """

            if (numpy_bool):
                rij = torch.from_numpy(rij)

            basis = torch.cat([self.calculate_bessel(rij, cutoff_functions, n) for n in range(1,self.num_descriptors+1)], dim=1)

            return basis

        def cutoff_function(self, rij, numpy_bool=False):
            """
            Calculate cutoff function for all rij.

            Args:
                x (torch.Tensor.float): Array of positions for this batch.
                neighlist (torch.Tensor.long): Sparse neighbor list for this batch.
                xneigh (torch.Tensor.float): Positions of neighbors corresponding to indices j in 
                    the neighbor list. 

            Returns:
                function (torch.Tensor.float): Cutoff function values for all rij. 
            """

            if (numpy_bool):
                rij = torch.from_numpy(rij)

            rmin = 3.5

            mask = rij > rmin
            if (rij.dtype==torch.float64):
                function = torch.empty(rij.size()).double() # need double if doing FD test
            else:
                function = torch.empty(rij.size())

            c = self.cutoff
            pi = torch.tensor(math.pi)

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
