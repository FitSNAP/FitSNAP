import numpy as np
#from fitsnap3lib.io.input import Config

try:
    import torch
    import math

    class Gaussian3Body():
        """
        Class to calculate Gaussian 3-body descriptors.

        Args:
            num_desecriptors (int): Number of 3-body descriptors.
            cutoff (float): Neighborlist cutoff.
        """
        def __init__(self, num_descriptors, cutoff):
            self.num_descriptors = num_descriptors
            self.cutoff = cutoff

            self.pi = torch.tensor(math.pi)
            self.eta = 4.
            self.mu = torch.linspace(-1,1,self.num_descriptors)

        def calculate(self, rij, diff_norm, unique_i, numpy_bool = False):
            """
            Calculate 3body descriptors for all pairs. In the following discussion, :code:`num_neighs` 
            is the total number of neighbors in the entire batch, also equivalent to the total 
            number of pairs.

            Args:
                rij (:obj:`torch.tensor`): Pairwise distances of all pairs in this batch, size 
                                           (num_neighs, 1) where :code:`num_neighs` is number of neighbors 
                                           for all atoms in this batch
                diff_norm (:obj:`torch.tensor`): Pairwise normalized dispalcements between all pairs 
                                                 in this batch, size (num_neighs, 3)
                unique_i (:obj:`torch.long`): Indices of atoms i in this batch, size (num_neighs)
                numpy_bool (bool, optional): Default False; use True if wanting to convert from 
                                             numpy to torch tensor

            Returns:
                :obj:`torch.tensor`: Tensor of size (num_neighs, num_3body_descriptors)
            """

            # TODO: Look into using generators to reduce list overhead
            #       https://stackoverflow.com/questions/51105841/faster-python-list-comprehension

            if (numpy_bool):
                rij = torch.from_numpy(rij)
                diff_norm = torch.from_numpy(diff_norm)
                unique_i = torch.from_numpy(unique_i)

            # cutoff function for all pairs, size (num_neigh)

            fcrik = self.cutoff_function(rij) #.flatten()

            ui = unique_i.unique()

            # Cram a bunch of calculations into a single list comprehension to reduce overhead.
            # torch.mm() calculates a matrix of dot products for all pairs, then we fill the diagonals
            # with zeros to ignore dot products of the same pair.

            """
            # use this if we have pre-created lists of diff_norm and fcrik
            list_of_fcrik = [fcrik[unique_i==i] for i in ui]
            list_of_rij = [diff_norm[unique_i==i] for i in ui]
            list_of_dij = [torch.sum(
                              torch.exp(-1.0*self.eta
                                  * (torch.mm(list_of_rij[i], 
                                      torch.transpose(list_of_rij[i],0,1)).fill_diagonal_(0)[:,:,None]
                                  -self.mu)**2) 
                              * list_of_fcrik[i][:,None], 
                           dim=1)
                           for i in ui] #range(len_ui)]
            """

            descriptors_3body = torch.cat([torch.sum(
                                        torch.exp(-1.0*self.eta
                                            * (torch.mm(diff_norm[unique_i==i], 
                                                torch.transpose(diff_norm[unique_i==i],0,1)).fill_diagonal_(0)[:,:,None]
                                            -self.mu)**2) 
                                        * fcrik[unique_i==i][:,None], 
                                      dim=1)
                                    for i in ui],
                                    dim=0)

            return descriptors_3body

        #def cutoff_function(self, x, unique_i, xneigh):
        def cutoff_function(self, rij):
            """
            Calculate cutoff function for all rij

            Args:
                rij (:obj:`torch.Tensor`): Pairwise distances of all pairs in this batch, size 
                                           (num_neighs, 1) where `num_neighs` is number of neighbors 
                                           for all atoms in this batch.
            """

            c = self.cutoff
            function = 0.5 + 0.5*torch.cos(self.pi*(rij-0)/(c-0))

            return function[:,0]
            
except ModuleNotFoundError:

    class Gaussian3Body():
        """
        Dummy class for factory to read if torch is not available for import.
        """
        def __init__(self):
            raise ModuleNotFoundError("No module named 'torch'")
