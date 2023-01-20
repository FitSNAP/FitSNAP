
import torch
from torch import from_numpy
from torch.nn import Parameter
import math
from fitsnap3lib.lib.neural_networks.descriptors.bessel import Bessel
from fitsnap3lib.lib.neural_networks.descriptors.g3b import Gaussian3Body
"""
Try to import mliap package after: https://github.com/lammps/lammps/pull/3388
See related bug: https://github.com/lammps/lammps/issues/3204
For now we only use the following two MLIAP features for writing LAMMPS-ready pytorch models.
"""

def create_torch_network(layer_sizes):
    """
    Creates a pytorch network architecture from layer sizes.
    This also performs standarization in the first linear layer.
    This only supports softplus as the nonlinear activation function.

    Args:
        layer_sizes (:obj:`list` of :obj:`int`): Number of nodes for each layer.

    Returns:
        :obj:`torch.nn.Sequential`: Neural network architecture
    """

    layers = []
    try:
        layers.append(torch.nn.Linear(layer_sizes[0], layer_sizes[0]))
        for i, layer in enumerate(layer_sizes):
            layers.append(torch.nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
            layers.append(torch.nn.Softplus())
    except IndexError:
        layers.pop()
    return torch.nn.Sequential(*layers)


class FitTorch(torch.nn.Module):
    """
    FitSNAP PyTorch model for pairwise networks.

    Attributes:
        networks (:obj:`list` of :obj:`torch.nn.Sequential`): Description of `attr1`.
        descriptor_count (:obj:`int`): Number of descriptors for a pair.
        num_radial (:obj:`int`): Number of radial descriptors for a pair.
        num_3body (:obj:`int`): Number of 3-body descriptors for a pair.
        cutoff (:obj:`float`): Radial cutoff for neighlist.
        n_elements (:obj:`int`): Number of element types.
        multi_element_option (:obj:`int`): Setting for how to deal with multiple element types. 

    """

    def __init__(self, networks, descriptor_count, num_radial, num_3body, cutoff, n_elements=1, multi_element_option=1, dtype=torch.float32):
        """
        Initializer.
        """
        super().__init__()

        # pytorch does a nifty thing where each attribute here makes a unique key in state_dict
        # we therefore need to make a unique instance attribute for each network
        # by doing this, self.state_dict() gets automatically populated by pytorch internals

        self.dtype = dtype

        for indx, model in enumerate(networks):
            networks[indx].to(self.dtype)
            setattr(self, "network_architecture"+str(indx), networks[indx])


        self.networks = networks

        # now self.state_dict is populated with the attributes declared above 
        # print("Model's state_dict:")
        # for param_tensor in self.state_dict():
        #     print(param_tensor, "\t", self.state_dict()[param_tensor].size())   
         
        self.num_descriptors = descriptor_count
        self.num_radial = num_radial
        self.num_3body = num_3body
        self.n_elem = n_elements
        self.multi_element_option = multi_element_option

        # create descriptor objects with settings, used to calculate descriptors in forward pass

        self.bessel = Bessel(num_radial, cutoff) # Bessel object provides functions to calculate descriptors
        self.g3b = Gaussian3Body(num_3body, cutoff)

    def forward(self, x, neighlist, transform_x, indices, atoms_per_structure, types, unique_i, unique_j, device, dtype=torch.float32):
        """
        Forward pass through the PyTorch network model, calculating both energies and forces.

        Args:
            x (:obj:`torch.Tensor.float`): Array of positions for this batch
            neighlist (:obj`torch.Tensor.long`): Sparse neighlist for this batch
            transform_x (:obj:`torch.Tensor.float`): Array of LAMMPS transformed positions of neighbors
                                                     for this batch. 
            indices (:obj:`torch.Tensor.long`): Array of configuration indices upon which to 
                                                contract pairwise energies, for this batch.
            atoms_per_structure (:obj:`torch.Tensor.long`): Number of atoms per configuration for 
                                                            this batch.
            types (:obj:`torch.Tensor.long`): Atom types starting from 0, for this batch.

            unique_i (:obj:`torch.Tensor.long`): Atoms i for all atoms in this batch indexed 
                                                 starting from 0 to (natoms_batch-1)
            unique_j (:obj:`torch.Tensor.long`): Neighbors j for all atoms in this batch indexed 
                                                 starting from 0 to (natoms_batch-1)
            dtype (:obj:`torch.float32`, optional): Data type used for torch tensors, default is 
                                                   torch.float32 for training, but we set to 
                                                   torch.float64 for finite difference tests to 
                                                   ensure correct force calculations.

            device: pytorch accelerator device object

        Returns:
            tuple of (predicted_energy_total, predicted_forces). First element is predicted energies 
            for this batch, second element is predicted forces for this batch.

        """

        # create neighbor positions by transforming atom j positions

        xneigh = transform_x + x[unique_j,:]

        # Calculate displacements and distances - needed for various descriptor functions.
        # NOTE: Only do this once so we don't bloat the computational graph.
        # diff size is (numneigh, 3)
        # diff_norm is size (numneigh, 3)
        # rij is size (numneigh,1)

        diff = x[unique_i] - xneigh
        diff_norm = torch.nn.functional.normalize(diff, dim=1) # need for g3b
        rij = torch.linalg.norm(diff, dim=1).unsqueeze(1)  # need for cutoff and various other functions

        #print(rij.size())
        #print(diff[:8,:])
        #print(rij.size())
        #print(neighlist)
        # Calculate cutoff functions once for pairwise terms here, because we use the same cutoff 
        # function for both radial basis and eij.

        cutoff_functions = self.bessel.cutoff_function(rij)

        # calculate Bessel radial basis

        rbf = self.bessel.radial_bessel_basis(rij, cutoff_functions)
        assert(rbf.size()[0] == neighlist.size()[0])

        #print("Max RBF:")
        #print(torch.max(rbf))

        # calculate 3 body descriptors 

        descriptors_3body = self.g3b.calculate(rij, diff_norm, unique_i)

        #print(f"Max d3body: {torch.max(descriptors_3body)}")

        # concatenate radial descriptors and 3body descriptors

        descriptors = torch.cat([rbf, descriptors_3body], dim=1) # num_pairs x num_descriptors

        assert(descriptors.size()[0] == xneigh.size()[0])

        # input descriptors to a network for each pair; calculate pairwise energies

        eij = self.networks[0](descriptors)

        #print(f"Max eij: {torch.max(eij)}")

        # now self.state_dict is populated with the attributes declared above 
        # print("Model's state_dict:")
        #for param_tensor in self.state_dict():
        #    print(param_tensor, "\t", self.state_dict()[param_tensor]) 

        assert(cutoff_functions.size() == eij.size())
        eij = torch.mul(eij,cutoff_functions)

        # calculate energy per config
        
        predicted_energy_total = torch.zeros(atoms_per_structure.size(), dtype=dtype).to(device)
        predicted_energy_total.index_add_(0, indices, eij.squeeze())

        # calculate spatial gradients

        gradients_wrt_x = torch.autograd.grad(predicted_energy_total, 
                                    x, 
                                    grad_outputs=torch.ones_like(predicted_energy_total), 
                                    create_graph=True)[0]

        assert(gradients_wrt_x.size() == x.size())

        # force is negative gradient

        predicted_forces = -1.0*gradients_wrt_x

        # TODO: For now we divide energy by 2, to fix this we need to incorporate the LAMMPS 
        # neighlist-transformed positions in the forward pass.

        predicted_energy_total = 1.0*predicted_energy_total

        return (predicted_energy_total, predicted_forces)

    def import_wb(self, weights, bias):
        """
        Imports weights and bias into FitTorch model

            Parameters:
                weights (list of numpy array of floats): Network weights at each layer
                bias (list of numpy array of floats): Network bias at each layer

        """

        assert len(weights) == len(bias)
        imported_parameter_count = sum(w.size + b.size for w, b in zip(weights, bias))
        combined = [None] * (len(weights) + len(bias))
        combined[::2] = weights
        combined[1::2] = bias

        assert len([p for p in self.network_architecture.parameters()]) == len(combined)
        assert sum(p.nelement() for p in self.network_architecture.parameters()) == imported_parameter_count

        state_dict = self.state_dict()
        for i, key in enumerate(state_dict.keys()):
            state_dict[key] = torch.tensor(combined[i])
        self.load_state_dict(state_dict)

    def write_lammps_torch(self, filename="FitTorch.pt"):
        """
        Saves lammps ready pytorch model.

            Parameters:
                filename (str): Filename for lammps usable pytorch model

        """
        
        #from lammps.mliap.pytorch import IgnoreElems, TorchWrapper, ElemwiseModels
        from fitsnap3lib.lib.neural_networks.write import PairNN, IgnoreElems

        # self.network_architecture0 is network model for the first element type

        """
        if self.n_elem == 1:
            print("Single element, saving model with IgnoreElems ML-IAP wrapper")
            model = IgnoreElems(self.network_architecture0)
        else:
            print("Multi element, saving model with ElemwiseModels ML-IAP wrapper")
            model = ElemwiseModels(self.networks, self.n_elem)
        """
        model = IgnoreElems(self.network_architecture0)
        linked_model = PairNN(model, n_descriptors=self.num_descriptors, n_elements=self.n_elem)
        torch.save(linked_model, filename)
        

    def load_lammps_torch(self, filename="FitTorch.pt"):
        """
        Loads lammps ready pytorch model.

            Parameters:
                filename (str): Filename of lammps usable pytorch model

        """
        model_state_dict = torch.load(filename).state_dict()
        list_of_old_keys = [*model_state_dict.keys()]
        new_dict = self.state_dict()
        assert len(model_state_dict) == len(new_dict)
        for i, key in enumerate(new_dict.keys()):
            new_dict[key] = model_state_dict[list_of_old_keys[i]]
        self.load_state_dict(new_dict)
