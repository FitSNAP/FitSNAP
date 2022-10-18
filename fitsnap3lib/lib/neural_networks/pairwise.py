
import torch
from torch import from_numpy
from torch.nn import Parameter
import math
from fitsnap3lib.lib.neural_networks.descriptors.bessel import Bessel
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

        Parameters:
            layer_sizes (list of ints): Size of each network layers

        Return:
            Network Architecture of type neural network sequential

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
    FitSNAP PyTorch network model. 

    Attributes
    ----------

    networks: list
        A list of nn.Sequential network architectures.
        Each type-type pairwise interaction is a different network.

    descriptor_count: int
        Length of descriptors for an atom

    energy_weight: float
        Weight of energy in loss function, used for determining if energy is fit or not

    force_weight: float
        Weight of force in loss function, used for determining if energy is fit or not

    n_elements: int
        Number of differentiable atoms types 

    multi_element_option: int
        Option for which multi-element network model to use
    """

    def __init__(self, networks, descriptor_count, energy_weight, force_weight, cutoff, n_elements=1, multi_element_option=1, dtype=torch.float32):
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
        self.n_elem = n_elements
        self.multi_element_option = multi_element_option

        self.energy_bool = True
        self.force_bool = True
        if (energy_weight==0.0):
            self.energy_bool = False
        if (force_weight==0.0):
            self.force_bool = False

        self.bessel = Bessel(descriptor_count, cutoff) # Bessel object provides functions to calculate descriptors

    def forward(self, x, neighlist, xneigh, indices, atoms_per_structure, types, unique_i, device, dtype=torch.float32):
        """
        Forward pass through the PyTorch network model, calculating both energies and forces.

        Attributes
        ----------

        x: torch.Tensor.float
            Array of descriptors for this batch

        neighlist: torch.Tensor.long
            Sparse neighlist for this batch

        indices: torch.Tensor.long
            Array of indices upon which to contract pairwise energies, for this batch

        atoms_per_structure: torch.Tensor.long
            Number of atoms per configuration for this batch

        types: torch.Tensor.long
            Atom types starting from 0, for this batch

        unique_i: atoms i for all atoms in this batch indexed starting from 0 to (natoms_batch-1)
        
        dtype: torch.float32
            Data type used for torch tensors, default is torch.float32 for easy training, but we set 
            to torch.float64 for finite difference tests to ensure correct force calculations.

        device: pytorch accelerator device object

        """

        # construct Bessel basis

        rbf = self.bessel.radial_bessel_basis(x, neighlist, unique_i, xneigh)
        assert(rbf.size()[0] == neighlist.size()[0])

        # this basis needs to be input to a network for each pair
        # calculate pairwise energies

        eij = self.networks[0](rbf)

        cutoff_functions = self.bessel.cutoff_function(x, neighlist, unique_i, xneigh)

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

        # TODO for now we divide energy by 2, to fix this we need to incorporate the LAMMPS 
        # neighlist-transformed positions in the forward pass

        predicted_energy_total = 0.5*predicted_energy_total

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
        from fitsnap3lib.lib.neural_networks.write import IgnoreElems, TorchWrapper, ElemwiseModels

        # self.network_architecture0 is network model for the first element type

        if self.n_elem == 1:
            print("Single element, saving model with IgnoreElems ML-IAP wrapper")
            model = IgnoreElems(self.network_architecture0)
        else:
            print("Multi element, saving model with ElemwiseModels ML-IAP wrapper")
            model = ElemwiseModels(self.networks, self.n_elem)
        linked_model = TorchWrapper(model, n_descriptors=self.num_descriptors, n_elements=self.n_elem)
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
