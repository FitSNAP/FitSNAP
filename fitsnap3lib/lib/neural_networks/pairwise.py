import torch
from torch import from_numpy
from torch.nn import Parameter
import math
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

    def __init__(self, networks, descriptor_count, energy_weight, force_weight, n_elements=1, multi_element_option=1, dtype=torch.float32):
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
         
        self.desc_len = descriptor_count
        self.n_elem = n_elements
        self.multi_element_option = multi_element_option

        self.energy_bool = True
        self.force_bool = True
        if (energy_weight==0.0):
            self.energy_bool = False
        if (force_weight==0.0):
            self.force_bool = False

    def calculate_rij(self, x, neighlist, xneigh):
        """
        Calculate radial distance between all pairs

        Attributes
        ----------

        x: torch.Tensor.float
            Array of positions for this batch
        
        neighlist: torch.Tensor.long
            Sparse neighlist for this batch

        xneigh: torch.Tensor.float
            Array of neighboring positions (ghost atoms) for this batch, 
            lined up with neighlist[:,1]

        Returns
        -------

        rij: torch.Tensor.float
            Pairwise distance tensor with size (number_neigh, 1)
        """

        # calculate all pairwise distances

        diff = x[neighlist[:,0]] - xneigh
        rij = torch.linalg.norm(diff, dim=1)

        rij = rij.unsqueeze(1)        

        return rij

    def calculate_bessel(self, rij, n):
        """
        Calculate radial bessel functions for all pairs

        Attributes
        ----------

        rij: torch.Tensor.float
            Pairwise distance tensor with size (number_neigh, 1)

        n: torch.Tensor.float
            Integer in float form representing Bessel radial parameter n

        Returns
        -------

        rbf: torch.Tensor.float
            Radial Bessel function calculation for base n, has size (number_neigh, 1)
        """

        # calculate Bessel

        c = 3.0 # cutoff
        pi = torch.tensor(math.pi)
        two_over_c = torch.tensor(2./c)
        rbf = torch.div(torch.sqrt(two_over_c)*torch.sin(((n*pi)/c)*rij), rij)     

        return rbf

    def radial_bessel_basis(self, x, neighlist, xneigh):
        """
        Calculate radial Bessel basis functions.

        Attributes
        ----------

        x: torch.Tensor.float
            Array of positions for this batch
        
        neighlist: torch.Tensor.long
            Sparse neighlist for this batch


        """

        num_rbf = 3 # number of radial basis functions
                    # e.g. 3 includes n = 1,2,3

        rij = self.calculate_rij(x, neighlist, xneigh)

        basis = torch.cat([self.calculate_bessel(rij, n) for n in range(1,num_rbf+1)], dim=1)

        """
        # calculate all pairwise distances

        diff = x[neighlist[:,0]] - xneigh
        rij = torch.linalg.norm(diff, dim=1)

        print(neighlist[0:7,0])

        print(diff[0:6,:])
        print(diff[0:6,0])
        print(rij[0:6])
        manual_derivative = torch.div(diff[0:6,0], rij[0:6])
        summ = torch.sum(manual_derivative)
        print(summ)

        c = 5.0 # cutoff
        num_rbf = 3 # number of radial bessel basis functions
        #print(rij.size())
        #print(rij.unsqueeze(1).size())

        rij_unsqueezed = rij.unsqueeze(1)
        print(rij_unsqueezed.size())


        basis = rij

        #test = x*2
        """

        return basis

    def forward(self, x, neighlist, xneigh, indices, atoms_per_structure, types, device, dtype=torch.float32):
        """
        Forward pass through the PyTorch network model, calculating both energies and forces.

        Attributes
        ----------

        x: torch.Tensor.float
            Array of descriptors for this batch

        neighlist: torch.Tensor.long
            Sparse neighlist for this batch

        indices: torch.Tensor.long
            Array of indices upon which to contract per atom energies, for this batch

        atoms_per_structure: torch.Tensor.long
            Number of atoms per configuration for this batch

        types: torch.Tensor.long
            Atom types starting from 0, for this batch
        
        dtype: torch.float32
            Data type used for torch tensors, default is torch.float32 for easy training, but we set 
            to torch.float64 for finite difference tests to ensure correct force calculations.

        device: pytorch accelerator device object

        """

        #print("^^^^^ pairwise!")
        #print(neighlist)

        # construct Bessel basis

        rbf = self.radial_bessel_basis(x,neighlist, xneigh)
        assert(rbf.size()[0] == neighlist.size()[0])

        # this basis needs to be input to a network for each pair

        #print(rbf.size())
        #print(self.networks[0])

        # calculate pairwise energies

        eij = self.networks[0](rbf)
        #print(eij.squeeze().size())

        # calculate energy per config
        #print(indices)
        predicted_energy_total = torch.zeros(atoms_per_structure.size(), dtype=dtype).to(device)
        #print(predicted_energy_total.size())
        predicted_energy_total.index_add_(0, indices, eij.squeeze())
        predicted_energy_total = 100.*predicted_energy_total

        #print(predicted_energy_total.size())
        #print(predicted_energy_total)

        # calculate spatial gradients

        gradients_wrt_x = torch.autograd.grad(predicted_energy_total, 
                                    x, 
                                    grad_outputs=torch.ones_like(predicted_energy_total), 
                                    create_graph=True)[0]

        #print(gradients_wrt_x.size())

        # force is negative gradient

        predicted_forces = -1.0*gradients_wrt_x

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
        linked_model = TorchWrapper(model, n_descriptors=self.desc_len, n_elements=self.n_elem)
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
