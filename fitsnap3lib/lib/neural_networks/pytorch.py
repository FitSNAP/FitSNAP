import torch
from torch import from_numpy
from torch.nn import Parameter
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

    # TODO: Make biases optional, need to also optionally ignore bias standardiziation in 
    # solvers.pytorch.
    try:
        layers.append(torch.nn.Linear(layer_sizes[0], layer_sizes[0], bias=True))
        for i, layer in enumerate(layer_sizes):
            layers.append(torch.nn.Linear(layer_sizes[i], layer_sizes[i + 1], bias=True))
            layers.append(torch.nn.Softplus())
            #layers.append(torch.nn.Sigmoid())
    except IndexError:
        layers.pop()

    # This adds all linear layers only.
    # TODO: Make linear models an option with gradient descent. 
    """
    print(len(layer_sizes))
    layers.append(torch.nn.Linear(layer_sizes[0], layer_sizes[0],bias=False))
    for i in range(len(layer_sizes)-1):
        print(f"{i}")
        print(f"{i} {layer_sizes[i]} {layer_sizes[i + 1]}")
        layers.append(torch.nn.Linear(layer_sizes[i], layer_sizes[i + 1],bias=False))
    """

    return torch.nn.Sequential(*layers)


class FitTorch(torch.nn.Module):
    """
    FitSNAP PyTorch network model. 

    Args:
        networks (list): List of nn.Sequential network architectures. Each list element is a 
            different network type if multi-element option = 2.
        descriptor_count (int): Length of descriptors for an atom.
        force_bool (bool): Boolean telling whether to calculate forces or not.
        n_elements (int): Number of differentiable atoms types 
        multi_element_option (int): Option for which multi-element network model to use.
    """

    def __init__(self, networks, descriptor_count, force_bool, n_elements=1, multi_element_option=1, dtype=torch.float32):
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
        self.force_bool = force_bool

    def forward(self, x, xd, indices, atoms_per_structure, types, xd_indx, unique_j, unique_i, device, dtype=torch.float32):
        """
        Forward pass through the PyTorch network model, calculating both energies and forces.

        Args:
            x (torch.Tensor.float): Array of descriptors for this batch.
            xd (torch.Tensor.float): Array of descriptor derivatives dDi/dRj for this batch.
            indices (torch.Tensor.long): Array of indices upon which to contract per atom energies, 
                for this batch.
            atoms_per_structure (torch.Tensor.long): Number of atoms per configuration for this 
                batch.
            types (torch.Tensor.long): Atom types starting from 0, for this batch.
            xd_indx (torch.Tensor.long): Array of indices corresponding to descriptor derivatives, 
                for this batch. These are concatenations of the direct LAMMPS dgradflag=1 output; we 
                rely on unique_j and unique_i for adjusted indices of this batch 
                (see dataloader.torch_collate).
            unique_j (torch.Tensor.long): Array of indices corresponding to unique atoms j in all 
                batches of configs. All forces in this batch will be contracted over these indices.
            unique_i (torch.Tensor.long): Array of indices corresponding to unique neighbors i in 
                all batches of configs. Forces on atoms j are summed over these neighbors and 
                contracted appropriately. 
            dtype (torch.float32): Data type used for torch tensors, default is torch.float32 for 
                easy training, but we set to torch.float64 for finite difference tests to ensure 
                correct force calculations.
            device: pytorch accelerator device object

        """   
        if (self.multi_element_option==1):
            per_atom_energies = self.network_architecture0(x)
        elif (self.multi_element_option==2):
            # Working, but not ideal due to stacking
            #atom_indices = torch.arange(x.size()[0])
            #per_atom_energies = torch.stack([self.networks[i](x) 
            #                                 for i in range(self.n_elem)])[types,atom_indices]

            # Slightly slower, but more general

            per_atom_energies = torch.zeros(types.size(dim=0), dtype=dtype).to(device)
            given_elems, elem_indices = torch.unique(types, return_inverse=True)
            for i, elem in enumerate(given_elems):
                per_atom_energies[elem_indices == i] = self.networks[elem](x[elem_indices == i]).flatten()

        # calculate energies

        if (self.energy_bool):
            predicted_energy_total = torch.zeros(atoms_per_structure.size(), dtype=dtype).to(device)
            predicted_energy_total.index_add_(0, indices, per_atom_energies.squeeze())
        else:
            predicted_energy_total = None

        # calculate forces

        if (self.force_bool):
            nd = x.size()[1] # number of descriptors
            natoms = atoms_per_structure.sum() # total number of atoms in this batch
    
            # boolean indices used to properly index descriptor gradients

            x_indices_bool = xd_indx[:,2]==0
            y_indices_bool = xd_indx[:,2]==1
            z_indices_bool = xd_indx[:,2]==2

            # neighbors i of atom j

            neigh_indices_x = unique_i[x_indices_bool]
            neigh_indices_y = unique_i[y_indices_bool] 
            neigh_indices_z = unique_i[z_indices_bool]

            dEdD = torch.autograd.grad(per_atom_energies, 
                                       x, 
                                       grad_outputs=torch.ones_like(per_atom_energies), 
                                       create_graph=True)[0]

            # extract proper dE/dD values to align with neighbors i of atoms j
 
            dEdD_x = dEdD[neigh_indices_x, :]
            dEdD_y = dEdD[neigh_indices_y, :]
            dEdD_z = dEdD[neigh_indices_z, :]

            dDdRx = xd[x_indices_bool] 
            dDdRy = xd[y_indices_bool] 
            dDdRz = xd[z_indices_bool] 

            # elementwise multiplication of dDdR and dEdD

            elementwise_x = torch.mul(dDdRx, dEdD_x) 
            elementwise_y = torch.mul(dDdRy, dEdD_y) 
            elementwise_z = torch.mul(dDdRz, dEdD_z) 

            # contract these elementwise components along rows with indices given by unique_j

            fx_components = torch.zeros(atoms_per_structure.sum(),nd, dtype=dtype).to(device) 
            fy_components = torch.zeros(atoms_per_structure.sum(),nd, dtype=dtype).to(device) 
            fz_components = torch.zeros(atoms_per_structure.sum(),nd, dtype=dtype).to(device) 

            # contract over unique j indices, which has same number of rows as dgrad 

            contracted_x = fx_components.index_add_(0,unique_j[x_indices_bool],elementwise_x) 
            contracted_y = fy_components.index_add_(0,unique_j[y_indices_bool],elementwise_y) 
            contracted_z = fz_components.index_add_(0,unique_j[z_indices_bool],elementwise_z) 

            # sum along bispectrum components to get force on each atom

            predicted_fx = torch.sum(contracted_x, dim=1) 
            predicted_fy = torch.sum(contracted_y, dim=1) 
            predicted_fz = torch.sum(contracted_z, dim=1) 

            # reshape to get 2D tensor

            predicted_fx = torch.reshape(predicted_fx, (natoms,1)) 
            predicted_fy = torch.reshape(predicted_fy, (natoms,1)) 
            predicted_fz = torch.reshape(predicted_fz, (natoms,1)) 

            # check that number of rows is equal to number of atoms

            assert predicted_fx.size()[0] == natoms

            # create a 3Nx1 array

            predicted_forces = torch.cat((predicted_fx,predicted_fy,predicted_fz), dim=1) 

            # don't need to multiply by -1 since compute snap already gives us negative derivatives

            predicted_forces = torch.flatten(predicted_forces)
            assert predicted_forces.size()[0] == 3*natoms

        else:
            predicted_forces = None

        return (predicted_energy_total, predicted_forces)

    def import_wb(self, weights, bias):
        """
        Imports weights and bias into FitTorch model

        Args:
            weights (list of numpy array of floats): Network weights at each layer.
            bias (list of numpy array of floats): Network bias at each layer.

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

        Args:
            filename (str): Filename for lammps usable pytorch model.

        """
        
        #from lammps.mliap.pytorch import IgnoreElems, TorchWrapper, ElemwiseModels
        from fitsnap3lib.lib.neural_networks.write import IgnoreElems, TorchWrapper, ElemwiseModels

        # self.network_architecture0 is network model for the first element type

        if self.n_elem == 1:
            #print("Single element, saving model with IgnoreElems ML-IAP wrapper")
            model = IgnoreElems(self.network_architecture0)
        else:
            #print("Multi element, saving model with ElemwiseModels ML-IAP wrapper")
            model = ElemwiseModels(self.networks, self.n_elem)
        linked_model = TorchWrapper(model, n_descriptors=self.desc_len, n_elements=self.n_elem)
        torch.save(linked_model, filename)
        

    def load_lammps_torch(self, filename="FitTorch.pt"):
        """
        Loads lammps ready pytorch model.

        Args:
            filename (str): Filename of lammps usable pytorch model.

        """
        model_state_dict = torch.load(filename).state_dict()
        list_of_old_keys = [*model_state_dict.keys()]
        new_dict = self.state_dict()
        assert len(model_state_dict) == len(new_dict)
        for i, key in enumerate(new_dict.keys()):
            new_dict[key] = model_state_dict[list_of_old_keys[i]]
        self.load_state_dict(new_dict)
