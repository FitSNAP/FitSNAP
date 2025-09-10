"""
This file houses PyTorch models for fitting to per-atom scalars (PAS).
It's similar to the basic pytorch.py used for fitting energies/forces, but convenient to separate 
into a separate file because there are many differences in the forward pass.

We use the `create_networks` function in pytorch.py since the structure is the same.
"""

import torch

"""
Try to import mliap package after: https://github.com/lammps/lammps/pull/3388
See related bug: https://github.com/lammps/lammps/issues/3204
For now we only use the following two MLIAP features for writing LAMMPS-ready pytorch models.
"""

class FitTorchPAS(torch.nn.Module):
    """
    FitSNAP PyTorch network model to arbitrary per-atom scalars.

    Attributes:
        networks (:obj:`list` of :obj:`torch.nn.Sequential`): Network architectures created by 
                                                              create_torch_network function.
        descriptor_count (:obj:`int`): Number of descriptors for an atom.
        n_elements (:obj:`int`): Number of element types.
        multi_element_option (:obj:`int`): Setting for how to deal with multiple element types. 
    """

    def __init__(self, networks, descriptor_count, n_elements=1, multi_element_option=1, dtype=torch.float32):
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

    def forward(self, x, atoms_per_structure, types, device, dtype=torch.float32):
        """

        FitSNAP PyTorch network model to arbitrary per-atom scalars.

        Attributes:
            x (:obj:`torch.Tensor.float`): Atom-centered descriptors for all atoms in this batch.
            atoms_per_structure (:obj:`torch.Tensor.long`): Number of atoms per configuration for 
                                                            this batch.
            types (:obj:`torch.Tensor.long`): Atom types starting from 0, for this batch.
            device: pytorch accelerator device object
            dtype (:obj:`torch.float32`, optional): Data type used for torch tensors, default is 
                                                   torch.float32 for training, but we set to 
                                                   torch.float64 for finite difference tests to 
                                                   ensure correct force calculations.

        Returns:
            tuple: Predicted per-atom scalars for this batch.                                                                          
        """

        if (self.multi_element_option==1):
            per_atom_scalars = self.network_architecture0(x).flatten()
   
        elif (self.multi_element_option==2):
            # Working, but not ideal due to stacking
            #atom_indices = torch.arange(x.size()[0])
            #per_atom_energies = torch.stack([self.networks[i](x) 
            #                                 for i in range(self.n_elem)])[types,atom_indices]

            # Slightly slower, but more general

            per_atom_scalars = torch.zeros(types.size(dim=0), dtype=dtype).to(device)
            given_elems, elem_indices = torch.unique(types, return_inverse=True)
            for i, elem in enumerate(given_elems): 
                per_atom_scalars[elem_indices == i] = self.networks[elem](x[elem_indices == i]).flatten()

        return per_atom_scalars

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
