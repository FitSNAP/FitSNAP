import torch
from torch import from_numpy
from torch.nn import Parameter

"""
Don't import mliap package, see bug: https://github.com/lammps/lammps/issues/3204
This will not be a problem for now, because we only use the following two MLIAP
features for writing LAMMPS pytorch files.
"""
#from lammps.mliap.pytorch import IgnoreElems, TorchWrapper


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
    FitSNAP PyTorch Neural Network Architecture Model
    Currently only fits on energies
    """

    def __init__(self, network_architecture, descriptor_count, energy_weight, force_weight, n_elements=1):
        """
        Saves lammps ready pytorch model.

            Parameters:
                network_architecture : A nn.Sequential network architecture
                descriptor_count (int): Length of descriptors for an atom
                energy_weight (float): Weight of energy in loss function, used
                                       for determining if energy is fit or not
                force_weight (float): Weight of force in loss function, used
                                      for determining if energy is fit or not
                n_elements (int): Number of differentiable atoms types

        """
        super().__init__()
        self.network_architecture = network_architecture
        self.desc_len = descriptor_count
        self.n_elem = n_elements

        self.energy_bool = True
        self.force_bool = True
        if (energy_weight==0.0):
            self.energy_bool = False
        if (force_weight==0.0):
            self.force_bool = False

    def forward(self, x, xd, indices, atoms_per_structure, xd_indx, unique_j):
        """
        Saves lammps ready pytorch model.

            Parameters:
                x (tensor of floats): Array of descriptors
                xd (tensor of floats): Array of descriptor derivatives dDi/dRj
                indices (tensor of ints): Array of indices upon which to contract per atom energies
                atoms_per_structure (tensor of ints): Number of atoms per configuration
                xd_indx (tensor of int64, long ints): array of indices corresponding to descriptor derivatives
                unique_j (tensor of int64, long ints): array of indices corresponding to unique atoms j in all batches of configs.
                                                       all forces in this batch will be contracted over these indices.

        """

        # calculate energies

        if (self.energy_bool):
            predicted_energy_total = torch.zeros(atoms_per_structure.size())
            predicted_energy_total.index_add_(0, indices, self.network_architecture(x).squeeze())
        else:
            predicted_energy_total = None

        # calculate forces

        if (self.force_bool):
            nd = x.size()[1] # number of descriptors
            natoms = atoms_per_structure.sum() # Total number of atoms in this batch

            #x_indices = xd_indx[0::3]
            #y_indices = xd_indx[1::3]
            #z_indices = xd_indx[2::3]
    
            # boolean indices used to properly index descriptor gradients

            x_indices_bool = xd_indx[:,2]==0
            y_indices_bool = xd_indx[:,2]==1
            z_indices_bool = xd_indx[:,2]==2

            # neighbors i of atom j

            #neigh_indices_x = xd_indx[0::3,0]
            #neigh_indices_y = xd_indx[1::3,0] 
            #neigh_indices_z = xd_indx[2::3,0]

            neigh_indices_x = xd_indx[x_indices_bool,0]
            neigh_indices_y = xd_indx[y_indices_bool,0] 
            neigh_indices_z = xd_indx[z_indices_bool,0]

            dEdD = torch.autograd.grad(self.network_architecture(x), x, grad_outputs=torch.ones_like(self.network_architecture(x)), create_graph=True)[0]

            #print(dEdD.size())
            #print(neigh_indices_x.max())
            #print(neigh_indices_z.max())

            # extract proper dE/dD values to align with neighbors i of atoms j

            # these are true if no neighlist pruning (comment out the block with the "strip" comment in lammps_snap.py)
            #assert(torch.all(xd_indx[x_indices_bool,0] == xd_indx[0::3,0]))
            #assert(torch.all(xd_indx[y_indices_bool,0] == xd_indx[1::3,0]))      
            #assert(torch.all(xd_indx[z_indices_bool,0] == xd_indx[2::3,0]))   
            #assert(torch.all(xd[x_indices_bool,0] == xd[0::3,0]))
            #assert(torch.all(xd[y_indices_bool,0] == xd[1::3,0]))      
            #assert(torch.all(xd[z_indices_bool,0] == xd[2::3,0]))
 
            #dEdD = dEdD[neigh_indices_x, :] #.requires_grad_(True)
            dEdD_x = dEdD[neigh_indices_x, :]
            dEdD_y = dEdD[neigh_indices_y, :]
            dEdD_z = dEdD[neigh_indices_z, :]

            dDdRx = xd[x_indices_bool] #.requires_grad_(True)
            dDdRy = xd[y_indices_bool] #.requires_grad_(True)
            dDdRz = xd[z_indices_bool] #.requires_grad_(True)   

            # elementwise multiplication of dDdR and dEdD

            elementwise_x = torch.mul(dDdRx, dEdD_x) #.requires_grad_(True)
            elementwise_y = torch.mul(dDdRy, dEdD_y) #.requires_grad_(True)
            elementwise_z = torch.mul(dDdRz, dEdD_z) #.requires_grad_(True)

            # contract these elementwise components along rows with indices given by unique_j

            fx_components = torch.zeros(atoms_per_structure.sum(),nd) #.double() #.requires_grad_(True)
            fy_components = torch.zeros(atoms_per_structure.sum(),nd) #.double() #.requires_grad_(True)
            fz_components = torch.zeros(atoms_per_structure.sum(),nd) #.double() #.requires_grad_(True)

            # contract over unique j indices, which has same number of rows as dgrad
            # replace unique_j[a_indices_bool] with xd_indx[a_indices_bool, 1] and it's the same result for batch size of 1

            contracted_x = fx_components.index_add_(0,unique_j[x_indices_bool],elementwise_x) #.requires_grad_(True)
            contracted_y = fy_components.index_add_(0,unique_j[y_indices_bool],elementwise_y) #.requires_grad_(True)
            contracted_z = fz_components.index_add_(0,unique_j[z_indices_bool],elementwise_z) #.requires_grad_(True)

            # sum along bispectrum components to get force on each atom

            predicted_fx = torch.sum(contracted_x, dim=1) #.requires_grad_(True)
            predicted_fy = torch.sum(contracted_y, dim=1) #.requires_grad_(True)
            predicted_fz = torch.sum(contracted_z, dim=1) #.requires_grad_(True)

            # reshape to get 2D tensor

            predicted_fx = torch.reshape(predicted_fx, (natoms,1)) #.requires_grad_(True)
            predicted_fy = torch.reshape(predicted_fy, (natoms,1)) #.requires_grad_(True)
            predicted_fz = torch.reshape(predicted_fz, (natoms,1)) #.requires_grad_(True)

            # check that number of rows is equal to number of atoms

            assert predicted_fx.size()[0] == natoms

            # create a 3Nx1 array

            predicted_forces = torch.cat((predicted_fx,predicted_fy,predicted_fz), dim=1) #.requires_grad_(True)
            #predicted_forces = -1.*torch.flatten(predicted_forces).float() #.requires_grad_(True) # need to be float to match targets

            # don't need to multiply by -1 since compute snap already gives us negative derivatives

            predicted_forces = torch.flatten(predicted_forces).float() #.requires_grad_(True) # need to be float to match targets
            assert predicted_forces.size()[0] == 3*natoms

        else:
            predicted_forces = None

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

        print("WARNING: Not writing LAMMPS torch file due to ML-IAP bug: https://github.com/lammps/lammps/issues/3204")

        """
        model = self.network_architecture
        if self.n_elem == 1:
            model = IgnoreElems(self.network_architecture)
        linked_model = TorchWrapper(model, n_descriptors=self.desc_len, n_elements=self.n_elem)
        torch.save(linked_model, filename)
        """

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
