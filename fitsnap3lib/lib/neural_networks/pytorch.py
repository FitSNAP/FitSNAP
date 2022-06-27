import torch
from torch import from_numpy
from torch.nn import Parameter
from lammps.mliap.pytorch import IgnoreElems, TorchWrapper


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

    def __init__(self, network_architecture, descriptor_count, n_elements=1):
        """
        Saves lammps ready pytorch model.

            Parameters:
                network_architecture : A nn.Sequential network architecture
                descriptor_count (int): Length of descriptors for an atom
                n_elements (int): Number of differentiable atoms types

        """
        super().__init__()
        self.network_architecture = network_architecture
        self.desc_len = descriptor_count
        self.n_elem = n_elements

    def forward(self, x, indices, atoms_per_structure):
        """
        Saves lammps ready pytorch model.

            Parameters:
                x (tensor of floats): Array of descriptors
                indices (tensor of ints): Array of indices upon which to contract per atom energies
                atoms_per_structure (tensor of ints): Number of atoms per configuration

        """

        predicted_energy_total = torch.zeros(atoms_per_structure.size())
        predicted_energy_total.index_add_(0, indices, self.network_architecture(x).squeeze())
        return predicted_energy_total

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
        model = self.network_architecture
        if self.n_elem == 1:
            model = IgnoreElems(self.network_architecture)
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
