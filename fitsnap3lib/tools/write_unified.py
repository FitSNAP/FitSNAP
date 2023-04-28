"""
interface for creating LAMMPS MLIAP Unified models.
"""
import pickle

import numpy as np
import torch
torch.set_default_dtype(torch.float32)

from fitsnap3lib.lib.neural_networks.pairwise import FitTorch

from lammps.mliap.mliap_unified_abc import MLIAPUnified

class MLIAPInterface(MLIAPUnified):
    """
    Class for creating ML-IAP Unified model based on hippynn graphs.
    """
    def __init__(self, network, element_types, ndescriptors=1,
                 model_device=torch.device("cpu")):
        """
        :param network: pytorch model state dict
        :param element_types: list of atomic symbols corresponding to element types
        :param ndescriptors: the number of descriptors to report to LAMMPS
        :param model_device: the device to send torch data to (cpu or cuda)
        """
        super().__init__()
        self.network = network
        self.network.to(torch.float64)
        self.element_types = element_types
        self.ndescriptors = ndescriptors
        self.model_device = model_device
        

        # Build the calculator
        rc = 4.5
        self.rcutfac = 0.5*rc # Actual cutoff will be 2*rc
        #print(self.model.nparams)
        self.nparams = 10
        #self.rcutfac, self.species_set, self.graph = setup_LAMMPS()
        #self.nparams = sum(p.nelement() for p in self.graph.parameters())
        #self.graph.to(torch.float64)

    def compute_descriptors(self, data):
        pass

    def as_tensor(self,array):
        return torch.as_tensor(array,device=self.model_device)

    def compute_gradients(self, data):
        pass

    def compute_forces(self, data):
        #print(">>>>> hey!")
        #elems = self.as_tensor(data.elems).type(torch.int64).reshape(1, data.ntotal)
        elems = self.as_tensor(data.elems).type(torch.int64)
        #z_vals = self.species_set[elems+1]
        pair_i = self.as_tensor(data.pair_i).type(torch.int64)
        pair_j = self.as_tensor(data.pair_j).type(torch.int64)
        rij = self.as_tensor(data.rij).type(torch.float64).requires_grad_(True)
        nlocal = self.as_tensor(data.nlistatoms) 
        #print(dir(data))

        """
        #positions = torch.tensor(config.positions).requires_grad_(True)
        positions = torch.tensor(config.x).requires_grad_(True)
        xneigh = torch.tensor(config.xneigh)
        transform_x = torch.tensor(config.transform_x)
        atom_types = torch.tensor(config.types).long()
        target = torch.tensor(config.energy).reshape(-1)
        # indexing 0th axis with None reshapes the tensor to be 2D for stacking later
        weights = torch.tensor(config.weights[None,:])
        target_forces = torch.tensor(config.forces)
        num_atoms = torch.tensor(config.natoms)
        neighlist = torch.tensor(config.neighlist).long()

        # convert quantities to desired dtype
  
        positions = positions.to(dtype)
        transform_x = transform_x.to(dtype)
        target = target.to(dtype)
        weights = weights.to(dtype)
        target_forces = target_forces.to(dtype)

        # make indices upon which to contract per-atom energies for this config

        config_indices = torch.arange(1).long() # this usually has len(batch) as arg in dataloader
        indices = torch.repeat_interleave(config_indices, neighlist.size()[0]) # config indices for each pair
        unique_i = neighlist[:,0]
        unique_j = neighlist[:,1]
        
        # need to unsqueeze num_atoms to get a tensor of definable size

        (energies,forces) = self.model(positions, neighlist, transform_x, 
                                      indices, num_atoms.unsqueeze(0), 
                                      atom_types, unique_i, unique_j, self.device, dtype)
        """

        #config_indices = torch.arange(1).long() # this usually has len(batch) as arg in dataloader
        #indices = torch.repeat_interleave(config_indices, rij.size()[0]) # config indices for each pair

        (total_energy, fij) = self.network(rij, None, None, None, nlocal, elems, pair_i, pair_j, "cpu", dtype=torch.float64, mode="lammps")

        # Test if we are using lammps-kokkos or not. Is there a more clear way to do that?
        if isinstance(data.elems,np.ndarray):
            return_device = 'cpu'
        else:
            # Hope that kokkos device and pytorch device are the same (default cuda)
            return_device = elems.device

        # TODO: Output peratom energies and use these variables here
        #atom_energy = atom_energy.squeeze(1).detach().to(return_device)
        total_energy = total_energy.detach().to(return_device)

        f = self.as_tensor(data.f)
        fij = fij.type(f.dtype).detach().to(return_device)
        
        if return_device=="cpu":
            fij = fij.numpy()
            #data.eatoms = atom_energy.numpy().astype(np.double)
        else:
            #eatoms = torch.as_tensor(data.eatoms,device=return_device)
            #eatoms.copy_(atom_energy)
            pass
         
        data.update_pair_forces(fij)
        data.energy = total_energy.item()
 
        pass

def setup_LAMMPS(energy):
    """

    :param energy: energy node for lammps interface
    :return: graph for computing from lammps MLIAP unified inputs.
    """

    model = TheModelClass(*args, **kwargs)

    save_state_dict = torch.load("Ta_Pytorch.pt")
    model.load_state_dict(save_state_dict["model_state_dict"])


    #model.load_state_dict(torch.load(PATH))
    model.eval()
    
    #model.eval()
    return model
