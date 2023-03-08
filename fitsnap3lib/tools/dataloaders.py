import torch.utils.data
from sys import float_info
import numpy as np
from torch.utils.data import DataLoader

class InRAMDataset(torch.utils.data.Dataset):
    """
    Parent class for storing and shuffling data, depending on the specific solver used. Child 
    classes are InRAMDatasetPyTorch and InRAMDatasetJAX, which process the list of Configuration 
    objects appropriately for their own solver methods.
    """

    def __init__(self, configs):
        """
        Initializer.

        Attributes
        ----------

        configs: list
            List of Configuration objects
        """

        self.configs = configs

        # TODO: could add more assertions for bug catching here

        for i, config in enumerate(self.configs):
            assert(np.shape(config.descriptors)[0] == config.natoms)

    def __len__(self):
        return len(self.configs)

    def __getitem__(self, idx):
        pass

class InRAMDatasetPyTorch(InRAMDataset):
    """
    
    """

    def __getitem__(self, idx):
        """
        Convert configuration quantities to tensors and return them, for a single configuration in a 
        batch. 

        TODO: We could eliminate this costly conversion by storing all of these as tensors instead 
        of numpy arrays from the beginning, when processing configs in the Calculator class.
        """

        config_descriptors = torch.tensor(self.configs[idx].descriptors).float()
        atom_types = torch.tensor(self.configs[idx].types).long()

        if (self.configs[idx].energy is not None):
            target = torch.tensor(self.configs[idx].energy).float().reshape(-1)
            # indexing 0th axis with None reshapes the tensor to be 2D for stacking later
            weights = torch.tensor(self.configs[idx].weights[None,:]).float()
        else:
            target = None
            weights = None

        if (self.configs[idx].forces is not None):
            target_forces = torch.tensor(self.configs[idx].forces).float()
            dgrad = torch.tensor(self.configs[idx].dgrad).float()
            dbdrindx = torch.tensor(self.configs[idx].dgrad_indices).long()
        else:
            target_forces = None
            dgrad = None
            dbdrindx = None

        if (self.configs[idx].pas is not None):
            # we are fitting per-atom scalars, don't use energies/forces
            assert(self.configs[idx].energy is None and self.configs[idx].forces is None)
            target = torch.tensor(self.configs[idx].pas).float()

        number_of_atoms = torch.tensor(self.configs[idx].natoms)

        configuration = {'x': config_descriptors,
                         'y': target, #target.reshape(-1),
                         'y_forces': target_forces,
                         'noa': number_of_atoms.reshape(-1), #number_of_atoms.reshape(-1),
                         't': atom_types,
                         'w': weights,
                         'dgrad': dgrad,
                         'dbdrindx': dbdrindx,
                         'configs': self.configs[idx]}

        return configuration


def torch_collate(batch):
    """
    Collate batch of data, which collates a stack of configurations from Dataset into a batch.
    """

    batch_of_descriptors = torch.cat([conf['x'] for conf in batch], dim=0)
    batch_of_types = torch.cat([conf['t'] for conf in batch], dim=0)
    number_of_atoms = torch.cat([conf['noa'] for conf in batch], dim=0)

    if (batch[0]['y'] is not None):
        # we fit to eneriges
        batch_of_targets = torch.cat([conf['y'] for conf in batch], dim=0)
    else:
        batch_of_targets = None

    if (batch[0]['w'] is not None):
        batch_of_weights = torch.cat([conf['w'] for conf in batch], dim=0)
    else:
        batch_of_weights = None
    
    if (batch[0]['y_forces'] is not None):
        # we fit to forces
        batch_of_target_forces = torch.cat([conf['y_forces'] for conf in batch], dim=0)
        batch_of_dgrad = torch.cat([conf['dgrad'] for conf in batch], dim=0)
        batch_of_dbdrindx = torch.cat([conf['dbdrindx'] for conf in batch], dim=0)
    else:
        batch_of_target_forces = None
        batch_of_dgrad = None
        batch_of_dbdrindx = None

    # make indices upon which to contract per-atom energies for this batch

    config_indices = torch.arange(len(batch))
    indices = torch.repeat_interleave(config_indices, number_of_atoms)

    # make batch of unique atoms j upon which force components will be contracted
    # these are indices j (0 to N-1 in a single config, for each atom in the entire batch, so that 
    # we treat the entire batch as a single config when calculating forces

    configs = [conf['configs'] for conf in batch] # batch of Configuration objects
    if (batch[0]['y_forces'] is not None):
        natoms_grow = 0
        unique_i_indices = []
        unique_j_indices = []
        for i, conf in enumerate(configs):
            unique_i_indices.append(torch.tensor(conf.dgrad_indices[:,0]+natoms_grow).long())
            unique_j_indices.append(torch.tensor(conf.dgrad_indices[:,1]+natoms_grow).long())
            natoms_grow += conf.natoms
        batch_of_unique_i = torch.cat(unique_i_indices, dim=0)
        batch_of_unique_j = torch.cat(unique_j_indices, dim=0)
    else:
        batch_of_unique_i = None
        batch_of_unique_j = None

    # batch of testing bools to check that we have proper training/testing configs:
    
    batch_of_testing_bools = [conf.testing_bool for conf in configs]

    # TODO: Add cheap asserts to catch possible bugs
    # use the following for debugging
    # filenames and associated quantities can be checked by confirming values in files
    #batch_of_filenames = [conf.filename for conf in configs]
    #print(batch_of_filenames)
    #print(number_of_atoms)
    #print(indices)
    #print(batch_of_targets*number_of_atoms) # should match energy in file
    #print(batch_of_target_forces)
    #print(batch_of_unique_j)

    collated_batch = {'x': batch_of_descriptors,
                      't': batch_of_types,
                      'y': batch_of_targets,
                      'y_forces': batch_of_target_forces,
                      'noa': number_of_atoms,
                      'w': batch_of_weights,
                      'i': indices,
                      'dgrad': batch_of_dgrad,
                      'dbdrindx': batch_of_dbdrindx,
                      'unique_j': batch_of_unique_j,
                      'unique_i': batch_of_unique_i,
                      'testing_bools': batch_of_testing_bools}

    return collated_batch


class InRAMDatasetJAX(InRAMDataset):
    """
    Overload __getitem__ PyTorch InRAMDataset class
    Note: jit precompiles for different sub(A) lengths
    each time a new length is encountered the update will be slower by 3 orders of magnitude
    """

    def __getitem__(self, idx):
        config_descriptors = self.descriptors[self.indices == idx]
        target = np.sum(self.targets[self.indices == idx])
        number_of_atoms = len(config_descriptors)
        indices = np.array([idx] * number_of_atoms)
        configuration = {'x': config_descriptors,
                         'y': target.reshape(-1),
                         'noa': number_of_atoms,
                         'i': indices}
        return configuration


def jax_collate(batch):
    """
    Collate batch of data, which collates a stack of configurations from Dataset into a batch
    """
    import jax.numpy as jnp
    batch_of_descriptors = jnp.array(np.concatenate([conf['x'] for conf in batch], axis=0))
    batch_of_targets = jnp.array([conf['y'] for conf in batch])
    number_of_atoms = jnp.array([conf['noa'] for conf in batch]).reshape((-1, 1))
    indices = jnp.array(np.concatenate([conf['i'] for conf in batch]) % len(batch))

    collated_batch = {'x': batch_of_descriptors,
                      'y': batch_of_targets,
                      'noa': number_of_atoms,
                      'i': indices,
                      'nseg': len(batch)}
    return collated_batch
