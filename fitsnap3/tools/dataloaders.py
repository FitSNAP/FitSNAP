import torch.utils.data
from torch.utils.data import DataLoader
from sys import float_info
import numpy as np


class InRAMDataset(torch.utils.data.Dataset):
    """Load A matrix Dataset from RAM"""

    def __init__(self, a_matrix, b, c, dbirj, natoms_per_config, dbdrindx, number_of_dbirj_rows, unique_j_indices, indices=None):
        """
        Args:
            a_matrix (numpy array): Matrix of descriptors with shape (Features, Descriptors)
            b (numpy array): Array of feature truth values with shape (Features, )
            c (numpy array): Array of force truth values with shape (nconfigs*natoms*3, )
            dbirj (numpy array): Array of dBi/dRj values organized as documented in compute snap
            natoms_per_config: Array of natoms for each config
            dbdrindx: array of indices corresponding to dbirj as documented in compute snap
            number_of_dbirj_rows: number of dbirj rows per config
            unique_j_indices: unique indices of dbirj componenents, will be used for force contraction.
            indices (numpy array): Array of indices that represent which atoms belong to which configs
        """


        """
        self.descriptors = a_matrix
        self.targets = b
        self.indices = indices
        self._length = None
        if self.indices is None:
            self._find_indices()

        print(len(a_matrix))
        print(len(b))
        assert len(a_matrix) == len(b) == len(self.indices)
        """
        self.descriptors = a_matrix
        self.targets = b
        self.target_forces = c
        self.dbirj = dbirj
        self.natoms_per_config = natoms_per_config
        self.dbdrindx = dbdrindx
        self.number_of_dbirj_rows = number_of_dbirj_rows
        self.unique_j_indices = unique_j_indices
        self.indices = indices
        self._length = None
        if self.indices is None:
            self._find_indices()
        #print(self.indices)
        #print(np.shape(self.descriptors))
        print(np.shape(self.dbirj))
        print(np.shape(self.dbdrindx))
        print(np.shape(self.number_of_dbirj_rows))
        print(self.number_of_dbirj_rows)


    def __len__(self):
        return self._length

    def __getitem__(self, idx):
        pass

    def _find_indices(self):
        """
        This is meant to be a temporary fix to the shortcomings of not using a distributed dataframe.
        Searches through targets and finds non-zeros, which will be the start of a new index.
        If a config ever has an energy of zero, this will not work.

        UPDATED:
        This shows which elements of the descriptors ('a'), targets ('b'), and other arrays belong to which config.
        These are needed for the __getitem__ function.
        """
        self.indices = []


        # Create indices for descriptors
        self.indices_descriptors = []
        config_indx = 0
        for natoms in self.natoms_per_config:
            for i in range(0,natoms):
                self.indices_descriptors.append(config_indx)
            config_indx = config_indx + 1
        self.indices_descriptors = np.array(self.indices_descriptors).astype(np.int32)
        #print(self.indices_descriptors)

        # Create indices for targets
        self.indices_targets = []
        config_indx = 0
        for natoms in self.natoms_per_config:
            self.indices_targets.append(config_indx)
            #for i in range(0,3*natoms):
            #    self.indices_targets.append(config_indx)
            config_indx = config_indx + 1
        self.indices_targets = np.array(self.indices_targets).astype(np.int32)
        #print(self.indices_targets)

        # Create indices for target forces
        self.indices_target_forces = []
        config_indx = 0
        for natoms in self.natoms_per_config:
            #self.indices_target_forces.append(config_indx)
            for i in range(0,3*natoms):
                self.indices_target_forces.append(config_indx)
            config_indx = config_indx + 1
        self.indices_target_forces = np.array(self.indices_target_forces).astype(np.int32)
        #print(self.indices_target_forces[0:163])

        # Create indices for dbirj and dbdrindx and unique_j_indices
        self.indices_dbirj = []
        config_indx = 0
        for ndbdr in self.number_of_dbirj_rows:
            for i in range(0,ndbdr):
                self.indices_dbirj.append(config_indx)
            config_indx = config_indx + 1
        self.indices_dbirj = np.array(self.indices_dbirj).astype(np.int32)
        #print(self.indices_dbirj[0:3301])


        i = -1
        for target in self.targets:
            if -float_info.epsilon > target or target > float_info.epsilon:
                i += 1
            self.indices.append(i)
        self.indices = np.array(self.indices)
        #self._length = len(np.unique(self.indices))

        # Set length to be number of configs for the __len__ function.
        self._length = np.shape(self.natoms_per_config)[0]


class InRAMDatasetPyTorch(InRAMDataset):
    """Load A matrix Dataset from RAM"""

    def __getitem__(self, idx):
        #print(idx)
        """
        config_descriptors = torch.tensor(self.descriptors[self.indices == idx]).float()
        target = torch.tensor(np.sum(self.targets[self.indices == idx])).float()
        number_of_atoms = torch.tensor(config_descriptors.size(0)).int()
        dbirj = torch.tensor of dbirj values
        dbdrindx = array of ints, indices corresponding to dbirj
        unique_j_indices = unique indices of j in dbdrindx, used for force contraction
        indices = torch.tensor([idx] * number_of_atoms)
        """
        config_descriptors = torch.tensor(self.descriptors[self.indices_descriptors == idx]).float()
        #print(config_descriptors.size())
        target = torch.tensor(self.targets[self.indices_targets == idx]).float()
        target_forces = torch.tensor(self.target_forces[self.indices_target_forces == idx]).float()
        #print(target.size())
        number_of_atoms = torch.tensor(self.natoms_per_config[idx])
        assert self.natoms_per_config[idx] == config_descriptors.size(0)
        dbirj = torch.tensor(self.dbirj[self.indices_dbirj == idx]).float()
        dbdrindx = torch.tensor(self.dbdrindx[self.indices_dbirj == idx]).long()
        unique_j_indices = torch.tensor(self.unique_j_indices[self.indices_dbirj == idx]).long()
        #print(unique_j_indices)
        indices = torch.tensor([idx] * number_of_atoms)
        configuration = {'x': config_descriptors,
                         'y': target, #target.reshape(-1),
                         'y_forces': target_forces,
                         'noa': number_of_atoms.reshape(-1), #number_of_atoms.reshape(-1),
                         'i': indices,
                         'dbirj': dbirj,
                         'dbdrindx': dbdrindx,
                         'unique_j': unique_j_indices}
        return configuration


def torch_collate(batch):
    """
    Collate batch of data, which collates a stack of configurations from Dataset into a batch
    """
    batch_of_descriptors = torch.cat([conf['x'] for conf in batch], dim=0)
    batch_of_targets = torch.cat([conf['y'] for conf in batch], dim=0)
    batch_of_target_forces = torch.cat([conf['y_forces'] for conf in batch], dim=0)
    #print([conf['noa'] for conf in batch])
    number_of_atoms = torch.cat([conf['noa'] for conf in batch], dim=0)
    indices = torch.cat([conf['i'] for conf in batch], dim=0) % len(batch)
    batch_of_dbirj = torch.cat([conf['dbirj'] for conf in batch], dim=0)
    batch_of_dbdrindx = torch.cat([conf['dbdrindx'] for conf in batch], dim=0)
    batch_of_unique_j = torch.cat([conf['unique_j'] for conf in batch], dim=0)
    # Subtract first index of batch of unique j so that we can contract properly on this batch
    batch_of_unique_j = batch_of_unique_j - batch_of_unique_j[0]
    #print(batch_of_unique_j)

    collated_batch = {'x': batch_of_descriptors,
                      'y': batch_of_targets,
                      'y_forces': batch_of_target_forces,
                      'noa': number_of_atoms,
                      'i': indices,
                      'dbirj': batch_of_dbirj,
                      'dbdrindx': batch_of_dbdrindx,
                      'unique_j': batch_of_unique_j}

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
