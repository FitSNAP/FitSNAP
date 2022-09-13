import torch.utils.data
from torch.utils.data import DataLoader
from sys import float_info
import numpy as np


class InRAMDataset(torch.utils.data.Dataset):
    """Load A matrix Dataset from RAM"""

    def __init__(self, a_matrix, b, c, t, w, dgrad, natoms_per_config, dbdrindx, 
                 number_of_dgrad_rows, unique_j_indices, indices=None):
        """
        Args:
            a_matrix (numpy array): Matrix of descriptors with shape (natoms*nconfigs, ndescriptors)
            b (numpy array): Array of energy truth values with shape (nconfigs, )
            c (numpy array): Array of force truth values with shape (nconfigs*natoms*3, )
            t (numpy array): Array of atom types with shape (natoms*nconfigs, )
            w (numpy array): Array of weights for energy (1st col) and force (2nd col) with shape
                             (nconfigs, 2)
            dgrad (numpy array): Array of dBi/dRj values organized as documented in compute snap
            natoms_per_config: Array of natoms for each config
            dbdrindx: array of indices corresponding to dgrad as documented in compute snap
            number_of_dgrad_rows: number of dgrad rows per config
            unique_j_indices: unique indices of dgrad componenents, will be used for force contraction.
            indices (numpy array): Array of indices that represent which atoms belong to which configs
        """

        self.descriptors = a_matrix
        self.targets = b
        self.target_forces = c
        self.atom_types = t - 1 # make atom types start at zero
        self.weights = w
        self.dgrad = dgrad
        self.natoms_per_config = natoms_per_config
        self.dbdrindx = dbdrindx
        self.number_of_dgrad_rows = number_of_dgrad_rows
        self.unique_j_indices = unique_j_indices
        self.indices = indices
        self._length = None
        if self.indices is None:
            self._find_indices()
        #print(self.indices)

        # TODO: could add some sort of assertion here


    def __len__(self):
        return self._length

    def __getitem__(self, idx):
        pass

    def _find_indices(self):
        """
        This is meant to be a temporary fix to the shortcomings of not using a distributed dataframe.
        Searches through targets and finds non-zeros, which will be the start of a new index.
        If a config ever has an energy of zero, this will not work.

        This shows which elements of the descriptors ('a'), targets ('b'), and other arrays belong to which config.
        These are needed for the __getitem__ function.
        """
        self.indices = []

        # create indices for descriptors and atom types

        self.indices_descriptors = []
        config_indx = 0
        for natoms in self.natoms_per_config:
            for i in range(0,natoms):
                self.indices_descriptors.append(config_indx)
            config_indx = config_indx + 1
        self.indices_descriptors = np.array(self.indices_descriptors).astype(np.int32)

        # create indices for targets

        self.indices_targets = []
        config_indx = 0
        for natoms in self.natoms_per_config:
            self.indices_targets.append(config_indx)
            config_indx = config_indx + 1
        self.indices_targets = np.array(self.indices_targets).astype(np.int32)

        # create indices for target forces

        self.indices_target_forces = []
        config_indx = 0
        for natoms in self.natoms_per_config:
            for i in range(0,3*natoms):
                self.indices_target_forces.append(config_indx)
            config_indx = config_indx + 1
        self.indices_target_forces = np.array(self.indices_target_forces).astype(np.int32)
        print(self.indices_target_forces[0:20])

        # create indices for dgrad and dbdrindx and unique_j_indices

        self.indices_dgrad = []
        config_indx = 0
        for ndbdr in self.number_of_dgrad_rows:
            for i in range(0,ndbdr):
                self.indices_dgrad.append(config_indx)
            config_indx = config_indx + 1
        self.indices_dgrad = np.array(self.indices_dgrad).astype(np.int32)

        i = -1
        for target in self.targets:
            if -float_info.epsilon > target or target > float_info.epsilon:
                i += 1
            self.indices.append(i)
        self.indices = np.array(self.indices)
        #self._length = len(np.unique(self.indices))

        # set length to be number of configs for the __len__ function

        self._length = np.shape(self.natoms_per_config)[0]


class InRAMDatasetPyTorch(InRAMDataset):
    """Load A matrix Dataset from RAM"""

    def __getitem__(self, idx):
        #print(idx)
        """
        config_descriptors = torch.tensor(self.descriptors[self.indices == idx]).float()
        target = torch.tensor(np.sum(self.targets[self.indices == idx])).float()
        number_of_atoms = torch.tensor(config_descriptors.size(0)).int()
        dgrad = torch.tensor of dgrad values
        dbdrindx = array of ints, indices corresponding to dgrad
        unique_j_indices = unique indices of j in dbdrindx, used for force contraction
        indices = torch.tensor([idx] * number_of_atoms)
        """
        config_descriptors = torch.tensor(self.descriptors[self.indices_descriptors == idx]).float()
        atom_types = torch.tensor(self.atom_types[self.indices_descriptors == idx]).long()
        target = torch.tensor(self.targets[self.indices_targets == idx]).float()
        weights = torch.tensor(self.weights[self.indices_targets == idx]).float()
        target_forces = torch.tensor(self.target_forces[self.indices_target_forces == idx]).float()
        number_of_atoms = torch.tensor(self.natoms_per_config[idx])
        number_of_dgrads = torch.tensor(self.number_of_dgrad_rows[idx])
        assert self.natoms_per_config[idx] == config_descriptors.size(0)
        dgrad = torch.tensor(self.dgrad[self.indices_dgrad == idx]).float()
        dbdrindx = torch.tensor(self.dbdrindx[self.indices_dgrad == idx]).long()
        #print(dbdrindx.size())
        unique_j_indices = torch.tensor(self.unique_j_indices[self.indices_dgrad == idx]).long()
        indices = torch.tensor([idx] * number_of_atoms)
        configuration = {'x': config_descriptors,
                         'y': target, #target.reshape(-1),
                         'y_forces': target_forces,
                         'noa': number_of_atoms.reshape(-1), #number_of_atoms.reshape(-1),
                         't': atom_types,
                         'w': weights,
                         'i': indices,
                         'dgrad': dgrad,
                         'dbdrindx': dbdrindx,
                         'ndgrad': number_of_dgrads.reshape(-1),
                         'unique_j': unique_j_indices}
        return configuration


def torch_collate(batch):
    """
    Collate batch of data, which collates a stack of configurations from Dataset into a batch
    """
    batch_of_descriptors = torch.cat([conf['x'] for conf in batch], dim=0)
    batch_of_types = torch.cat([conf['t'] for conf in batch], dim=0)
    batch_of_targets = torch.cat([conf['y'] for conf in batch], dim=0)
    batch_of_weights = torch.cat([conf['w'] for conf in batch], dim=0)
    batch_of_target_forces = torch.cat([conf['y_forces'] for conf in batch], dim=0)
    number_of_atoms = torch.cat([conf['noa'] for conf in batch], dim=0)
    indices = torch.cat([conf['i'] for conf in batch], dim=0) % len(batch)
    batch_of_dgrad = torch.cat([conf['dgrad'] for conf in batch], dim=0)
    batch_of_dbdrindx = torch.cat([conf['dbdrindx'] for conf in batch], dim=0)
    batch_of_unique_j = torch.cat([conf['unique_j'] for conf in batch], dim=0)
    batch_of_ndgrad = torch.cat([conf['ndgrad'] for conf in batch], dim=0)

    # make a list of indices upon which to contract per-atom energies to calculate config energies
    # this is made by knowing the number of atoms per config in this batch
    # e.g. a batch with 3 configs and 2,4,3 atoms will have indices ordered like:
    # [0,0,1,1,1,1,2,2,2]
    
    config_tag = 0
    l = []
    for natoms in number_of_atoms.tolist():
        for i in range(0,natoms):
            l.append(config_tag)
        config_tag = config_tag+1
    indices = torch.tensor(l).long() 

    # make a list of unique j so that we can contract forces properly on this batch
    # obtain unique atoms j, the central atom in Fj = sum_i{dBi/dBj}, for all atoms in all configs in this batch
    # do this by labeling the atoms in all configs consecutively
    # e.g. a batch with 3 configs and 2,4,3 atoms will have unique_j ordered like:
    # [0,0,0,0,1,1,1,1,2,2,2,2,3,3,3,3,4,4,4,4,5,5,5,5,6,6,6,6,7,7,7,7,8,8,8,8]
    # (assuming all atoms have 4 neighbors, the number of neighbors can be different)
    
    l = batch_of_unique_j.tolist()
    unique_l = set(l)
    seen = []
    tag = 0
    unique_tags = [-1]*len(l)
    #print(unique_tags)
    for i,n in enumerate(l):
        #print(f"{i} {n}")
        if n not in seen:
            tag = tag+1
            seen.append(n)
            unique_tags[i] = tag-1
        else:
            unique_tags[i] = tag-1
    batch_of_unique_j = torch.tensor(unique_tags).long()
    
    # this fixes the neigh_indices bug for batch_size>1:
    # organize the first column (neighbors i in Fj = sum_i{dBi/dRj} ) of dgrad_indices so that the
    # minimum index (0) of each config starts at the number of atoms of the previous config in the batch
    # this helps align the indices later when doing contraction over force components

    # loop over all configs in this batch
    # TODO: need to add onto natoms_indx and ndgrad_indx in a loop, like the hardcoded version below
    natoms_indx = 0
    ndgrad_indx = 0
    nconfigs = number_of_atoms.size()[0]
    for m in range(1,nconfigs): 
        natoms_indx += number_of_atoms[m-1]
        ndgrad_indx += batch_of_ndgrad[m-1]
        batch_of_dbdrindx[ndgrad_indx:ndgrad_indx+batch_of_ndgrad[m],0] += natoms_indx
    
    # hard-coded version of the above loop, this was working for batch size of 4:
    """
    if (number_of_atoms.size()[0] > 1):
        natoms_indx = number_of_atoms[0]
        ndgrad_indx = batch_of_ndgrad[0]
        #print(f"{natoms_indx} {ndgrad_indx}")
        batch_of_dbdrindx[ndgrad_indx:ndgrad_indx+batch_of_ndgrad[1],0] += natoms_indx

    if (number_of_atoms.size()[0] > 2):
        natoms_indx = natoms_indx + number_of_atoms[1]
        ndgrad_indx = ndgrad_indx + batch_of_ndgrad[1]
        #print(f"{natoms_indx} {ndgrad_indx}")
        batch_of_dbdrindx[ndgrad_indx:ndgrad_indx+batch_of_ndgrad[2],0] += natoms_indx

    if (number_of_atoms.size()[0] > 3):
        natoms_indx = natoms_indx + number_of_atoms[2]
        ndgrad_indx = ndgrad_indx + batch_of_ndgrad[2]
        #print(f"{natoms_indx} {ndgrad_indx}")
        batch_of_dbdrindx[ndgrad_indx:ndgrad_indx+batch_of_ndgrad[3],0] += natoms_indx
    """
    # for debugging:
    """
    if ( (number_of_atoms[0] == 4) and (number_of_atoms[1] ==2)):
        torch.set_printoptions(threshold=10_000)
        #print("----- batch of dgrad:")
        #print(batch_of_ndgrad)
        print(number_of_atoms)  
        #print(indices)
        print(batch_of_dbdrindx)
    """

    # asserts to catch bugs
    # max index of neighbors i in the batch must be equal to number of atoms in the batch
    assert (torch.max(batch_of_dbdrindx[:,0]) == (torch.sum(number_of_atoms)-1))
    # max index of neighbors i must equal max index of unique_j
    assert (torch.max(batch_of_dbdrindx[:,0]) == (torch.max(batch_of_unique_j)))

    collated_batch = {'x': batch_of_descriptors,
                      't': batch_of_types,
                      'y': batch_of_targets,
                      'y_forces': batch_of_target_forces,
                      'noa': number_of_atoms,
                      'w': batch_of_weights,
                      'i': indices,
                      'dgrad': batch_of_dgrad,
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
