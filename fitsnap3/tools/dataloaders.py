import torch.utils.data
from torch.utils.data import DataLoader
from sys import float_info
import numpy as np


class InRAMDataset(torch.utils.data.Dataset):
    """Load A matrix Dataset from RAM"""

    def __init__(self, a_matrix, b, indices=None):
        """
        Args:
            a_matrix (numpy array): Matrix of descriptors with shape (Features, Descriptors)
            b (numpy array): Array of feature truth values with shape (Features, )
            indices (numpy array): Array of indices that represent which atoms belong to which configs
        """
        self.descriptors = a_matrix
        self.targets = b
        self.indices = indices
        self._length = None
        if self.indices is None:
            self._find_indices()

        assert len(a_matrix) == len(b) == len(self.indices)

    def __len__(self):
        return self._length

    def __getitem__(self, idx):
        pass

    def _find_indices(self):
        """
        This is meant to be a temporary fix to the shortcomings of not using a distributed dataframe.
        Searches through targets and finds non-zeros, which will be the start of a new index.
        If a config ever has an energy of zero, this will not work.
        """
        self.indices = []
        i = -1
        for target in self.targets:
            if -float_info.epsilon > target or target > float_info.epsilon:
                i += 1
            self.indices.append(i)
        self.indices = np.array(self.indices)
        self._length = len(np.unique(self.indices))


class InRAMDatasetPyTorch(InRAMDataset):
    """Load A matrix Dataset from RAM"""

    def __getitem__(self, idx):
        config_descriptors = torch.tensor(self.descriptors[self.indices == idx]).float()
        target = torch.tensor(np.sum(self.targets[self.indices == idx])).float()
        number_of_atoms = torch.tensor(config_descriptors.size(0)).int()
        indices = torch.tensor([idx] * number_of_atoms)
        configuration = {'x': config_descriptors,
                         'y': target.reshape(-1),
                         'noa': number_of_atoms.reshape(-1),
                         'i': indices}
        return configuration


def torch_collate(batch):
    """
    Collate batch of data, which collates a stack of configurations from Dataset into a batch
    """
    batch_of_descriptors = torch.cat([conf['x'] for conf in batch], dim=0)
    batch_of_targets = torch.cat([conf['y'] for conf in batch], dim=0)
    number_of_atoms = torch.cat([conf['noa'] for conf in batch], dim=0)
    indices = torch.cat([conf['i'] for conf in batch], dim=0) % len(batch)

    collated_batch = {'x': batch_of_descriptors, 'y': batch_of_targets, 'noa': number_of_atoms, 'i': indices}

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

