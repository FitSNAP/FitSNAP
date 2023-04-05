import numpy as np

class Configuration():
    """
    Class to store training data for each config, allows us to easily collate and process configs in 
    the dataloader.

    Attributes:
        natoms (int): Number of atoms for this config.
        filename (string): Name of file that config was loaded from.
        group (string): Name of group that this config belongs to.
        energy (double): Energy of config.
        testing_bool (bool): True if config is used for testing, not training.
        weights (numpy.array): Array of energy and force weight for this config.
        descriptors (numpy.array): Descriptors for this config, shape (natoms, ndescriptors).
        types (numpy.array): Array of atom type indices.
        forces (numpy.array): Forces for this config with shape (3*natoms). Ordered like f1x, f1y, 
                              f1z, f2x, f2y, f2z, ...
    """
    def __init__(self, natoms):
        self.natoms = natoms
        self.filename = ""
        self.group = ""
        self.energy = None
        self.testing_bool = None
        self.weights = None
        self.descriptors = None
        self.types = None
        self.forces = None

        # dgrad quantities are used for traditional descriptor-based networks

        self.dgrad = None
        self.dgrad_indices = None

        # neighbor list is used for custom networks

        self.neighlist = None
        self.numneigh = None
        self.positions = None # 1D 3N 
        self.x = None # Nx3
        self.xneigh = None # neighbor positions lined up with neighlist[:,1]
        self.transform_x = None # neighlist transformed positions such that xneigh = transform_x + x

        # per-atom scalar quantities

        self.pas = None
