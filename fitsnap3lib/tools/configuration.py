import numpy as np

class Configuration():
    """
    Class to store training data for each config, allows us to easily collate and process configs in 
    the dataloader.
    """
    def __init__(self, natoms):
        self.natoms = natoms
        self.filename = ""
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