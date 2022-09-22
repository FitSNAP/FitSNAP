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
        self.weights = None
        self.descriptors = None
        self.types = None
        self.forces = None
        self.dgrad = None
        self.dgrad_indices = None