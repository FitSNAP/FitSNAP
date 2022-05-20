from ..sections import Section


try:
    import torch
    from ....lib.neural_networks.pytorch import create_torch_network


    class PYTORCH(Section):

        def __init__(self, name, config, args):
            super().__init__(name, config, args)
            self.allowedkeys = ['layer_sizes', 'learning_rate', 'num_epochs', 'batch_size', 'save_state_output',
                                'save_freq', 'save_state_input', 'output_file']
            self._check_section()

            self._check_if_used("SOLVER", "solver", "SVD")

            self.layer_sizes = self.get_value("PYTORCH", "layer_sizes", "num_desc 512 512 1").split()
            if self.layer_sizes[0] == "num_desc":
                self.layer_sizes[0] = Section.num_desc
            self.layer_sizes = [int(layer_size) for layer_size in self.layer_sizes]
            self.learning_rate = self.get_value("PYTORCH", "learning_rate", "1.0E-2", "float")
            self.num_epochs = self.get_value("PYTORCH", "num_epochs", "10", "int")
            self.batch_size = self.get_value("PYTORCH", "batch_size", "10", "int")
            self.save_freq = self.get_value("PYTORCH", "save_freq", "10", "int")
            self.save_state_output = self.check_path(self.get_value("PYTORCH", "save_state_output", "FitTorchModel"))
            self.save_state_input = self.check_path(self.get_value("PYTORCH", "save_state_input", None))
            self.output_file = self.check_path(self.get_value("PYTORCH", "output_file", "FitTorch_Pytorch.pt"))
            self.network_architecture = create_torch_network(self.layer_sizes)
            self.delete()

except ModuleNotFoundError:

    class PYTORCH(Section):
        """
        Dummy class for factory to read if torch is not available for import.
        """
        def __init__(self, name, config, args):
            super().__init__(name, config, args)
            raise ModuleNotFoundError("No module named 'torch'")

except NameError:

    class PYTORCH(Section):
        """
        Dummy class for factory to read if MLIAP error is occuring.
        """
        def __init__(self, name, config, args):
            super().__init__(name, config, args)
            raise NameError("MLIAP error.")
