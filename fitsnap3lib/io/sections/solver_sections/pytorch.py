from fitsnap3lib.io.sections.sections import Section

try:
    import torch
    from fitsnap3lib.lib.neural_networks.pytorch import create_torch_network    
    from torch.nn import Parameter


    class PYTORCH(Section):

        def __init__(self, name, config, args):
            super().__init__(name, config, args)
            self.allowedkeys = ['layer_sizes', 'learning_rate', 'num_epochs', 'batch_size', 'save_state_output',
                                'save_freq', 'save_state_input', 'output_file', 'energy_weight', 'force_weight',
                                'training_fraction', 'multi_element_option', 'num_elements', 'manual_seed_flag']
            self._check_section()

            self._check_if_used("SOLVER", "solver", "SVD")

            self.layer_sizes = self.get_value("PYTORCH", "layer_sizes", "num_desc 512 512 1").split()
            self.num_elements = self.get_value("PYTORCH", "num_elements", "1", "int")
            # Section.num_desc is summed over all element types, e.g. twojmax=6 and ntypes=2 gives
            # 60 descriptors, but our per-atom networks should only have 30 descriptors, so here
            # we divide by number of types
            # this therefore requires that all atom types have the same number of descriptors, but
            # can easily be changed later
            if self.layer_sizes[0] == "num_desc":
                assert (Section.num_desc % self.num_elements == 0)
                self.layer_sizes[0] = int(Section.num_desc/self.num_elements)
            self.layer_sizes = [int(layer_size) for layer_size in self.layer_sizes]
            self.learning_rate = self.get_value("PYTORCH", "learning_rate", "1.0E-2", "float")
            self.num_epochs = self.get_value("PYTORCH", "num_epochs", "10", "int")
            self.batch_size = self.get_value("PYTORCH", "batch_size", "4", "int")
            self.save_freq = self.get_value("PYTORCH", "save_freq", "10", "int")
            self.energy_weight = self.get_value("PYTORCH", "energy_weight", "1e-4", "float")
            self.force_weight = self.get_value("PYTORCH", "force_weight", "1.0", "float")
            self.training_fraction = self.get_value("PYTORCH", "training_fraction", "0.8", "float")
            self.multi_element_option = self.get_value("PYTORCH", "multi_element_option", "1", "int")
            self.manual_seed_flag = self.get_value("PYTORCH", "manual_seed_flag", "False", "bool")

            self.save_state_output = self.check_path(self.get_value("PYTORCH", "save_state_output", "FitTorchModel"))
            self.save_state_input = self.check_path(self.get_value("PYTORCH", "save_state_input", None))
            self.output_file = self.check_path(self.get_value("PYTORCH", "output_file", "FitTorch_Pytorch.pt"))

            if (self.manual_seed_flag):
                torch.manual_seed(0)

            self.networks = []
            if (self.multi_element_option==1):
                self.networks.append(create_torch_network(self.layer_sizes))
            elif (self.multi_element_option==2):
                for t in range(self.num_elements):
                    self.networks.append(create_torch_network(self.layer_sizes))
            
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

except RuntimeError:

    class PYTORCH(Section):
        """
        Dummy class for factory to read if MLIAP error is occuring.
        """

        def __init__(self, name, config, args):
            super().__init__(name, config, args)
            raise RuntimeError("MLIAP error.")
