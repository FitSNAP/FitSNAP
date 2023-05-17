from fitsnap3lib.io.sections.sections import Section
try:
    import torch
    from fitsnap3lib.lib.neural_networks.pairwise import create_torch_network    
    from torch.nn import Parameter


    class NETWORK(Section):

        def __init__(self, name, config, pt, infile, args):
            super().__init__(name, config, pt, infile, args)
            self.allowedkeys = ['layer_sizes', 'learning_rate', 'num_epochs', 'batch_size', 'save_state_output',
                                'save_freq', 'save_state_input', 'output_file', 'energy_weight', 'force_weight',
                                'training_fraction', 'multi_element_option', 'num_elements', 'manual_seed_flag']
            self._check_section()

            self._check_if_used("SOLVER", "solver", "SVD")

            self.layer_sizes = self.get_value('NETWORK', "layer_sizes", "num_desc 512 512 1").split()
            self.learning_rate = self.get_value('NETWORK', "learning_rate", "1.0E-2", "float")
            self.num_epochs = self.get_value('NETWORK', "num_epochs", "10", "int")
            self.batch_size = self.get_value('NETWORK', "batch_size", "4", "int")
            self.save_freq = self.get_value('NETWORK', "save_freq", "10", "int")
            self.energy_weight = self.get_value('NETWORK', "energy_weight", "NaN", "float")
            self.force_weight = self.get_value('NETWORK', "force_weight", "NaN", "float")
            self.global_weight_bool = False
            self.training_fraction = self.get_value('NETWORK', "training_fraction", "NaN", "float")
            self.global_fraction_bool = False
            self.multi_element_option = self.get_value('NETWORK', "multi_element_option", "1", "int")
            self.manual_seed_flag = self.get_value('NETWORK', "manual_seed_flag", "False", "bool")
            self.save_state_output = self.check_path(self.get_value('NETWORK', "save_state_output", "FitTorchModel"))
            self.save_state_input = self.check_path(self.get_value('NETWORK', "save_state_input", None))
            self.output_file = self.check_path(self.get_value('NETWORK', "output_file", "FitTorch_Pytorch.pt"))
            self.dtype_setting = self.get_value('NETWORK', "dtype_setting", "1", "int")
            if (self.dtype_setting==1):
                self.dtype = torch.float32
            else:
                self.dtype = torch.float64

            # catch errors associated with settings, and set necessary flags for later

            if (self.energy_weight != self.energy_weight and self.force_weight == self.force_weight):
                raise Exception("Must use global energy weight with global force weight.")
            elif (self.energy_weight == self.energy_weight and self.force_weight != self.force_weight):
                raise Exception("Must use global force weight with global energy weight.")
            elif (self.energy_weight == self.energy_weight and self.force_weight == self.force_weight):
                if (pt._rank==0):
                    print("----- Global weights set: Overriding group weights.")
                self.global_weight_bool = True
            if (self.training_fraction == self.training_fraction):
                if (pt._rank==0):
                    print("----- Global training fraction set: Overriding group fractions.")
                self.global_fraction_bool = True
            if (self.manual_seed_flag):
                if (pt._rank==0):
                    print("----- manual_seed_flag=1: Setting random seed to 0 for debugging.")
                torch.manual_seed(0)
            
            self.delete()
            
except ModuleNotFoundError:

    class NETWORK(Section):
        """
        Dummy class for factory to read if torch is not available for import.
        """
        def __init__(self, name, config, args):
            super().__init__(name, config, args)
            raise ModuleNotFoundError("No module named 'torch'")

except NameError:

    class NETWORK(Section):
        """
        Dummy class for factory to read if MLIAP error is occuring.
        """
        def __init__(self, name, config, args):
            super().__init__(name, config, args)
            raise NameError("MLIAP error.")

except RuntimeError:

    class NETWORK(Section):
        """
        Dummy class for factory to read if MLIAP error is occuring.
        """

        def __init__(self, name, config, args):
            super().__init__(name, config, args)
            raise RuntimeError("MLIAP error.")
