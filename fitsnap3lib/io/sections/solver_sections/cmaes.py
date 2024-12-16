from fitsnap3lib.io.sections.sections import Section

try:

    #import torch

    class CMAES(Section):

        def __init__(self, name, config, pt, infile, args):
            super().__init__(name, config, pt, infile, args)
            self.allowedkeys = ['population_size', 'sigma']
            self._check_section()

            #self._check_if_used("SOLVER", "solver", "SVD")

            self.population_size = self.get_value("CMAES", "population_size", "36", "int")
            self.sigma = self.get_value("CMAES", "sigma", "0.1", "float")

            self.delete()

            # catch errors associated with settings, and set necessary flags for later

except ModuleNotFoundError:

    class CMAES(Section):
        # Dummy class for factory to read if torch is not available for import.
        def __init__(self, name, config, pt, infile, args):
            super().__init__(name, config, pt, infile, args)
            raise ModuleNotFoundError("No module named 'torch'")

except NameError:

    class CMAES(Section):
        """
        Dummy class for factory to read if MLIAP error is occuring.
        """
        def __init__(self, name, config, pt, infile, args):
            super().__init__(name, config, pt, infile, args)
            raise NameError("MLIAP error.")

except RuntimeError:

    class CMAES(Section):
        """
        Dummy class for factory to read if MLIAP error is occuring.
        """
        def __init__(self, name, config, pt, infile, args):
            super().__init__(name, config, pt, infile, args)
            raise RuntimeError("MLIAP error.")
