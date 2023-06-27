from fitsnap3lib.io.sections.sections import Section


try:
    from jax import random
    import jax.numpy as jnp


    def random_layer_params(m, n, key, scale=1e-2):
        """
        A helper function to randomly initialize weights and biases
        for a dense neural network layer
        """
        w_key, b_key = random.split(key)
        return scale * random.normal(w_key, (n, m)), scale * random.normal(b_key, (n,))


    class JAX(Section):

        def __init__(self, name, config, pt, infile, args):
            raise Exception("JAX solver not implemented yet.")
            super().__init__(name, config, args)
            self.allowedkeys = ['layer_sizes', 'learning_rate', 'num_epochs', 'batch_size', 'output_style',
                                'output_file', 'save_state_input', 'opt_state_input', 'opt_state_output']
            self._check_section()

            self._check_if_used("SOLVER", "solver", "SVD")

            temp_layer_sizes = self.get_value("JAX", "layer_sizes", "num_desc 512 512 1").split()
            if temp_layer_sizes[0] == "num_desc":
                temp_layer_sizes[0] = Section.num_desc
            self.layer_sizes = [Section.num_desc]
            self.layer_sizes.extend(temp_layer_sizes)
            self.layer_sizes = [int(layer_size) for layer_size in self.layer_sizes]
            self.learning_rate = self.get_value("JAX", "learning_rate", "1.0E-2", "float")
            self.num_epochs = self.get_value("JAX", "num_epochs", "10", "int")
            self.batch_size = self.get_value("JAX", "batch_size", "10", "int")
            self.output_style = self.get_value("JAX", "output_style", "None")
            self.output_file = self.check_path(self.get_value("JAX", "output_file", "FitTorch_JAX.pt"))
            self.save_state_input = self.check_path(self.get_value("JAX", "save_state_input", "None"))
            self.opt_state_input = self.check_path(self.get_value("JAX", "opt_state_input", "None"))
            self.opt_state_output = self.check_path(self.get_value("JAX", "opt_state_output", "None"))
            self.key = random.PRNGKey(0)
            self.keys = random.split(self.key, len(self.layer_sizes))
            # list of tuples [(w_1, b_1), (w_2, b_2), ..., (w_f, b_f)]
            # num of param arrays = num of layer intersections
            # dimensions of weights = (num_output_layer, num_input_layer)
            # dimensions of b = (num_output_layer, )
            self.params = [random_layer_params(m, n, k) for m, n, k in zip(self.layer_sizes[:-1],
                                                                           self.layer_sizes[1:],
                                                                           self.keys)]
            self.delete()

except ModuleNotFoundError:

    class JAX(Section):

        def __init__(self, name, config, args):
            super().__init__(name, config, args)
            raise ModuleNotFoundError("No module named 'JAX'")

except RuntimeError:

    class JAX(Section):

        def __init__(self, name, config, args):
            super().__init__(name, config, args)
            raise RuntimeError("No module named 'JAX'")
