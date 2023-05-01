from fitsnap3lib.solvers.solver import Solver
import numpy as np
from functools import partial
from time import time
import pickle


try:
    from fitsnap3lib.lib.neural_networks.pytorch import create_torch_network, FitTorch
    from fitsnap3lib.lib.neural_networks.jax import jnp, loss, accuracy, mae, jit, grad, adam, apply_updates
    from fitsnap3lib.tools.dataloaders import InRAMDatasetJAX, DataLoader, jax_collate

    class JAX(Solver):

        def __init__(self, name, pt, config):
            super().__init__(name, pt, config, linear=False)
            self.params = self.config.sections["JAX"].params
            self.learning_rate = self.config.sections["JAX"].learning_rate
            self.optimizer = adam(self.learning_rate)
            self.targets = None
            self.indices = []
            self.num_atoms = None
            self.opt_state = None
            self.training_loader = None
            self.training_data = None
            self.truth_mean = None
            self.min_dict = {
                'epoch': 0,
                'params': self.params,
                'rmse': 100,
                'mae': 100
            }

        def inital_setup(self):
            if self.config.sections["JAX"].save_state_input is not None:
                self.load_pytorch()
            else:
                # Standardization step
                inv_std = 1 / np.std(self.pt.shared_arrays['a'].array, axis=0)
                mean_inv_std = jnp.array(np.mean(self.pt.shared_arrays['a'].array, axis=0) * inv_std)
                inv_std = jnp.array(inv_std * np.eye(len(inv_std)))
                self.params[0] = (inv_std, mean_inv_std)
            training = [not elem for elem in self.pt.fitsnap_dict['Testing']]
            self.truth_mean = np.mean(self.pt.shared_arrays['b'].array, axis=0)

            self.training_data = InRAMDatasetJAX(self.pt.shared_arrays['a'].array[training],
                                                 self.pt.shared_arrays['b'].array[training])
            self.training_loader = DataLoader(self.training_data,
                                              batch_size=self.config.sections["JAX"].batch_size,
                                              shuffle=False,
                                              collate_fn=jax_collate,
                                              num_workers=0)
            if self.config.sections["JAX"].opt_state_input is not None:
                with open(self.config.sections["JAX"].opt_state_input, 'rb') as fp:
                    self.opt_state = pickle.load(fp)
            else:
                self.opt_state = self.optimizer.init(self.params)
            self.targets = jnp.array(np.reshape(self.pt.shared_arrays['b'].array[self.pt.shared_arrays['a'].energy_index], (-1, 1)))
            self.num_atoms = jnp.array(np.reshape(self.pt.shared_arrays['number_of_atoms'].array, (-1, 1)))
            for i, ind in enumerate(self.pt.shared_arrays['number_of_atoms'].array):
                temp = [i]*ind
                self.indices.extend(temp)
            self.indices = jnp.array(self.indices)

        @partial(jit, static_argnames=['self', 'num_segments', 'optimizer'])
        def update(self, params, x, y, indices, num_atoms, optimizer, opt_state, num_segments):
            grads = grad(loss)(params, x, y, indices, num_atoms, num_segments)
            updates, opt_state = optimizer.update(grads, opt_state)
            return apply_updates(params, updates), opt_state

        #@pt.rank_zero
        def perform_fit(self):
            @self.pt.rank_zero
            def decorated_perform_fit():
                fit_start = time()
                self.inital_setup()
                for epoch in range(self.config.sections["JAX"].num_epochs):
                    # need to get take sub(A) which is of length num_configs*num_atoms_per_config
                    batch_start = time()
                    for batch in self.training_loader:
                        descriptors = (batch['x'] - self.mean) / self.std
                        # truths = batch['y'] / self.truth_mean
                        # print(descriptors, batch['x'])
                        self.params, self.opt_state = self.update(self.params,
                                                                  descriptors,
                                                                  batch['y'],
                                                                  batch['i'],
                                                                  batch['noa'],
                                                                  self.optimizer,
                                                                  self.opt_state,
                                                                  batch['nseg'])
                    train_acc = accuracy(self.params, (self.pt.shared_arrays['a'].array-self.mean)/self.std, self.targets, self.indices, self.num_atoms)
                    train_mae = mae(self.params, (self.pt.shared_arrays['a'].array-self.mean)/self.std, self.targets, self.indices, self.num_atoms)
                    print(epoch, train_mae, train_acc, time()-batch_start)
                    if train_mae < self.min_dict['mae']:
                        self.min_dict['mae'] = train_mae
                        self.min_dict['epoch'] = epoch
                        self.min_dict['rmse'] = train_acc
                        self.min_dict['params'] = self.params
                print("Min MAE {}, Min RMSE {}, epoch min mae happened {}".format(self.min_dict['mae'],
                                                                                  self.min_dict['rmse'],
                                                                                  self.min_dict['epoch']))
                print("Fitting Time", time()-fit_start)
                self.fit = self.min_dict['params']
                if self.config.sections["JAX"].output_style == 'pytorch':
                    self.create_pytorch()
                if self.config.sections["JAX"].opt_state_output is not None:
                    with open(self.config.sections["JAX"].opt_state_output, 'wb') as fp:
                        pickle.dump(self.opt_state, fp)

            decorated_perform_fit()

        def create_pytorch(self):
            #from ..lib.neural_networks.pytorch import create_torch_network, FitTorch

            pytorch_network = create_torch_network(self.config.sections["JAX"].layer_sizes)
            pytorch_model = FitTorch(pytorch_network, self.config.sections["CALCULATOR"].num_desc)
            weights, bias = map(list, zip(*self.min_dict['params']))
            weights = [np.asarray(w) for w in weights]
            bias = [np.asarray(b) for b in bias]
            pytorch_model.import_wb(weights, bias)
            pytorch_model.write_lammps_torch(self.config.sections["JAX"].output_file)

        def load_pytorch(self):
            #from ..lib.neural_networks.pytorch import create_torch_network, FitTorch

            pytorch_network = create_torch_network(self.config.sections["JAX"].layer_sizes)
            pytorch_model = FitTorch(pytorch_network, self.config.sections["CALCULATOR"].num_desc)
            pytorch_model.load_lammps_torch(self.config.sections["JAX"].save_state_input)
            state_dict = pytorch_model.state_dict()
            weights = []
            bias = []
            for i, key in enumerate(state_dict.keys()):
                if i % 2 == 0:
                    weights.append(jnp.array(state_dict[key].numpy()))
                else:
                    bias.append(jnp.array(state_dict[key].numpy()))
            self.params = list(map(lambda x, y: (x, y), weights, bias))

except ModuleNotFoundError:

    class JAX(Solver):

        def __init__(self, name):
            super().__init__(name)
            raise ModuleNotFoundError("No module named 'JAX'")

except NameError:

    class JAX(Solver):
        """
        Dummy class for factory to read if MLIAP error is occuring.
        """
        def __init__(self, name):
            super().__init__(name)
            raise NameError("MLIAP error.")

except RuntimeError:

    class JAX(Solver):
        """
        Dummy class for factory to read if MLIAP error is occuring.
        """
        def __init__(self, name):
            super().__init__(name)
            raise RuntimeError("MLIAP error.")
