import sys
from fitsnap3lib.solvers.solver import Solver
from time import time
import numpy as np
import psutil


try:
    from fitsnap3lib.lib.neural_networks.pairwise import FitTorch, create_torch_network
    from fitsnap3lib.tools.dataloader.pairwise import InRAMDatasetPyTorch, torch_collate, DataLoader
    from fitsnap3lib.tools.configuration import Configuration
    import torch

    class NETWORK(Solver):
        """
        A class to use custom networks

        Args:
            name: Name of the solver class.

        Attributes:
            optimizer (:obj:`torch.optim.Adam`): Torch Adam optimization object
            model (:obj:`torch.nn.Module`): Network model that maps descriptors to a per atom attribute
            loss_function (:obj:`torch.loss.MSELoss`): Mean squared error loss function
            learning_rate (:obj:`float`): Learning rate for gradient descent
            scheduler (:obj:`torch.optim.lr_scheduler.ReduceLROnPlateau`): Learning rate scheduler
            device: Accelerator device
            training_data (:obj:`torch.utils.data.Dataset`): Torch dataset for training
            training_loader (:obj:`torch.utils.data.DataLoader`): Data loader for loading in datasets
        """

        def __init__(self, name, pt, config):
            super().__init__(name, pt, config, linear=False)

            self.global_weight_bool = self.config.sections['NETWORK'].global_weight_bool
            self.energy_weight = self.config.sections['NETWORK'].energy_weight
            self.force_weight = self.config.sections['NETWORK'].force_weight

            self.global_fraction_bool = self.config.sections['NETWORK'].global_fraction_bool
            self.training_fraction = self.config.sections['NETWORK'].training_fraction

            self.multi_element_option = self.config.sections["NETWORK"].multi_element_option
            if (self.config.sections["CALCULATOR"].calculator == "LAMMPSCUSTOM"):
                self.num_elements = self.config.sections["CUSTOM"].numtypes
                self.num_radial = self.config.sections["CUSTOM"].num_radial
                self.num_3body = self.config.sections["CUSTOM"].num_3body
                self.num_desc_per_element = self.config.sections["CUSTOM"].num_descriptors/self.num_elements
                self.num_desc = self.config.sections["CUSTOM"].num_descriptors
                self.cutoff = self.config.sections["CUSTOM"].cutoff


            self.dtype = self.config.sections["NETWORK"].dtype
            self.layer_sizes = self.config.sections["NETWORK"].layer_sizes
            if self.layer_sizes[0] == "num_desc":
                self.layer_sizes[0] = int(self.num_desc)
            self.layer_sizes = [int(layer_size) for layer_size in self.layer_sizes]

            # create list of networks based on multi-element option

            self.networks = []
            if (self.multi_element_option==1):
                self.networks.append(create_torch_network(self.layer_sizes))
            elif (self.multi_element_option==2):
                for t in range(self.num_elements):
                    self.networks.append(create_torch_network(self.layer_sizes))

            self.optimizer = None
            self.model = FitTorch(self.networks,
                                  self.num_desc,
                                  self.num_radial,
                                  self.num_3body,
                                  self.cutoff,
                                  self.num_elements,
                                  self.multi_element_option,
                                  self.dtype)
            self.loss_function = torch.nn.MSELoss()
            self.learning_rate = self.config.sections["NETWORK"].learning_rate
            if self.config.sections['NETWORK'].save_state_input is not None:
                try:
                    self.model.load_lammps_torch(self.config.sections['NETWORK'].save_state_input)
                except AttributeError:
                    self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
                    save_state_dict = torch.load(self.config.sections['NETWORK'].save_state_input)
                    self.model.load_state_dict(save_state_dict["model_state_dict"])
                    self.optimizer.load_state_dict(save_state_dict["optimizer_state_dict"])
            if self.optimizer is None:
                parameter_list = [{'params': self.model.parameters()}]
                self.optimizer = torch.optim.Adam(parameter_list, lr=self.learning_rate)
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer,
                                                                        mode='min',
                                                                        factor=0.5,
                                                                        patience=49,
                                                                        verbose=True,
                                                                        threshold=0.0001,
                                                                        threshold_mode='abs')

            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            # self.device = "cpu"
            self.pt.single_print("Pytorch device is set to", self.device)
            self.model = self.model.to(self.device)
            self.total_data = None
            self.training_data = None
            self.validation_data = None
            self.training_loader = None

        def weighted_mse_loss(self, prediction, target, weight):
            return torch.mean(weight * (prediction - target)**2)

        def create_datasets(self):
            """
            Creates the dataset to be used for training and the data loader for the batch system.
            """

            # TODO: when only fitting to energy, we don't need all this extra data, and could save 
            # resources by only fitting some configs to forces. 

            self.configs = [Configuration(int(natoms)) for natoms in self.pt.fitsnap_dict['NumAtoms']]

            # add descriptors and atom types

            indx_natoms_low = 0
            indx_forces_low = 0
            indx_neighlist_low = 0
            for i, config in enumerate(self.configs):
                
                indx_natoms_high = indx_natoms_low + config.natoms
                indx_forces_high = indx_forces_low + 3*config.natoms
                nrows_neighlist = int(self.pt.fitsnap_dict["NumNeighs"][i])
                indx_neighlist_high = indx_neighlist_low + nrows_neighlist

                # 'a' contains per-atom quantities

                config.types = self.pt.shared_arrays['a'].array[indx_natoms_low:indx_natoms_high,0] - 1 # start types at zero
                config.numneighs = self.pt.shared_arrays['a'].array[indx_natoms_low:indx_natoms_high,1]
                config.x = self.pt.shared_arrays['a'].array[indx_natoms_low:indx_natoms_high, 2:]

                # 'b' contains per-config quantities

                config.energy = self.pt.shared_arrays['b'].array[i]
                config.weights = self.pt.shared_arrays['w'].array[i]

                # 'c' contains per-atom 3-vector quantities in 1D form

                config.forces = self.pt.shared_arrays['c'].array[indx_forces_low:indx_forces_high]
                config.positions = self.pt.shared_arrays['x'].array[indx_forces_low:indx_forces_high]

                # dictionaries contain per-config quantities

                config.filename = self.pt.fitsnap_dict['Configs'][i]
                config.group = self.pt.fitsnap_dict['Groups'][i]
                config.testing_bool = self.pt.fitsnap_dict['Testing'][i]
                config.numneigh = int(self.pt.fitsnap_dict['NumNeighs'][i])
                
                # other shared arrays contain per-atom per-neighbor quantities or others

                config.neighlist = self.pt.shared_arrays['neighlist'].array[indx_neighlist_low:indx_neighlist_high,0:2]
                config.xneigh = self.pt.shared_arrays['xneigh'].array[indx_neighlist_low:indx_neighlist_high, :]
                config.transform_x = self.pt.shared_arrays['transform_x'].array[indx_neighlist_low:indx_neighlist_high, :]
                assert(np.all(np.round(config.xneigh,6) == np.round(config.transform_x + config.x[config.neighlist[:,1].astype(int),:],6)) )

                indx_natoms_low += config.natoms
                indx_forces_low += 3*config.natoms
                indx_neighlist_low += nrows_neighlist

            # check that we make assignments (not copies) of data, to save memory
            
            assert(np.shares_memory(self.configs[0].neighlist, self.pt.shared_arrays['neighlist'].array))

            # convert to data loader form

            self.total_data = InRAMDatasetPyTorch(self.configs)

            # randomly shuffle and split into training/validation data if using global fractions

            if (self.global_fraction_bool):

                if (self.training_fraction == 0.0):
                    raise Exception("Training fraction must be > 0.0 for now, later we might implement 0.0 training fraction for testing on a test set")
                if ( (self.training_fraction > 1.0) or (self.training_fraction < 0.0) ):
                    raise Exception("Training fraction cannot be > 1.0 or < 0.0")

                self.train_size = int(self.training_fraction * len(self.total_data))
                self.test_size = len(self.total_data) - self.train_size
                self.training_data, self.validation_data = \
                    torch.utils.data.random_split(self.total_data, 
                                                  [self.train_size, self.test_size])

            else: 

                # we are using group training/testing fractions

                training_bool_indices = [not elem for elem in self.pt.fitsnap_dict['Testing']]
                training_indices = [i for i, x in enumerate(training_bool_indices) if x]
                testing_indices = [i for i, x in enumerate(training_bool_indices) if not x]
                self.training_data = torch.utils.data.Subset(self.total_data, training_indices)
                self.validation_data = torch.utils.data.Subset(self.total_data, testing_indices)

            # make training and validation data loaders for batch training
            # TODO: make shuffling=True an option; this shuffles data every epoch, could give more robust fits.

            self.training_loader = DataLoader(self.training_data,
                                              batch_size=self.config.sections["NETWORK"].batch_size,
                                              shuffle=False, #True
                                              collate_fn=torch_collate,
                                              num_workers=0)
            self.validation_loader = DataLoader(self.validation_data,
                                              batch_size=self.config.sections["NETWORK"].batch_size,
                                              shuffle=False, #True
                                              collate_fn=torch_collate,
                                              num_workers=0)

        def perform_fit(self):
            """
            Performs the pytorch fitting for a lammps potential
            """

            assert (self.pt._rank==0)
            self.create_datasets()

            if self.config.sections['NETWORK'].save_state_input is None:

                # standardization
                # need to perform on all network types in the model

                inv_std = 1/np.std(self.pt.shared_arrays['descriptors'].array, axis=0)
                mean_inv_std = np.mean(self.pt.shared_arrays['descriptors'].array, axis=0) * inv_std
                state_dict = self.model.state_dict()

                # look for the first layer for all types of networks, these are keys like
                # network_architecture0.0.weight and network_architecture0.0.bias
                # for the first network, and
                # network_architecture1.0.weight and network_architecture0.1.bias for the next,
                # and so forth

                ntypes = self.num_elements
                num_networks = len(self.networks)
                keys = [*state_dict.keys()]
                for t in range(0,num_networks):
                    first_layer_weight = "network_architecture"+str(t)+".0.weight"
                    first_layer_bias = "network_architecture"+str(t)+".0.bias"
                    state_dict[first_layer_weight] = torch.tensor(inv_std)*torch.eye(len(inv_std))
                    state_dict[first_layer_bias] = torch.tensor(mean_inv_std)

                # load the new state_dict with the standardized weights
                
                self.model.load_state_dict(state_dict)

            train_losses_epochs = []
            val_losses_epochs = []
            # lists for storing training energies and forces
            target_force_plot = []
            model_force_plot = []
            target_energy_plot = []
            model_energy_plot = []
            # lists for storing validation energies and forces
            target_force_plot_val = []
            model_force_plot_val = []
            target_energy_plot_val = []
            model_energy_plot_val = []
            natoms_per_config = [] # stores natoms per config for calculating eV/atom errors later.
            for epoch in range(self.config.sections['NETWORK'].num_epochs):
                print(f"----- epoch: {epoch}")
                start = time()

                # loop over training data

                train_losses_step = []
                loss = None
                self.model.train()
                for i, batch in enumerate(self.training_loader):
                    positions = batch['x'].to(self.device).requires_grad_(True)
                    xneigh = batch['xneigh'].to(self.device)
                    transform_x = batch['transform_x'].to(self.device)
                    atom_types = batch['t'].to(self.device)
                    targets = batch['y'].to(self.device).requires_grad_(True)
                    target_forces = batch['y_forces'].to(self.device).requires_grad_(True)
                    indices = batch['i'].to(self.device)
                    indices_atom = batch['i_atom'].to(self.device)
                    num_atoms = batch['noa'].to(self.device)
                    weights = batch['w'].to(self.device)
                    neighlist = batch['neighlist'].to(self.device)
                    numneigh = batch['numneigh'].to(self.device)
                    unique_i = batch['unique_i'].to(self.device)
                    unique_j = batch['unique_j'].to(self.device)
                    testing_bools = batch['testing_bools']
                    (energies,forces) = self.model(positions, neighlist, transform_x, 
                                                   indices, num_atoms, atom_types, unique_i, unique_j, self.device)
                    energies = torch.div(energies,num_atoms)

                    # ravel the forces for calculating loss

                    forces = forces.ravel()

                    # good assert for verifying that we have proper training configs:
                    #for test_bool in testing_bools:
                    #    assert(test_bool)

                    # make indices showing which config a force belongs to

                    indices_forces = torch.repeat_interleave(indices_atom, 3)
                    force_weights = weights[indices_forces,1]

                    if (self.energy_weight != 0):
                        energies = energies.to(self.device)
                    if (self.force_weight != 0):
                        forces = forces.to(self.device)

                    if (epoch == self.config.sections['NETWORK'].num_epochs-1):

                        if (self.force_weight !=0):
                            target_force_plot.append(target_forces.cpu().detach().numpy())
                            model_force_plot.append(forces.cpu().detach().numpy())
                        if (self.energy_weight !=0):
                            target_energy_plot.append(targets.cpu().detach().numpy())
                            model_energy_plot.append(energies.cpu().detach().numpy())

                    # assert that model and target force dimensions match

                    if (self.force_weight !=0):
                        assert target_forces.size() == forces.size()

                    if (self.energy_weight==0.0):
                        loss = self.force_weight*self.loss_function(forces, target_forces)
                    elif (self.force_weight==0.0):
                        loss = self.energy_weight*self.loss_function(energies, targets)
                    else:
                        if (self.global_weight_bool):
                            loss = self.energy_weight*self.loss_function(energies, targets) \
                                + self.force_weight*self.loss_function(forces, target_forces)
                        else:
                            loss = self.weighted_mse_loss(energies, targets, weights[:,0]) \
                                + self.weighted_mse_loss(forces, target_forces, force_weights)
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                    train_losses_step.append(loss.item())

                # loop over validation data

                val_losses_step = []
                self.model.eval()

                for i, batch in enumerate(self.validation_loader):
                    positions = batch['x'].to(self.device).requires_grad_(True)
                    xneigh = batch['xneigh'].to(self.device)
                    transform_x = batch['transform_x'].to(self.device)
                    atom_types = batch['t'].to(self.device)
                    targets = batch['y'].to(self.device).requires_grad_(True)
                    target_forces = batch['y_forces'].to(self.device).requires_grad_(True)
                    indices = batch['i'].to(self.device)
                    indices_atom = batch['i_atom'].to(self.device)
                    num_atoms = batch['noa'].to(self.device)
                    weights = batch['w'].to(self.device)
                    neighlist = batch['neighlist'].to(self.device)
                    numneigh = batch['numneigh'].to(self.device)
                    unique_i = batch['unique_i'].to(self.device)
                    unique_j = batch['unique_j'].to(self.device)
                    testing_bools = batch['testing_bools']
                    (energies,forces) = self.model(positions, neighlist, transform_x, 
                                                   indices, num_atoms, atom_types, unique_i, unique_j, self.device)
                    energies = torch.div(energies,num_atoms)

                    # ravel the forces for calculating loss

                    forces = forces.ravel()

                    # good assert for verifying that we have proper training configs:
                    #for test_bool in testing_bools:
                    #    assert(test_bool)

                    # make indices showing which config a force belongs to

                    indices_forces = torch.repeat_interleave(indices_atom, 3)
                    force_weights = weights[indices_forces,1]

                    if (self.energy_weight != 0):
                        energies = energies.to(self.device)
                    if (self.force_weight != 0):
                        forces = forces.to(self.device)

                    if (epoch == self.config.sections['NETWORK'].num_epochs-1):

                        if (self.force_weight !=0):
                            target_force_plot_val.append(target_forces.cpu().detach().numpy())
                            model_force_plot_val.append(forces.cpu().detach().numpy())
                            natoms_per_config.append(num_atoms.cpu().detach().numpy())
                        if (self.energy_weight !=0):
                            target_energy_plot_val.append(targets.cpu().detach().numpy())
                            model_energy_plot_val.append(energies.cpu().detach().numpy())

                    # assert that model and target force dimensions match

                    if (self.force_weight !=0):
                        assert target_forces.size() == forces.size()

                    # assert number of atoms is correct
                    
                    if (self.energy_weight !=0):
                        natoms_batch = np.sum(num_atoms.cpu().detach().numpy())
                        nforce_components_batch = forces.size()[0]
                        assert (3*natoms_batch == nforce_components_batch)

                    # calculate loss

                    if (self.energy_weight==0.0):
                        loss = self.force_weight*self.loss_function(forces, target_forces)
                    elif (self.force_weight==0.0):
                        loss = self.energy_weight*self.loss_function(energies, targets)
                    else:
                        if (self.global_weight_bool):
                            loss = self.energy_weight*self.loss_function(energies, targets) \
                                + self.force_weight*self.loss_function(forces, target_forces)
                        else:
                            loss = self.weighted_mse_loss(energies, targets, weights[:,0]) \
                                + self.weighted_mse_loss(forces, target_forces, force_weights)

                    val_losses_step.append(loss.item())

                # average training and validation losses across all batches

                self.pt.single_print("Batch averaged train/val loss:", np.mean(np.asarray(train_losses_step)), np.mean(np.asarray(val_losses_step)))
                train_losses_epochs.append(np.mean(np.asarray(train_losses_step)))
                val_losses_epochs.append(np.mean(np.asarray(val_losses_step)))
                self.pt.single_print("Epoch time", time()-start)
                if epoch % self.config.sections['NETWORK'].save_freq == 0:
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'loss': loss},
                        self.config.sections['NETWORK'].save_state_output
                    )

            if (self.force_weight != 0.0):

                # print target and model forces

                # training
                target_force_plot = np.concatenate(target_force_plot)
                model_force_plot = np.concatenate(model_force_plot)
                target_force_plot = np.array([target_force_plot]).T
                model_force_plot = np.array([model_force_plot]).T
                dat = np.concatenate((model_force_plot, target_force_plot), axis=1)
                np.savetxt("force_comparison.dat", dat)
                # validation
                if (target_force_plot_val):
                    target_force_plot_val = np.concatenate(target_force_plot_val)
                    model_force_plot_val = np.concatenate(model_force_plot_val)
                    target_force_plot_val = np.array([target_force_plot_val]).T
                    model_force_plot_val = np.array([model_force_plot_val]).T
                    dat_val = np.concatenate((model_force_plot_val, target_force_plot_val), axis=1)
                    np.savetxt("force_comparison_val.dat", dat_val) 

            if (self.energy_weight != 0.0):

                # print target and model energies

                # training
                target_energy_plot = np.concatenate(target_energy_plot)
                model_energy_plot = np.concatenate(model_energy_plot)

                target_energy_plot = np.array([target_energy_plot]).T
                model_energy_plot = np.array([model_energy_plot]).T

                dat = np.concatenate((model_energy_plot, target_energy_plot), axis=1)
                np.savetxt("energy_comparison.dat", dat)
                # validation
                if (target_energy_plot_val):
                    target_energy_plot_val = np.concatenate(target_energy_plot_val)
                    model_energy_plot_val = np.concatenate(model_energy_plot_val)
                    natoms_per_config = np.concatenate(natoms_per_config)
                    target_energy_plot_val = np.array([target_energy_plot_val]).T
                    model_energy_plot_val = np.array([model_energy_plot_val]).T
                    natoms_per_config = np.array([natoms_per_config]).T
                    dat_val = np.concatenate((model_energy_plot_val, target_energy_plot_val, natoms_per_config), axis=1)
                    np.savetxt("energy_comparison_val.dat", dat_val)

            # print training loss vs. epoch data

            epochs = np.arange(self.config.sections['NETWORK'].num_epochs)
            epochs = np.array([epochs]).T
            train_losses_epochs = np.array([train_losses_epochs]).T
            val_losses_epochs = np.array([val_losses_epochs]).T
            loss_dat = np.concatenate((epochs,train_losses_epochs,val_losses_epochs),axis=1)
            np.savetxt("loss_vs_epochs.dat", loss_dat)

            self.pt.single_print("Average loss over batches is", np.mean(np.asarray(train_losses_step)))
            
            self.model.write_lammps_torch(self.config.sections['NETWORK'].output_file)
            
            self.fit = None

        def evaluate_configs(self, config_idx = 0, standardize_bool = True, dtype=torch.float64, eval_device='cpu'):
            """
            Evaluates energies and forces on configs for testing purposes. 

            Args:
                config_idx (int): Index of config to evaluate. None if evaluating all configs.
                standardize_bool (bool): True to standardize weights, False otherwise. Useful if 
                    comparing inputs with a previously standardized model.
                dtype (torch.dtype): Optional override of the global dtype.
                eval_device (torch.device): Optional device to evaluate on, defaults to CPU to
                    prevent device mismatch when training on GPU.

            Returns:
                A tuple (energy, force) for the config given by `config_idx`. The tuple will contain 
                lists of energy/force for each config if `config_idx` is None.
            """

            @self.pt.sub_rank_zero
            def decorated_evaluate_configs():

                if (standardize_bool):
                    if self.config.sections['NETWORK'].save_state_input is None:

                        # standardization
                        # need to perform on all network types in the model
                        # TODO for pairwise networks: move this somewhere else, since we don't yet have the descriptors.
                        # TODO perhaps move this to wherever we calculate the descriptors?
                        # TODO we should only have to calculate pairwise descriptors once in order to do this.
                        # TODO only standardize the descriptors, not the derivatives

                        inv_std = 1/np.std(self.pt.shared_arrays['descriptors'].array, axis=0)
                        mean_inv_std = np.mean(self.pt.shared_arrays['descriptors'].array, axis=0) * inv_std
                        state_dict = self.model.state_dict()

                        # look for the first layer for all types of networks, these are keys like
                        # network_architecture0.0.weight and network_architecture0.0.bias
                        # for the first network, and
                        # network_architecture1.0.weight and network_architecture0.1.bias for the next,
                        # and so forth

                        ntypes = self.num_elements
                        num_networks = len(self.networks)
                        keys = [*state_dict.keys()]
                        for t in range(0,num_networks):
                            first_layer_weight = "network_architecture"+str(t)+".0.weight"
                            first_layer_bias = "network_architecture"+str(t)+".0.bias"
                            state_dict[first_layer_weight] = torch.tensor(inv_std)*torch.eye(len(inv_std))
                            state_dict[first_layer_bias] = torch.tensor(mean_inv_std)

                        # load the new state_dict with the standardized weights
                        
                        self.model.load_state_dict(state_dict)

                # only evaluate, no weight gradients

                self.model.eval()

                if (config_idx is not None):
                    # Evaluate a single config.
                    config = self.configs[config_idx]
                      
                    #positions = torch.tensor(config.positions).requires_grad_(True)
                    positions = torch.tensor(config.x).requires_grad_(True)
                    xneigh = torch.tensor(config.xneigh)
                    transform_x = torch.tensor(config.transform_x)
                    atom_types = torch.tensor(config.types).long()
                    target = torch.tensor(config.energy).reshape(-1)
                    # indexing 0th axis with None reshapes the tensor to be 2D for stacking later
                    weights = torch.tensor(config.weights[None,:])
                    target_forces = torch.tensor(config.forces)
                    num_atoms = torch.tensor(config.natoms)
                    neighlist = torch.tensor(config.neighlist).long()

                    # convert quantities to desired dtype
              
                    positions = positions.to(dtype)
                    transform_x = transform_x.to(dtype)
                    target = target.to(dtype)
                    weights = weights.to(dtype)
                    target_forces = target_forces.to(dtype)

                    # make indices upon which to contract per-atom energies for this config

                    config_indices = torch.arange(1).long() # this usually has len(batch) as arg in dataloader
                    indices = torch.repeat_interleave(config_indices, neighlist.size()[0]) # config indices for each pair
                    unique_i = neighlist[:,0]
                    unique_j = neighlist[:,1]
                    
                    # need to unsqueeze num_atoms to get a tensor of definable size

                    (energies,forces) = self.model(positions, neighlist, transform_x, 
                                                  indices, num_atoms.unsqueeze(0), 
                                                  atom_types, unique_i, unique_j, self.device, dtype)
                    
                else:
                    # Evaluate all configs and store energy/force in list.
                    energies = []
                    forces = []
                    for config in self.configs:
                        #positions = torch.tensor(config.positions).requires_grad_(True)
                        positions = torch.tensor(config.x).requires_grad_(True)
                        xneigh = torch.tensor(config.xneigh)
                        transform_x = torch.tensor(config.transform_x)
                        atom_types = torch.tensor(config.types).long()
                        target = torch.tensor(config.energy).reshape(-1)
                        # indexing 0th axis with None reshapes the tensor to be 2D for stacking later
                        weights = torch.tensor(config.weights[None,:])
                        target_forces = torch.tensor(config.forces)
                        num_atoms = torch.tensor(config.natoms)
                        neighlist = torch.tensor(config.neighlist).long()

                        # convert quantities to desired dtype
                  
                        positions = positions.to(dtype)
                        transform_x = transform_x.to(dtype)
                        target = target.to(dtype)
                        weights = weights.to(dtype)
                        target_forces = target_forces.to(dtype)

                        # make indices upon which to contract per-atom energies for this config

                        config_indices = torch.arange(1).long() # this usually has len(batch) as arg in dataloader
                        indices = torch.repeat_interleave(config_indices, neighlist.size()[0]) # config indices for each pair
                        unique_i = neighlist[:,0]
                        unique_j = neighlist[:,1]
                        
                        # need to unsqueeze num_atoms to get a tensor of definable size

                        (e_model,f_model) = self.model(positions, neighlist, transform_x, 
                                                      indices, num_atoms.unsqueeze(0), 
                                                      atom_types, unique_i, unique_j, self.device, dtype)

                        energies.append(e_model)
                        forces.append(f_model)
                    
                return (energies, forces)
                """
                if (option==1):

                    if (evaluate_all):

                        energies_configs = []
                        forces_configs = []
                        for config in self.configs:
                          
                            #positions = torch.tensor(config.positions).requires_grad_(True)
                            positions = torch.tensor(config.x).requires_grad_(True)
                            xneigh = torch.tensor(config.xneigh)
                            transform_x = torch.tensor(config.transform_x)
                            atom_types = torch.tensor(config.types).long()
                            target = torch.tensor(config.energy).reshape(-1)
                            # indexing 0th axis with None reshapes the tensor to be 2D for stacking later
                            weights = torch.tensor(config.weights[None,:])
                            target_forces = torch.tensor(config.forces)
                            num_atoms = torch.tensor(config.natoms)
                            neighlist = torch.tensor(config.neighlist).long()

                            # convert quantities to desired dtype
                      
                            positions = positions.to(dtype)
                            transform_x = transform_x.to(dtype)
                            target = target.to(dtype)
                            weights = weights.to(dtype)
                            target_forces = target_forces.to(dtype)

                            # make indices upon which to contract per-atom energies for this config

                            config_indices = torch.arange(1).long() # this usually has len(batch) as arg in dataloader
                            indices = torch.repeat_interleave(config_indices, neighlist.size()[0]) # config indices for each pair
                            unique_i = neighlist[:,0]
                            unique_j = neighlist[:,1]
                            
                            # need to unsqueeze num_atoms to get a tensor of definable size

                            (energies,forces) = self.model(positions, neighlist, transform_x, 
                                                          indices, num_atoms.unsqueeze(0), 
                                                          atom_types, unique_i, unique_j, self.device, dtype)

                            energies_configs.append(energies)
                            forces_configs.append(forces)

                        return(energies_configs, forces_configs)

                    else:
                        energies_configs = []
                        forces_configs = []
                        config=self.configs[config_index]
                          
                        #positions = torch.tensor(config.positions).requires_grad_(True)
                        positions = torch.tensor(config.x).requires_grad_(True)
                        xneigh = torch.tensor(config.xneigh)
                        transform_x = torch.tensor(config.transform_x)
                        atom_types = torch.tensor(config.types).long()
                        target = torch.tensor(config.energy).reshape(-1)
                        # indexing 0th axis with None reshapes the tensor to be 2D for stacking later
                        weights = torch.tensor(config.weights[None,:])
                        target_forces = torch.tensor(config.forces)
                        num_atoms = torch.tensor(config.natoms)
                        neighlist = torch.tensor(config.neighlist).long()

                        # convert quantities to desired dtype
                  
                        positions = positions.to(dtype)
                        transform_x = transform_x.to(dtype)
                        target = target.to(dtype)
                        weights = weights.to(dtype)
                        target_forces = target_forces.to(dtype)

                        # make indices upon which to contract per-atom energies for this config

                        config_indices = torch.arange(1).long() # this usually has len(batch) as arg in dataloader
                        indices = torch.repeat_interleave(config_indices, neighlist.size()[0]) # config indices for each pair
                        unique_i = neighlist[:,0]
                        unique_j = neighlist[:,1]
                        
                        # need to unsqueeze num_atoms to get a tensor of definable size

                        (energies,forces) = self.model(positions, neighlist, transform_x, 
                                                       indices, num_atoms.unsqueeze(0), 
                                                       atom_types, unique_i, unique_j, self.device, dtype)

                        energies_configs.append(energies)
                        forces_configs.append(forces)

                        return(energies_configs, forces_configs)
                        """

            (energies, forces) = decorated_evaluate_configs()

            return(energies, forces)
            
except ModuleNotFoundError:

    class NETWORK(Solver):
        """
        Dummy class for factory to read if torch is not available for import.
        """
        def __init__(self, name):
            super().__init__(name)
            raise ModuleNotFoundError("No module named 'Pytorch'")

except NameError:

    class NETWORK(Solver):
        """
        Dummy class for factory to read if MLIAP error is occuring.
        """
        def __init__(self, name):
            super().__init__(name)
            raise NameError("MLIAP error.")

except RuntimeError:

    class NETWORK(Solver):
        """
        Dummy class for factory to read if MLIAP error is occuring.
        """
        def __init__(self, name):
            super().__init__(name)
            raise RuntimeError("MLIAP error.")
