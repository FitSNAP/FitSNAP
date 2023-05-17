
import sys
from fitsnap3lib.solvers.solver import Solver
from time import time
import numpy as np
import psutil

try:
    from fitsnap3lib.lib.neural_networks.pytorch import FitTorch, create_torch_network
    from fitsnap3lib.lib.neural_networks.pas import FitTorchPAS
    from fitsnap3lib.tools.dataloaders import InRAMDatasetPyTorch, torch_collate, DataLoader
    from fitsnap3lib.tools.configuration import Configuration
    import torch

    class PYTORCH(Solver):
        """
        A class to wrap Modules to ensure lammps mliap compatability.

        Args:
            name: Name of the solver class.

        Attributes:
            optimizer (torch.optim.Adam): PyTorch Adam optimization object.
            model (torch.nn.Module): Network model that maps descriptors to a per atom quantity 
                (e.g. energy).
            loss_function (torch.loss.MSELoss): Mean squared error loss function.
            learning_Rate (float): Learning rate for gradient descent.
            scheduler (torch.optim.lr_scheduler.ReduceLROnPlateau): Torch learning rate scheduler.
            device: Torch accelerator device.
            training_data (torch.utils.data.Dataset): Torch dataset for training.
            training_loader (torch.utils.data.DataLoader): Data loader for loading datasets.
        """

        def __init__(self, name, pt, config):
            super().__init__(name, pt, config, linear=False)

            self.global_weight_bool = self.config.sections['PYTORCH'].global_weight_bool
            self.energy_weight = self.config.sections['PYTORCH'].energy_weight
            self.force_weight = self.config.sections['PYTORCH'].force_weight

            self.global_fraction_bool = self.config.sections['PYTORCH'].global_fraction_bool
            self.training_fraction = self.config.sections['PYTORCH'].training_fraction

            self.multi_element_option = self.config.sections["PYTORCH"].multi_element_option
            if (self.config.sections["CALCULATOR"].calculator == "LAMMPSSNAP"):
                self.num_elements = self.config.sections["BISPECTRUM"].numtypes
            elif (self.config.sections["CALCULATOR"].calculator == "LAMMPSPACE"):
                self.num_elements = self.config.sections["ACE"].numtypes
            else:
                raise Exception("Unsupported calculator for PyTorch solver.")

            self.num_desc_per_element = self.config.sections["CALCULATOR"].num_desc/self.num_elements

            self.dtype = self.config.sections["PYTORCH"].dtype
            self.layer_sizes = self.config.sections["PYTORCH"].layer_sizes
            if self.layer_sizes[0] == "num_desc":
                #assert (Section.num_desc % self.num_elements == 0)
                self.layer_sizes[0] = int(self.num_desc_per_element)
            self.layer_sizes = [int(layer_size) for layer_size in self.layer_sizes]

            # create list of networks based on multi-element option

            self.networks = []
            if (self.multi_element_option==1):
                self.networks.append(create_torch_network(self.layer_sizes))
            elif (self.multi_element_option==2):
                for t in range(self.num_elements):
                    self.networks.append(create_torch_network(self.layer_sizes))

            self.optimizer = None
            self.force_bool = self.pt.fitsnap_dict['force']
            self.per_atom_scalar_bool = self.pt.fitsnap_dict['per_atom_scalar']
            if (self.pt.fitsnap_dict['energy'] or self.pt.fitsnap_dict['force']):
                self.model = FitTorch(self.networks, #config.sections["PYTORCH"].networks,
                                      self.num_desc_per_element,
                                      self.force_bool,
                                      self.num_elements,
                                      self.multi_element_option,
                                      self.dtype)
            elif (self.pt.fitsnap_dict['per_atom_scalar']):
                self.model = FitTorchPAS(self.networks,
                                         self.num_desc_per_element,
                                         self.num_elements,
                                         self.multi_element_option,
                                         self.dtype)            
            self.loss_function = torch.nn.MSELoss()
            self.learning_rate = self.config.sections["PYTORCH"].learning_rate
            if self.config.sections['PYTORCH'].save_state_input is not None:
                try:
                    self.model.load_lammps_torch(self.config.sections['PYTORCH'].save_state_input)
                except AttributeError:
                    self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
                    save_state_dict = torch.load(self.config.sections['PYTORCH'].save_state_input)
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
            if (self.config.args.verbose):
                self.pt.single_print("Pytorch device is set to", self.device)
            self.model = self.model.to(self.device)
            self.total_data = None
            self.training_data = None
            self.validation_data = None
            self.training_loader = None

        def weighted_mse_loss(self, prediction, target, weight):
            return torch.mean(weight * (prediction - target)**2)

        def create_datasets(self, configs=None, pt=None):
            """
            Creates the dataset to be used for training and the data loader for the batch system.

            Args:
                configs: Optional list of Configuration objects. If not supplied, we generate a 
                         configs list using the shared arrays.
                pt: Optional ParallelTools instance containing data we want to fit to.
            """

            # TODO: when only fitting to energy, we don't need all this extra data, and could save 
            # resources by only fitting some configs to forces.

            #print(f">>> rank {pt._rank} dict {pt.fitsnap_dict}")
            if pt is None:
                pt = self.pt

            if configs is None:
                self.configs = [Configuration(int(natoms)) for natoms in pt.fitsnap_dict['NumAtoms']]

                # add descriptors and atom types

                indx_natoms_low = 0
                indx_forces_low = 0
                indx_dgrad_low = 0
                for i, config in enumerate(self.configs):
                    
                    indx_natoms_high = indx_natoms_low + config.natoms
                    indx_forces_high = indx_forces_low + 3*config.natoms

                    if (pt.fitsnap_dict['force']):
                        nrows_dgrad = int(pt.fitsnap_dict["NumDgradRows"][i])
                        indx_dgrad_high = indx_dgrad_low + nrows_dgrad
                        config.forces = pt.shared_arrays['c'].array[indx_forces_low:indx_forces_high]
                        config.dgrad = pt.shared_arrays['dgrad'].array[indx_dgrad_low:indx_dgrad_high]
                        config.dgrad_indices = pt.shared_arrays['dbdrindx'].array[indx_dgrad_low:indx_dgrad_high]
                        indx_dgrad_low += nrows_dgrad

                    if (pt.fitsnap_dict['energy']):
                        config.energy = pt.shared_arrays['b'].array[i]
                        config.weights = pt.shared_arrays['w'].array[i]

                    if (pt.fitsnap_dict['per_atom_scalar']):
                        config.pas = pt.shared_arrays['pas'].array[indx_natoms_low:indx_natoms_high]

                    config.filename = pt.fitsnap_dict['Configs'][i]
                    config.group = pt.fitsnap_dict['Groups'][i]
                    config.testing_bool = pt.fitsnap_dict['Testing'][i]
                    config.descriptors = pt.shared_arrays['a'].array[indx_natoms_low:indx_natoms_high]
                    config.types = pt.shared_arrays['t'].array[indx_natoms_low:indx_natoms_high] - 1 # start types at zero

                    indx_natoms_low += config.natoms
                    indx_forces_low += 3*config.natoms

                # check that we make assignments (not copies) of data, to save memory

                assert(np.shares_memory(self.configs[0].descriptors, pt.shared_arrays['a'].array))

            else:
                self.configs = configs

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

                #blah = [not config.testing_bool for config in self.configs]

                #training_bool_indices = [not elem for elem in self.pt.fitsnap_dict['Testing']]
                training_bool_indices = [not config.testing_bool for config in self.configs]
                training_indices = [i for i, x in enumerate(training_bool_indices) if x]
                testing_indices = [i for i, x in enumerate(training_bool_indices) if not x]
                self.training_data = torch.utils.data.Subset(self.total_data, training_indices)
                self.validation_data = torch.utils.data.Subset(self.total_data, testing_indices)

            # make training and validation data loaders for batch training

            self.training_loader = DataLoader(self.training_data,
                                              batch_size=self.config.sections["PYTORCH"].batch_size,
                                              shuffle=self.config.sections['PYTORCH'].shuffle_flag,
                                              collate_fn=torch_collate,
                                              num_workers=0)

            self.validation_loader = DataLoader(self.validation_data,
                                              batch_size=self.config.sections["PYTORCH"].batch_size,
                                              shuffle=self.config.sections['PYTORCH'].shuffle_flag,
                                              collate_fn=torch_collate,
                                              num_workers=0) if len(self.validation_data) > 1 else []
        #@pt.sub_rank_zero
        def perform_fit(self, configs: list=None, pt=None, outfile: str=None, verbose: bool=True):
            """
            Performs PyTorch fitting using previously calculated descriptors. 

            Args:
                configs: Optional list of Configuration objects to perform fitting on.
                pt: ParallelTools instance containing shared arrays and data we want 
                    to fit to.
                outfile: Optional output file to write progress to.
                verbose: Optional flag to print progress to screen; overrides the verbose CLI.
            """

            @self.pt.sub_rank_zero
            def perform_fit(pt=None,outfile=None):
                pt = self.pt if pt is None else pt
                if outfile is not None:
                    fh = open(outfile, 'w')
                
                #self.create_datasets()
                if self.config.sections['PYTORCH'].save_state_input is None:

                    # standardization
                    # need to perform on all network types in the model

                    inv_std = 1/np.std(pt.shared_arrays['a'].array, axis=0)
                    mean_inv_std = np.mean(pt.shared_arrays['a'].array, axis=0) * inv_std
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
                target_pas_plot = []
                model_pas_plot = []
                # lists for storing validation energies and forces
                target_force_plot_val = []
                model_force_plot_val = []
                target_energy_plot_val = []
                model_energy_plot_val = []
                target_pas_plot_val = []
                model_pas_plot_val = []
                natoms_per_config = [] # stores natoms per config for calculating eV/atom errors later.
                if (self.config.args.verbose or verbose):
                    self.pt.single_print(f"{'Epoch': <2} {'Train': ^10} {'Val': ^10} {'Time (s)': >2}")
                for epoch in range(self.config.sections["PYTORCH"].num_epochs):
                    start = time()

                    # loop over training data

                    train_losses_step = []
                    loss = None
                    self.model.train()
                    for i, batch in enumerate(self.training_loader):
                        descriptors = batch['x'].to(self.device).requires_grad_(True)
                        atom_types = batch['t'].to(self.device)
                        indices = batch['i'].to(self.device)
                        num_atoms = batch['noa'].to(self.device)
                        targets = batch['y'].to(self.device).requires_grad_(True)

                        if (self.pt.fitsnap_dict['energy']):
                            #targets = batch['y'].to(self.device).requires_grad_(True)
                            weights = batch['w'].to(self.device)
                        else:
                            #targets = None
                            weights = None

                        if (self.pt.fitsnap_dict['force']):
                            target_forces = batch['y_forces'].to(self.device).requires_grad_(True)
                            dgrad = batch['dgrad'].to(self.device).requires_grad_(True)
                            dbdrindx = batch['dbdrindx'].to(self.device)
                            unique_j = batch['unique_j'].to(self.device)
                            unique_i = batch['unique_i'].to(self.device)
                            # make indices showing which config a force belongs to
                            indices_forces = torch.repeat_interleave(indices, 3)
                            # get force weights
                            force_weights = weights[indices_forces,1]
                        else:
                            target_forces = None
                            dgrad = None
                            dbdrindx = None
                            unique_j = None
                            unique_i = None

                        if (self.pt.fitsnap_dict['energy'] or (self.pt.fitsnap_dict['force'])):
                            # we are fitting energies/forces
                            (energies,forces) = self.model(descriptors, dgrad, indices, num_atoms, 
                                                           atom_types, dbdrindx, unique_j, unique_i, 
                                                           self.device)
                            energies = torch.div(energies,num_atoms)
                        else:
                            # calculate per-atom scalars
                            pas = self.model(descriptors, num_atoms, atom_types, self.device)
                            energies = None
                            forces = None

                        # good assert for verifying that we have proper training configs:
                        # testing_bools = batch['testing_bools']
                        #for test_bool in testing_bools:
                        #    assert(test_bool)

                        if (self.pt.fitsnap_dict['energy']):
                            energies = energies.to(self.device)
                        if (self.pt.fitsnap_dict['force']):
                            forces = forces.to(self.device)
                        if (self.pt.fitsnap_dict['per_atom_scalar']):
                            pas = pas.to(self.device)

                        if (epoch == self.config.sections["PYTORCH"].num_epochs-1):

                            if (self.pt.fitsnap_dict['force']):
                                target_force_plot.append(target_forces.cpu().detach().numpy())
                                model_force_plot.append(forces.cpu().detach().numpy())
                            if (self.pt.fitsnap_dict['energy']):
                                target_energy_plot.append(targets.cpu().detach().numpy())
                                model_energy_plot.append(energies.cpu().detach().numpy())
                            if (self.pt.fitsnap_dict['per_atom_scalar']):
                                target_pas_plot.append(targets.cpu().detach().numpy())
                                model_pas_plot.append(pas.cpu().detach().numpy())

                        # assert that model and target force dimensions match

                        if (self.pt.fitsnap_dict['force']):
                            assert target_forces.size() == forces.size()

                        # calculate loss

                        if (self.pt.fitsnap_dict['energy'] and self.pt.fitsnap_dict['force']):
                            if (self.global_weight_bool):
                                loss = self.energy_weight*self.loss_function(energies, targets) \
                                     + self.force_weight*self.loss_function(forces, target_forces)
                            else:
                                loss = self.weighted_mse_loss(energies, targets, weights[:,0]) \
                                    + self.weighted_mse_loss(forces, target_forces, force_weights)
                        elif (self.pt.fitsnap_dict['energy']):
                            if (self.global_weight_bool):
                                loss = self.energy_weight*self.loss_function(energies, targets)
                            else:
                                loss = self.weighted_mse_loss(energies, targets, weights[:,0])
                        elif (self.pt.fitsnap_dict['per_atom_scalar']):
                            # currently just using MSE on entire set, no weights
                            # TODO: add groups weights
                            #loss = torch.sqrt(self.loss_function(pas, targets))
                            loss = self.loss_function(pas, targets)

                        self.optimizer.zero_grad()
                        loss.backward()
                        self.optimizer.step()
                        train_losses_step.append(loss.item())

                    # loop over validation data

                    val_losses_step = []
                    self.model.eval()
                    for i, batch in enumerate(self.validation_loader):
                        descriptors = batch['x'].to(self.device).requires_grad_(True)
                        atom_types = batch['t'].to(self.device)
                        indices = batch['i'].to(self.device)
                        num_atoms = batch['noa'].to(self.device)
                        targets = batch['y'].to(self.device).requires_grad_(True)

                        if (self.pt.fitsnap_dict['energy']):
                            #targets = batch['y'].to(self.device).requires_grad_(True)
                            weights = batch['w'].to(self.device)
                        else:
                            #targets = None
                            weights = None

                        if (self.pt.fitsnap_dict['force']):
                            target_forces = batch['y_forces'].to(self.device).requires_grad_(True)
                            dgrad = batch['dgrad'].to(self.device).requires_grad_(True)
                            dbdrindx = batch['dbdrindx'].to(self.device)
                            unique_j = batch['unique_j'].to(self.device)
                            unique_i = batch['unique_i'].to(self.device)
                            # make indices showing which config a force belongs to
                            indices_forces = torch.repeat_interleave(indices, 3)
                            # get force weights
                            force_weights = weights[indices_forces,1]
                        else:
                            target_forces = None
                            dgrad = None
                            dbdrindx = None
                            unique_j = None
                            unique_i = None

                        if (self.pt.fitsnap_dict['energy'] or (self.pt.fitsnap_dict['force'])):
                            # we are fitting energies/forces
                            (energies,forces) = self.model(descriptors, dgrad, indices, num_atoms, 
                                                           atom_types, dbdrindx, unique_j, unique_i, 
                                                           self.device)
                            energies = torch.div(energies,num_atoms)
                        else:
                            # calculate per-atom scalars
                            pas = self.model(descriptors, num_atoms, atom_types, self.device)
                            energies = None
                            forces = None

                        if (self.pt.fitsnap_dict['energy']):
                            energies = energies.to(self.device)
                        if (self.pt.fitsnap_dict['force']):
                            forces = forces.to(self.device)
                        if (self.pt.fitsnap_dict['per_atom_scalar']):
                            pas = pas.to(self.device)

                        if (epoch == self.config.sections["PYTORCH"].num_epochs-1):

                            if (self.pt.fitsnap_dict['force']):
                                target_force_plot_val.append(target_forces.cpu().detach().numpy())
                                model_force_plot_val.append(forces.cpu().detach().numpy())
                                #natoms_per_config.append(num_atoms.cpu().detach().numpy())
                            if (self.pt.fitsnap_dict['energy']):
                                target_energy_plot_val.append(targets.cpu().detach().numpy())
                                model_energy_plot_val.append(energies.cpu().detach().numpy())
                            if (self.pt.fitsnap_dict['per_atom_scalar']):
                                target_pas_plot_val.append(targets.cpu().detach().numpy())
                                model_pas_plot_val.append(pas.cpu().detach().numpy())

                        # assert that model and target force dimensions match

                        if (self.pt.fitsnap_dict['force']):
                            assert target_forces.size() == forces.size()

                        # assert number of atoms is correct
                        """
                        if (self.energy_weight !=0):
                            natoms_batch = np.sum(num_atoms.cpu().detach().numpy())
                            nforce_components_batch = forces.size()[0]
                            assert (3*natoms_batch == nforce_components_batch)
                        """

                        # calculate loss

                        if (self.pt.fitsnap_dict['energy'] and self.pt.fitsnap_dict['force']):
                            if (self.global_weight_bool):
                                loss = self.energy_weight*self.loss_function(energies, targets) \
                                     + self.force_weight*self.loss_function(forces, target_forces)
                            else:
                                loss = self.weighted_mse_loss(energies, targets, weights[:,0]) \
                                    + self.weighted_mse_loss(forces, target_forces, force_weights)
                        elif (self.pt.fitsnap_dict['energy']):
                            if (self.global_weight_bool):
                                loss = self.energy_weight*self.loss_function(energies, targets)
                            else:
                                loss = self.weighted_mse_loss(energies, targets, weights[:,0])
                        elif (self.pt.fitsnap_dict['per_atom_scalar']):
                            # currently just using MSE on entire set, no weights
                            # TODO: add groups weights
                            #loss = torch.sqrt(self.loss_function(pas, targets))
                            loss = self.loss_function(pas, targets)
                        
                        val_losses_step.append(loss.item())

                    # average training and validation losses across all batches

                    progress_str = f"{epoch: <2} {np.mean(np.asarray(train_losses_step)): ^10.3e} {np.mean(np.asarray(val_losses_step)): ^10.3e} {time()-start: >2.3e}"
                    #self.pt.single_print(progress_str)
                    if (self.config.args.verbose or verbose):
                        self.pt.single_print(progress_str)
                    if outfile is not None:
                        fh.write(progress_str + "\n")
                    train_losses_epochs.append(np.mean(np.asarray(train_losses_step)))
                    val_losses_epochs.append(np.mean(np.asarray(val_losses_step)))
                    if epoch % self.config.sections['PYTORCH'].save_freq == 0:
                        torch.save({
                            'epoch': epoch,
                            'model_state_dict': self.model.state_dict(),
                            'optimizer_state_dict': self.optimizer.state_dict(),
                            'loss': loss},
                            self.config.sections['PYTORCH'].save_state_output
                        )

                # TODO: Remove the following commented block after long-term use confirms that new
                #       detailed errors make this redundant. 
                """
                if (self.pt.fitsnap_dict['force']):

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

                if (self.pt.fitsnap_dict['energy']):

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
                        #natoms_per_config = np.concatenate(natoms_per_config)
                        target_energy_plot_val = np.array([target_energy_plot_val]).T
                        model_energy_plot_val = np.array([model_energy_plot_val]).T
                        #natoms_per_config = np.array([natoms_per_config]).T
                        dat_val = np.concatenate((model_energy_plot_val, target_energy_plot_val), axis=1)
                        np.savetxt("energy_comparison_val.dat", dat_val)
                """

                if (self.pt.fitsnap_dict['per_atom_scalar']):

                    # print target and model energies
                    # training
                    target_pas_plot = np.concatenate(target_pas_plot)
                    model_pas_plot = np.concatenate(model_pas_plot)

                    target_pas_plot = np.array([target_pas_plot]).T
                    model_pas_plot = np.array([model_pas_plot]).T

                    dat = np.concatenate((model_pas_plot, target_pas_plot), axis=1)
                    np.savetxt("pas_comparison.dat", dat)
                    # validation
                    if (target_pas_plot_val):
                        target_pas_plot_val = np.concatenate(target_pas_plot_val)
                        model_pas_plot_val = np.concatenate(model_pas_plot_val)
                        #natoms_per_config = np.concatenate(natoms_per_config)
                        target_pas_plot_val = np.array([target_pas_plot_val]).T
                        model_pas_plot_val = np.array([model_pas_plot_val]).T
                        #natoms_per_config = np.array([natoms_per_config]).T
                        dat_val = np.concatenate((model_pas_plot_val, target_pas_plot_val), axis=1)
                        np.savetxt("pas_comparison_val.dat", dat_val)

                # print training loss vs. epoch data

                epochs = np.arange(self.config.sections["PYTORCH"].num_epochs)
                epochs = np.array([epochs]).T
                train_losses_epochs = np.array([train_losses_epochs]).T
                val_losses_epochs = np.array([val_losses_epochs]).T
                loss_dat = np.concatenate((epochs,train_losses_epochs,val_losses_epochs),axis=1)
                np.savetxt("loss_vs_epochs.dat", loss_dat)
                
                #if 'lammps.mliap' in sys.modules:
                self.model.write_lammps_torch(self.config.sections["PYTORCH"].output_file)
                #else:
                #    print("Warning: This interpreter is not compatible with python-based mliap for LAMMPS. If you are using a Mac please make sure you have compiled python from source with './configure --enabled-shared' ")
                #    print("Warning: FitSNAP will continue without ML-IAP")
                
                self.fit = None

            self.create_datasets(configs=configs, pt=pt)
            perform_fit(pt=pt, outfile=outfile)

        #@pt.sub_rank_zero
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
                #self.create_datasets()
                # Convert model to dtype
                self.model.to(dtype)
                # Send model to device
                self.model.to(eval_device)

                if (standardize_bool):
                    if self.config.sections['PYTORCH'].save_state_input is None:

                        # standardization
                        # need to perform on all network types in the model

                        inv_std = 1/np.std(self.pt.shared_arrays['a'].array, axis=0)
                        mean_inv_std = np.mean(self.pt.shared_arrays['a'].array, axis=0) * inv_std
                        state_dict = self.model.state_dict()

                        # look for the first layer for all types of networks, these are keys like
                        # network_architecture0.0.weight and network_architecture0.0.bias
                        # for the first network, and
                        # network_architecture1.0.weight and network_architecture0.1.bias for the next,
                        # and so forth

                        ntypes = self.num_elements
                        num_networks = len(self.networks)
                        keys = [*state_dict.keys()]
                        #for t in range(0,ntypes):
                        for t in range(0,num_networks):
                            first_layer_weight = "network_architecture"+str(t)+".0.weight"
                            first_layer_bias = "network_architecture"+str(t)+".0.bias"
                            state_dict[first_layer_weight] = torch.tensor(inv_std)*torch.eye(len(inv_std))
                            state_dict[first_layer_bias] = torch.tensor(mean_inv_std)

                        # load the new state_dict with the standardized weights
                        
                        self.model.load_state_dict(state_dict)

                # only evaluate, no weight gradients

                self.model.eval()

                # Evaluate a single config:

                if (config_idx is not None):
                    config = self.configs[config_idx]
                    descriptors = torch.tensor(config.descriptors).requires_grad_(True)
                    atom_types = torch.tensor(config.types).long()
                    target = torch.tensor(config.energy).reshape(-1)
                    # indexing 0th axis with None reshapes the tensor to be 2D for stacking later
                    weights = torch.tensor(config.weights[None,:])
                    num_atoms = torch.tensor(config.natoms)
                    """
                    target_forces = torch.tensor(config.forces)
                    num_atoms = torch.tensor(config.natoms)
                    dgrad = torch.tensor(config.dgrad)
                    dbdrindx = torch.tensor(config.dgrad_indices).long()
                    """
                    if (self.pt.fitsnap_dict['force']):
                        target_forces = torch.tensor(config.forces)
                        dgrad = torch.tensor(config.dgrad)
                        dbdrindx = torch.tensor(config.dgrad_indices).long()

                        # illustrate what unique_i and unique_j are

                        unique_i = dbdrindx[:,0]
                        unique_j = dbdrindx[:,1]

                        # convert quantities to desired type

                        target_forces = target_forces.to(dtype)
                        dgrad = dgrad.to(dtype)
                    else:
                        target_forces = None
                        dgrad = None
                        dbdrindx = None
                        unique_j = None
                        unique_i = None

                    # convert quantities to desired dtype
              
                    descriptors = descriptors.to(dtype)
                    target = target.to(dtype)
                    weights = weights.to(dtype)
                    #target_forces = target_forces.to(dtype)
                    #dgrad = dgrad.to(dtype)

                    # make indices upon which to contract per-atom energies for this config

                    config_indices = torch.arange(1).long() # this usually has len(batch) as arg in dataloader
                    indices = torch.repeat_interleave(config_indices, num_atoms)
                    
                    # illustrate what unique_i and unique_j are

                    #unique_i = dbdrindx[:,0]
                    #unique_j = dbdrindx[:,1]

                    (energies,forces) = self.model(descriptors, dgrad, indices, num_atoms, 
                                                  atom_types, dbdrindx, unique_j, unique_i, 
                                                  eval_device, dtype)

                else:
                    
                    # Evaluate all configs.

                    energies = []
                    forces = []
                    for config in self.configs:
                      
                        descriptors = torch.tensor(config.descriptors).requires_grad_(True)
                        atom_types = torch.tensor(config.types).long()
                        target = torch.tensor(config.energy).reshape(-1)
                        # indexing 0th axis with None reshapes the tensor to be 2D for stacking later
                        weights = torch.tensor(config.weights[None,:])
                        num_atoms = torch.tensor(config.natoms)
                        """
                        target_forces = torch.tensor(config.forces)
                        num_atoms = torch.tensor(config.natoms)
                        dgrad = torch.tensor(config.dgrad)
                        dbdrindx = torch.tensor(config.dgrad_indices).long()
                        """
                        if (self.pt.fitsnap_dict['force']):
                            target_forces = torch.tensor(config.forces)
                            dgrad = torch.tensor(config.dgrad)
                            dbdrindx = torch.tensor(config.dgrad_indices).long()

                            # illustrate what unique_i and unique_j are

                            unique_i = dbdrindx[:,0]
                            unique_j = dbdrindx[:,1]

                            # convert quantities to desired type

                            target_forces = target_forces.to(dtype)
                            dgrad = dgrad.to(dtype)
                        else:
                            target_forces = None
                            dgrad = None
                            dbdrindx = None
                            unique_j = None
                            unique_i = None

                        # convert quantities to desired dtype
                  
                        descriptors = descriptors.to(dtype)
                        target = target.to(dtype)
                        weights = weights.to(dtype)
                        #target_forces = target_forces.to(dtype)
                        #dgrad = dgrad.to(dtype)

                        # make indices upon which to contract per-atom energies for this config

                        config_indices = torch.arange(1).long() # this usually has len(batch) as arg in dataloader
                        indices = torch.repeat_interleave(config_indices, num_atoms)

                        # illustrate what unique_j and unique_i are

                        #unique_i = dbdrindx[:,0]
                        #unique_j = dbdrindx[:,1]

                        (e_model,f_model) = self.model(descriptors, dgrad, indices, num_atoms, 
                                                      atom_types, dbdrindx, unique_j, unique_i, 
                                                      eval_device, dtype)
                        energies.append(e_model)
                        forces.append(f_model)

                return(energies, forces)

            (energies,forces) = decorated_evaluate_configs()

            return(energies,forces)


except ModuleNotFoundError:

    class PYTORCH(Solver):
        """
        Dummy class for factory to read if torch is not available for import.
        """
        def __init__(self, name):
            super().__init__(name)
            raise ModuleNotFoundError("No module named 'Pytorch'")

except NameError:

    class PYTORCH(Solver):
        """
        Dummy class for factory to read if MLIAP error is occuring.
        """
        def __init__(self, name):
            super().__init__(name)
            raise NameError("MLIAP error.")

except RuntimeError:

    class PYTORCH(Solver):
        """
        Dummy class for factory to read if MLIAP error is occuring.
        """
        def __init__(self, name):
            super().__init__(name)
            raise RuntimeError("MLIAP error.")
