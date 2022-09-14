
import sys
from fitsnap3lib.solvers.solver import Solver
from fitsnap3lib.parallel_tools import ParallelTools
from fitsnap3lib.io.input import Config
from time import time
import numpy as np

#config = Config()
#pt = ParallelTools()

try:
    from fitsnap3lib.lib.neural_networks.pytorch import FitTorch
    from fitsnap3lib.tools.dataloaders import InRAMDatasetPyTorch, torch_collate, DataLoader
    import torch

    class PYTORCH(Solver):
        """
        A class to wrap Modules to ensure lammps mliap compatability.

        ...

        Attributes
        ----------
        optimizer : torch.optim.Adam
            Torch Adam optimization object

        model : torch.nn.Module
            Network model that maps descriptors to a per atom attribute

        loss_function : torch.loss.MSELoss
            Mean squared error loss function

        learning_rate: float
            Learning rate for gradient descent

        scheduler: torch.optim.lr_scheduler.ReduceLROnPlateau
            Learning rate scheduler

        device : torch.nn.Module (None)
            Accelerator device

        training_data: torch.utils.data.Dataset
            Torch dataset for loading in pieces of the A matrix

        training_loader: torch.utils.data.DataLoader
            Data loader for loading in datasets

        Methods
        -------
        create_datasets():
            Creates the dataset to be used for training and the data loader for the batch system.

        perform_fit():
            Performs the pytorch fitting for a lammps potential
        """

        def __init__(self, name):
            """
            Initializes attributes for the pytorch solver.

                Parameters:
                    name : Name of solver class

            """
            super().__init__(name, linear=False)

            self.pt = ParallelTools()
            self.config = Config()

            self.global_weight_bool = self.config.sections['PYTORCH'].global_weight_bool
            self.energy_weight = self.config.sections['PYTORCH'].energy_weight
            self.force_weight = self.config.sections['PYTORCH'].force_weight

            self.global_fraction_bool = self.config.sections['PYTORCH'].global_fraction_bool
            self.training_fraction = self.config.sections['PYTORCH'].training_fraction

            self.multi_element_option = self.config.sections["PYTORCH"].multi_element_option
            self.num_elements = self.config.sections["PYTORCH"].num_elements
            self.num_desc_per_element = self.config.sections["CALCULATOR"].num_desc/self.num_elements

            self.optimizer = None
            self.model = FitTorch(self.config.sections["PYTORCH"].networks,
                                  self.num_desc_per_element,
                                  self.energy_weight,
                                  self.force_weight,
                                  self.num_elements,
                                  self.multi_element_option)
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

            # observe if config GROUPS are getting shuffled before PyTorch shuffling:
            #print(self.pt.fitsnap_dict['Configs'])

            # TODO: when only fitting to energy, we don't need all this extra data

            self.total_data = InRAMDatasetPyTorch(self.pt.shared_arrays['a'].array,
                                                  self.pt.shared_arrays['b'].array,
                                                  self.pt.shared_arrays['c'].array,
                                                  self.pt.shared_arrays['t'].array,
                                                  self.pt.shared_arrays['w'].array,
                                                  self.pt.shared_arrays['dgrad'].array,
                                                  self.pt.shared_arrays['number_of_atoms'].array,
                                                  self.pt.shared_arrays['dbdrindx'].array,
                                                  self.pt.shared_arrays["number_of_dgradrows"].array,
                                                  self.pt.shared_arrays["unique_j_indices"].array)

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
            # not sure if shuffling=True works, 
            # but data.random_split() above shuffles the input data,
            # and the group fractions are shuffled with random_sampling=1 flag.

            self.training_loader = DataLoader(self.training_data,
                                              batch_size=self.config.sections["PYTORCH"].batch_size,
                                              shuffle=False,
                                              collate_fn=torch_collate,
                                              num_workers=0)
            self.validation_loader = DataLoader(self.validation_data,
                                              batch_size=self.config.sections["PYTORCH"].batch_size,
                                              shuffle=False,
                                              collate_fn=torch_collate,
                                              num_workers=0)

        #@pt.sub_rank_zero
        def perform_fit(self):
            """
            Performs the pytorch fitting for a lammps potential
            """

            @self.pt.sub_rank_zero
            def decorated_perform_fit():
                self.create_datasets()
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

                    ntypes = self.config.sections["PYTORCH"].num_elements
                    keys = [*state_dict.keys()]
                    for t in range(0,ntypes):
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
                for epoch in range(self.config.sections["PYTORCH"].num_epochs):
                    print(f"----- epoch: {epoch}")
                    start = time()

                    # loop over training data

                    train_losses_step = []
                    loss = None
                    self.model.train()
                    for i, batch in enumerate(self.training_loader):
                        descriptors = batch['x'].to(self.device).requires_grad_(True)
                        atom_types = batch['t'].to(self.device)
                        targets = batch['y'].to(self.device).requires_grad_(True)
                        target_forces = batch['y_forces'].to(self.device).requires_grad_(True)
                        indices = batch['i'].to(self.device)
                        num_atoms = batch['noa'].to(self.device)
                        weights = batch['w'].to(self.device)
                        dgrad = batch['dgrad'].to(self.device).requires_grad_(True)
                        dbdrindx = batch['dbdrindx'].to(self.device)
                        unique_j = batch['unique_j'].to(self.device)
                        (energies,forces) = self.model(descriptors, dgrad, indices, num_atoms, atom_types, dbdrindx, unique_j, self.device)
                        energies = torch.div(energies,num_atoms)

                        # make indices showing which config a force belongs to

                        indices_forces = torch.repeat_interleave(indices, 3)
                        force_weights = weights[indices_forces,1]

                        if (self.energy_weight != 0):
                            energies = energies.to(self.device)
                        if (self.force_weight != 0):
                            forces = forces.to(self.device)

                        if (epoch == self.config.sections["PYTORCH"].num_epochs-1):

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
                        descriptors = batch['x'].to(self.device).requires_grad_(True)
                        atom_types = batch['t'].to(self.device)
                        targets = batch['y'].to(self.device).requires_grad_(True)
                        target_forces = batch['y_forces'].to(self.device).requires_grad_(True)
                        indices = batch['i'].to(self.device)
                        num_atoms = batch['noa'].to(self.device)
                        weights = batch['w'].to(self.device)
                        dgrad = batch['dgrad'].to(self.device).requires_grad_(True)
                        dbdrindx = batch['dbdrindx'].to(self.device)
                        unique_j = batch['unique_j'].to(self.device)
                        (energies,forces) = self.model(descriptors, dgrad, indices, num_atoms, atom_types, dbdrindx, unique_j, self.device)
                        energies = torch.div(energies,num_atoms)
                        
                        # make indices showing which config a force belongs to

                        indices_forces = torch.repeat_interleave(indices, 3)
                        force_weights = weights[indices_forces,1]

                        if (self.energy_weight != 0):
                            energies = energies.to(self.device)
                        if (self.force_weight != 0):
                            forces = forces.to(self.device)

                        if (epoch == self.config.sections["PYTORCH"].num_epochs-1):

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
                    if epoch % self.config.sections['PYTORCH'].save_freq == 0:
                        torch.save({
                            'epoch': epoch,
                            'model_state_dict': self.model.state_dict(),
                            'optimizer_state_dict': self.optimizer.state_dict(),
                            'loss': loss},
                            self.config.sections['PYTORCH'].save_state_output
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

                epochs = np.arange(self.config.sections["PYTORCH"].num_epochs)
                epochs = np.array([epochs]).T
                train_losses_epochs = np.array([train_losses_epochs]).T
                val_losses_epochs = np.array([val_losses_epochs]).T
                loss_dat = np.concatenate((epochs,train_losses_epochs,val_losses_epochs),axis=1)
                np.savetxt("loss_vs_epochs.dat", loss_dat)

                self.pt.single_print("Average loss over batches is", np.mean(np.asarray(train_losses_step)))
                
                if 'lammps.mliap' in sys.modules:
                    self.model.write_lammps_torch(self.config.sections["PYTORCH"].output_file)
                else:
                    print("Warning: This interpreter is not compatible with python-based mliap for LAMMPS. If you are using a Mac please make sure you have compiled python from source with './configure --enabled-shared' ")
                    print("Warning: FitSNAP will continue without ML-IAP")
                
                self.fit = None

            decorated_perform_fit()


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
