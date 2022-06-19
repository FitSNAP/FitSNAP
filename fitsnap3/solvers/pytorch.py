
from .solver import Solver
from ..parallel_tools import pt
from ..io.input import config
from time import time
import numpy as np

try:
    from ..lib.neural_networks.pytorch import FitTorch
    from ..tools.dataloaders import InRAMDatasetPyTorch, torch_collate, DataLoader
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
            self.optimizer = None
            self.model = FitTorch(config.sections["PYTORCH"].network_architecture,
                                  config.sections["CALCULATOR"].num_desc)
            self.loss_function = torch.nn.MSELoss()
            self.learning_rate = config.sections["PYTORCH"].learning_rate
            if config.sections['PYTORCH'].save_state_input is not None:
                try:
                    self.model.load_lammps_torch(config.sections['PYTORCH'].save_state_input)
                except AttributeError:
                    self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
                    save_state_dict = torch.load(config.sections['PYTORCH'].save_state_input)
                    self.model.load_state_dict(save_state_dict["model_state_dict"])
                    self.optimizer.load_state_dict(save_state_dict["optimizer_state_dict"])
            if self.optimizer is None:
                self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer,
                                                                        mode='min',
                                                                        factor=0.5,
                                                                        patience=49,
                                                                        verbose=True,
                                                                        threshold=0.0001,
                                                                        threshold_mode='abs')

            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            pt.single_print("Pytorch device is set to", self.device)
            self.training_data = None
            self.training_loader = None

        def create_datasets(self):
            """
            Creates the dataset to be used for training and the data loader for the batch system.
            """

            training = [not elem for elem in pt.fitsnap_dict['Testing']]

            self.training_data = InRAMDatasetPyTorch(pt.shared_arrays['a'].array[training],
                                                     pt.shared_arrays['b'].array[training])
            self.training_loader = DataLoader(self.training_data,
                                              batch_size=config.sections["PYTORCH"].batch_size,
                                              shuffle=False,
                                              collate_fn=torch_collate,
                                              num_workers=0)

        @pt.sub_rank_zero
        def perform_fit(self):
            """
            Performs the pytorch fitting for a lammps potential
            """
            self.create_datasets()
            if config.sections['PYTORCH'].save_state_input is None:
                # standardization
                inv_std = 1/np.std(pt.shared_arrays['a'].array, axis=0)
                mean_inv_std = np.mean(pt.shared_arrays['a'].array, axis=0) * inv_std
                state_dict = self.model.state_dict()
                keys = [*state_dict.keys()][:2]
                state_dict[keys[0]] = torch.tensor(inv_std)*torch.eye(len(inv_std))
                state_dict[keys[1]] = torch.tensor(mean_inv_std)
                self.model.load_state_dict(state_dict)

            for epoch in range(config.sections["PYTORCH"].num_epochs):
                start = time()
                # need to get take sub(A) which is of length num_configs*num_atoms_per_config

                train_losses_step = []
                loss = None
                for i, batch in enumerate(self.training_loader):
                    self.model.train()
                    descriptors = batch['x'].to(self.device).requires_grad_(True)
                    targets = batch['y'].to(self.device).requires_grad_(True)
                    indices = batch['i'].to(self.device)
                    num_atoms = batch['noa'].to(self.device)
                    energies = torch.reshape(self.model(descriptors, indices, num_atoms), (-1,)).to(self.device)
                    loss = self.loss_function(energies/num_atoms, targets)
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                    train_losses_step.append(loss.item())
                pt.single_print("Average loss over batches is", np.mean(np.asarray(train_losses_step)))
                pt.single_print("Epoch time", time()-start)
                if epoch % config.sections['PYTORCH'].save_freq == 0:
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'loss': loss},
                        config.sections['PYTORCH'].save_state_output
                    )
            self.model.write_lammps_torch(config.sections["PYTORCH"].output_file)
            self.fit = None


except ModuleNotFoundError:

    class Pytorch(Solver):
        """
        Dummy class for factory to read if torch is not available for import.
        """
        def __init__(self, name):
            super().__init__(name)
            raise ModuleNotFoundError("No module named 'Pytorch'")

except NameError:

    class Pytorch(Solver):
        """
        Dummy class for factory to read if MLIAP error is occuring.
        """
        def __init__(self, name):
            super().__init__(name)
            raise NameError("MLIAP error.")
