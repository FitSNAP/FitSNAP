PyTorch Models
==============

Interfacing with PyTorch allows us to conveniently fit neural network potentials using descriptors
that exist in LAMMPS. We may then use these neural network models to run high-performance MD 
simulations in LAMMPS. These capabilities are explained below.

Fitting Neural Network Potentials
---------------------------------

Similarly to how we fit linear models, we can input descriptors into nonlinear models such as 
neural networks. To do this, we can use the same FitSNAP input script that we use for linear 
models, with some slight changes to the sections. First we must add a :code:`PYTORCH` section, 
which for the tantalum example looks like::

    [PYTORCH]
    layer_sizes =  num_desc 60 60 1
    learning_rate = 1.5e-4 
    num_epochs = 100
    batch_size = 4
    save_state_output = Ta_Pytorch.pt
    energy_weight = 1e-2
    force_weight = 1.0
    training_fraction = 1.0
    multi_element_option = 1
    num_elements = 1

We must also add a :code:`nonlinear = 1` key in the :code:`CALCULATOR` section, and set 
:code:`solver = PYTORCH` in the :code:`SOLVER` section. Now the input script is ready to fit a 
neural network potential.

The :code:`PYTORCH` section keys are explained in more detail below.

- :code:`layer_sizes` determines the network architecture. We lead with a :code:`num_desc` parameter
  which tells FitSNAP that the number of nodes in the first layer are equal to the number of 
  descriptors. The argument here is a list where each element determines the number of nodes in 
  each layer.

- :code:`learning_rate` determines how fast the network minimizes the loss function. We find that
  a learning rate around :code:`1e-4` works well when fitting to forces, and when using our current
  loss function.

- :code:`num_epochs` sets the number of gradient descent iterations.

- :code:`batch_size` determines how many configs to average gradients for when looping over batches
  in a single epoch. We find that a batch size around 4 works well for our models.

- :code:`save_state_output` is the name of the PyTorch model file to write after every
  epoch. This model can be loaded for testing purposes later.

- :code:`save_state_input` is the name of a PyTorch model that may be loaded for the purpose of 
  restarting an existing fit, or for calculating test errors.

- :code:`energy_weight` is a scalar constant multiplied by the mean squared energy error in the 
  loss function. Declaring this parameter will override the weights in the GROUPS section for all 
  configs. We therefore call this the *global energy weight*. If you want to specify energy weights 
  for each group, do so in the GROUPS section.

- :code:`force_weight` is a scalar constant multiplied by the mean squared force error in the loss
  function. Declaring this parameter will override the weights in the GROUPS section for all 
  configs. We therefore call this the *global force weight*. If you want to specify force weights 
  for each group, do so in the GROUPS section.

- :code:`training_fraction` is a decimal fraction of how much of the total data should be trained
  on. The leftover :code:`1.0 - training_fraction` portion is used for calculating validation errors
  during a fit. Declaring this parameter will override the training/testing fractions in the GROUPS
  section for all configs. We therefore call this the *global training fraction*. If you want to 
  specify training/testing fractions for each group, do so in the GROUPS section.

- :code:`multi_element_option` is a scalar that determines how to handle multiple element types.

    - 1: All element types share the same network. Descriptors may still be different per type.
    - 2: Each element type has its own network.
    - 3: (Coming soon) One-hot encoding of element types, where each type shares the same network.

- :code:`num_elements` number of unique atom elements, or more specifically number of unique 
  networks.

- :code:`manual_seed_flag` set to 0 by default, can set to 1 if want to force a random seed which is
  useful for debugging purposes.


Outputs and Error Calculation
-----------------------------

FitSNAP outputs include files that aid in error calculation, and files that can be used to restart 
a fit or even run MD simulations in LAMMPS.

Error/Comparison files
^^^^^^^^^^^^^^^^^^^^^^

After training a potential, FitSNAP produces outputs that can be used to intrepret the quality of a 
fit on the training and/or validation data. The following comparison files are written after a fit:

- :code:`energy_comparison.dat` energy comparisons for all configs in the training set. Each row 
<<<<<<< HEAD
corresponds to a specific configuration in the training set. The first column is the model energy, 
and the 2nd column is the target energy. 
=======
  corresponds to a specific configuration in the training set. The first column is the model energy, 
  and the 2nd column is the target energy. 
>>>>>>> master

- :code:`energy_comparison_val.dat` energy comparisons for all configs in the validation set. 
  Format is same as above.

- :code:`force_comparison.dat` force comparisons for all atoms in all configs in the training set.
  Each row corresponds to a single atom's Cartesian component for a specific config in the training 
  set. The first column is the model energy, and the 2nd column is the target energy.

- :code:`force_comparison_val.dat` same as above, but for the validation set.

These outputs allow you to compare the configuration energies, or per-atom forces, however you want
after a fit. For example, in the `Ta_PyTorch_NN example <https://github.com/FitSNAP/FitSNAP/tree/master/examples/Ta_PyTorch_NN>`_
, we provide python scripts that help post-process these files to calculate mean absolute error or 
plot comparisons in energies and forces.

PyTorch model files
^^^^^^^^^^^^^^^^^^^

FitSNAP outputs two PyTorch :code:`.pt` models file after fitting. One is used for restarting a fit
based on an existing model, specifically the model name supplied by the user in the 
:code:`save_state_output` keyword of the input script. In the `Ta_PyTorch_NN example <https://github.com/FitSNAP/FitSNAP/tree/master/examples/Ta_PyTorch_NN>`_
we can see this keyword is :code:`Ta_Pytorch.pt`. This file will therefore be saved every epoch, and 
it may be fed into FitSNAP via the :code:`save_state_input` keyword to restart another fit from that
particular model.

The other PyTorch model is used for running MD simulations in LAMMPS after a fit. This file has the 
name :code:`FitTorch_Pytorch.pt`, and is used to run MD in LAMMPS via the ML-IAP package. An example 
is given for tantalum here: https://github.com/FitSNAP/FitSNAP/tree/master/examples/Ta_PyTorch_NN/MD 

Calculate errors on a test set
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Users may want to use models to calculate errors on a test set that was completely separate from the
training/validation sets used in fitting. To do this, we change the input script to read an existing
PyTorch model file, e.g. for Ta::

    [PYTORCH]
    layer_sizes =  num_desc 60 60 1
    learning_rate = 1.5e-4 
    num_epochs = 1 ##### Set to 1 for testing
    batch_size = 4
    save_state_input = Ta_Pytorch.pt ##### Load an existing model
    energy_weight = 1e-2
    force_weight = 1.0
    training_fraction = 1.0
    multi_element_option = 1
    num_elements = 1

Notice how we are now using :code:`save_state_input` instead of :code:`save_state_output`, and that 
we set :code:`num_epochs = 1`. This will load the existing PyTorch model, and perform a single epoch
which involves calculating the energy and force comparisons (mentioned above) for the current model, 
on whatever user-defined groups of configs in the groups section.We can therefore use the energy and 
force comparison files here to calculate mean absolute errors, e.g. with the script in 
the `Ta_PyTorch_NN example <https://github.com/FitSNAP/FitSNAP/tree/master/examples/Ta_PyTorch_NN>`_




