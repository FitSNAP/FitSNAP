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
  loss function. Declaring this parameter will override the weights in the GROUPS section for all configs.

- :code:`force_weight` is a scalar constant multiplied by the mean squared force error in the loss
  function. Declaring this parameter will override the weights in the GROUPS section for all configs.

- :code:`training_fraction` is a decimal fraction of how much of the total data should be trained
  on. The leftover code:`1.0 - training_fraction` portion is used for calculating validation errors
  during a fit.

- :code:`multi_element_option` is a scalar that determines how to handle multiple element types.

    - 1: All element types share the same network. Descriptors may still be different per type.
    - 2: Each element type has its own network.
    - 3: (Coming soon) One-hot encoding of element types, where each type shares the same network.

- :code:`num_elements` number of unique atom elements, or more specifically number of unique 
  networks.




