from __future__ import print_function
import sys
import ctypes
from ctypes import c_double
import numpy as np
from lammps import lammps, LMP_TYPE_ARRAY, LMP_STYLE_GLOBAL
from matplotlib import pyplot as plt
#plt.rcParams.update({'font.size': 18})
SMALL_SIZE = 12
MEDIUM_SIZE = 16
BIGGER_SIZE = 20
plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
import torch
from torch import nn
from torch import tensor

def create_torch_network(layer_sizes):
    """
    Creates a pytorch network architecture from layer sizes.
    This also performs standarization in the first linear layer.
    This only supports softplus as the nonlinear activation function.

        Parameters:
            layer_sizes (list of ints): Size of each network layers

        Return:
            Network Architecture of type neural network sequential

    """
    layers = []
    try:

        layers.append(torch.nn.Linear(layer_sizes[0], layer_sizes[0]))
        for i, layer in enumerate(layer_sizes):
            layers.append(torch.nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
            layers.append(torch.nn.Softplus())
            #layers.append(torch.nn.ReLU())
    except IndexError:
        layers.pop()

    # Fill weights with ones
    """
    nlayers = len(layers)
    print(f"{nlayers} layers.")
    for l in range(0,nlayers):
        print(layers[l])
        if (isinstance(layers[l],nn.Linear)):
            print(f"Linear layer l={l}")
            layers[l].weight.data.fill_(1.0)
            layers[l].bias.data.fill_(0.05)
    """

    return torch.nn.Sequential(*layers)

"""
Define the model
"""
class FitTorch(torch.nn.Module):
    """
    FitSNAP PyTorch Neural Network Architecture Model
    Currently only fits on energies
    """

    def __init__(self, network_architecture, descriptor_count, n_elements=1):
        """
        Saves lammps ready pytorch model.

            Parameters:
                network_architecture : A nn.Sequential network architecture
                descriptor_count (int): Length of descriptors for an atom
                n_elements (int): Number of differentiable atoms types

        """
        super().__init__()
        self.network_architecture = network_architecture.double()
        self.desc_len = descriptor_count
        self.n_elem = n_elements

    def forward(self, x, xd, indices, atoms_per_structure, force_indices):
        """
        Saves lammps ready pytorch model.

            Parameters:
                x (tensor of floats): Array of descriptors
                x_derivatives (tensor of floats): Array of descriptor derivatives
                indices (tensor of ints): Array of indices upon which to contract per atom energies
                atoms_per_structure (tensor of ints): Number of atoms per configuration

        """

        nbatch = int(x.size()[0]/natoms)

        # calculate energies

        predicted_energy_total = torch.zeros(atoms_per_structure.size()).double()
        predicted_energy_total.index_add_(0, indices, self.network_architecture(x).squeeze())


        # calculate forces

        x_indices = force_indices[0::3]
        y_indices = force_indices[1::3]
        z_indices = force_indices[2::3]
        atom_indices = torch.tensor(force_indices[0::3,1].astype(int),dtype=torch.long) # atoms i are repeated for each cartesian direction
        neigh_indices = torch.tensor(force_indices[0::3,0].astype(int),dtype=torch.long) # neighbors j are repeated for each cartesian direction

        dEdD = torch.autograd.grad(self.network_architecture(x), x, grad_outputs=torch.ones_like(self.network_architecture(x)))
        dEdD = dEdD[0][neigh_indices,:].double() # these need to be dotted with dDdR in the x, y, and z directions.

        dDdRx = xd[0::3]
        dDdRy = xd[1::3]
        dDdRz = xd[2::3]

        elementwise_x = torch.mul(dDdRx, dEdD)
        elementwise_y = torch.mul(dDdRy, dEdD)
        elementwise_z = torch.mul(dDdRz, dEdD)

        # need to contract these along rows with indices given by force_indices[:,1]

        fx_components = torch.zeros(natoms,nd).double()
        fy_components = torch.zeros(natoms,nd).double()
        fz_components = torch.zeros(natoms,nd).double()

        contracted_x = fx_components.index_add_(0,atom_indices,elementwise_x)
        contracted_y = fy_components.index_add_(0,atom_indices,elementwise_y)
        contracted_z = fz_components.index_add_(0,atom_indices,elementwise_z)

        # sum along bispectrum components to get force on each atom.

        predicted_fx = torch.sum(contracted_x, dim=1)
        predicted_fy = torch.sum(contracted_y, dim=1)
        predicted_fz = torch.sum(contracted_z, dim=1)

        # reshape to get 2D tensor

        predicted_fx = torch.reshape(predicted_fx, (natoms,1))
        predicted_fy = torch.reshape(predicted_fy, (natoms,1))
        predicted_fz = torch.reshape(predicted_fz, (natoms,1))

        # concatenate along the columns

        #predicted_forces = -1.0*torch.cat((predicted_fx,predicted_fy,predicted_fz), dim=1)
        # no need to multiply by -1 since compute snap already does this for us

        predicted_forces = torch.cat((predicted_fx,predicted_fy,predicted_fz), dim=1)

        return (predicted_energy_total, predicted_forces)

    def import_wb(self, weights, bias):
        """
        Imports weights and bias into FitTorch model

            Parameters:
                weights (list of numpy array of floats): Network weights at each layer
                bias (list of numpy array of floats): Network bias at each layer

        """

        assert len(weights) == len(bias)
        imported_parameter_count = sum(w.size + b.size for w, b in zip(weights, bias))
        combined = [None] * (len(weights) + len(bias))
        combined[::2] = weights
        combined[1::2] = bias

        assert len([p for p in self.network_architecture.parameters()]) == len(combined)
        assert sum(p.nelement() for p in self.network_architecture.parameters()) == imported_parameter_count

        state_dict = self.state_dict()
        for i, key in enumerate(state_dict.keys()):
            state_dict[key] = torch.tensor(combined[i])
        self.load_state_dict(state_dict)

def prepare_lammps(seed):
    """
    LAMMPS setup commands. Prepares LAMMPS to do a calculation.
    """

    lmp.command("clear")
    lmp.command("units metal")
    lmp.command("boundary	p p p")
    lmp.command("atom_modify	map hash")
    lmp.command(f"lattice         bcc {latparam}")
    lmp.command(f"region		box block 0 {nx} 0 {ny} 0 {nz}")
    lmp.command(f"create_box	{ntypes} box")
    lmp.command(f"create_atoms	{ntypes} box")
    lmp.command("mass 		* 180.88")
    lmp.command(f"displace_atoms 	all random 0.1 0.1 0.1 {seed}")
    lmp.command(f"pair_style zero 7.0")
    lmp.command(f"pair_coeff 	* *")
    lmp.command(f"compute 	snap all snap {snap_options}")
    lmp.command(f"thermo 		100")


# Get model force
def calc_model_forces(x0, seed):
    """
    Calculate model forces using the NN expression for forces, and descriptor
    gradients extracted from LAMMPS.
    """
    natoms = int(np.shape(x0)[0]/3)
    prepare_lammps(seed)
    x0 = lmp.numpy.extract_atom("x").flatten()
    for indx in range(0,3*natoms):
        x[indx]=x0[indx]
    lmp.scatter_atoms("x",1,3,x)
    lmp.command(f"run 0")

    lmp_snap = lmp.numpy.extract_compute("snap",0, 2)

    descriptors = lmp_snap[:natoms, 3:(nd+3)]
    dDdR_length = np.shape(lmp_snap)[0]-natoms-1
    dDdR = lmp_snap[natoms:(natoms+dDdR_length), 3:(nd+3)]
    force_indices = lmp_snap[natoms:(natoms+dDdR_length), 0:3].astype(np.int32)

    # strip rows with all zero descriptor gradients to save memory

    nonzero_rows = lmp_snap[natoms:(natoms+dDdR_length),3:(nd+3)] != 0.0
    nonzero_rows = np.any(nonzero_rows, axis=1)
    dDdR = dDdR[nonzero_rows, :]
    force_indices = force_indices[nonzero_rows,:]
    dDdR_length = np.shape(dDdR)[0]

    descriptors = torch.from_numpy(descriptors).double().requires_grad_()
    dDdR = torch.from_numpy(dDdR).double().requires_grad_()

    (energies, forces) = model(descriptors, dDdR, indices, num_atoms, force_indices)

    forces = forces.detach().numpy()
    return forces

def calc_fd_forces(x0, seed):
    """
    Calculate finite difference force.
    We take finite differences between +h and -h on all atoms, using the PyTorch
    model to calculated energy.
    """

    natoms = int(np.shape(x0)[0]/3)

    a = 0
    forces = np.zeros((natoms,3)) # only x direction for now.
    for i in range(0,natoms):
        for a in range(0,3):

            atomindx = 3*i + a

            # +h

            prepare_lammps(seed)
            x0 = lmp.numpy.extract_atom("x").flatten()

            for indx in range(0,3*natoms):
                x[indx]=x0[indx]
            x[atomindx] += h
            x1 = x0
            x1[atomindx] += h

            lmp.scatter_atoms("x",1,3,x)
            lmp.command(f"run 0")

            lmp_snap = lmp.numpy.extract_compute("snap",0, 2)
            lmp_snap1 = lmp_snap

            # calculate energy

            descriptors = lmp_snap[:natoms, 3:(nd+3)]
            d1 = descriptors
            dDdR_length = np.shape(lmp_snap)[0]-natoms-1
            dDdR = lmp_snap[natoms:(natoms+dDdR_length), 3:(nd+3)]
            force_indices = lmp_snap[natoms:(natoms+dDdR_length), 0:3].astype(np.int32)

            # strip rows with all zero descriptor gradients to save memory

            nonzero_rows = lmp_snap[natoms:(natoms+dDdR_length),3:(nd+3)] != 0.0
            nonzero_rows = np.any(nonzero_rows, axis=1)
            dDdR = dDdR[nonzero_rows, :]
            force_indices = force_indices[nonzero_rows,:]
            dDdR_length = np.shape(dDdR)[0]

            descriptors = torch.from_numpy(descriptors).double().requires_grad_()
            dDdR = torch.from_numpy(dDdR).double().requires_grad_()
            (energies, force_junk) = model(descriptors, dDdR, indices, num_atoms, force_indices)
            e1 = energies.detach().numpy()[0]

            # -h

            prepare_lammps(seed)
            x0 = lmp.numpy.extract_atom("x").flatten()
            for indx in range(0,3*natoms):
                x[indx]=x0[indx]
            x[atomindx] -= h
            x2 = x0
            x2[atomindx] -= h

            lmp.scatter_atoms("x",1,3,x)
            lmp.command(f"run 0")

            lmp_snap = lmp.numpy.extract_compute("snap",0, 2)
            lmp_snap2 = lmp_snap

            # calculate energy

            descriptors = lmp_snap[:natoms, 3:(nd+3)]
            d2 = descriptors
            dDdR_length = np.shape(lmp_snap)[0]-natoms-1 #6
            #dDdR = lmp_snap[natoms:(natoms+dDdR_length), 0:nd]
            dDdR = lmp_snap[natoms:(natoms+dDdR_length), 3:(nd+3)]
            #force_indices = lmp_snap[natoms:(natoms+dDdR_length), nd:(nd+3)].astype(np.int32)
            force_indices = lmp_snap[natoms:(natoms+dDdR_length), 0:3].astype(np.int32)

            # strip rows with all zero descriptor gradients to save memory

            nonzero_rows = lmp_snap[natoms:(natoms+dDdR_length),3:(nd+3)] != 0.0
            nonzero_rows = np.any(nonzero_rows, axis=1)
            dDdR = dDdR[nonzero_rows, :]
            force_indices = force_indices[nonzero_rows,:]
            dDdR_length = np.shape(dDdR)[0]


            descriptors = torch.from_numpy(descriptors).double().requires_grad_()
            dDdR = torch.from_numpy(dDdR).double().requires_grad_()
            (energies, force_junk) = model(descriptors, dDdR, indices, num_atoms, force_indices)
            e2 = energies.detach().numpy()[0]

            # multiply by -1 since negative gradient

            force = -1.0*((e1-e2)/(2*h))
            forces[i,a]=force

    return forces

# finite difference parameters
h = 1e-4
nconfigs=1 # because we do 1 config at a time

# simulation parameters

nsteps=0
nrep=2
latparam=2.0
nx=nrep
ny=nrep
nz=nrep
ntypes=2

# SNAP options

twojmax=6
m = (twojmax/2)+1
K = int(m*(m+1)*(2*m+1)/6)
nd = K
print(f"nd : {K}")
rcutfac=1.0 #1.0
rfac0=0.99363
rmin0=0
radelem1=2.3
radelem2=2.0
wj1=1.0
wj2=0.96
quadratic=0
bzero=0
switch=1
bikflag=1
dgradflag=1
snap_options=f'{rcutfac} {rfac0} {twojmax} {radelem1} {radelem2} {wj1} {wj2} rmin0 {rmin0} quadraticflag {quadratic} bzeroflag {bzero} switchflag {switch} bikflag {bikflag} dgradflag {dgradflag}'

lmp = lammps(cmdargs=["-log", "none", "-screen", "none"])

# get positions, natoms, number descriptors, length of dgrad

prepare_lammps(1)
lmp.command(f"run 0")

# these need to be run after run 0 otherwise you'll get a segfault since compute variables don't get initialized.

lmp_snap = lmp.numpy.extract_compute("snap",0, 2)
natoms = lmp.get_natoms()
dDdR_length = np.shape(lmp_snap)[0]-natoms-1
dDdR = lmp_snap[natoms:(natoms+dDdR_length), 3:(nd+3)]
x0 = lmp.numpy.extract_atom("x").flatten()
descriptors = lmp_snap[:natoms, 3:(nd+3)]
force_indices = lmp_snap[natoms:(natoms+dDdR_length), 0:3].astype(np.int32)

# strip rows with all zero descriptor gradients to save memory

nonzero_rows = lmp_snap[natoms:(natoms+dDdR_length),3:(nd+3)] != 0.0
nonzero_rows = np.any(nonzero_rows, axis=1)
dDdR = dDdR[nonzero_rows, :]
force_indices = force_indices[nonzero_rows,:]
dDdR_length = np.shape(dDdR)[0]

x_indices = force_indices[0::3]
y_indices = force_indices[1::3]
z_indices = force_indices[2::3]

# define indices upon which to contract per-atom energies

indices = []
for m in range(0,nconfigs):
    for i in range(0,natoms):
        indices.append(m)
indices = torch.tensor(indices, dtype=torch.int64)

# number of atoms per config is needed for future energy calculation.

num_atoms = natoms*torch.ones(nconfigs,dtype=torch.int32)

# define the network parameters based on number of descriptors
#layer_sizes = ['num_desc', '10', '8', '6', '1'] # FitSNAP style

print(f"number descriptors: {nd}")
layer_sizes = [nd, nd, nd, 1]

# build the model

network_architecture = create_torch_network(layer_sizes)
"""
for name, param in network_architecture.named_parameters():
    print("-----")
    print(name)
    print(param)
"""
model = FitTorch(network_architecture, nd).double()

n3 = 3*natoms

# allocate c array for positions.

x = (n3*c_double)()

start = 1
end = 101

print(f"Calculating forces for {end-1} random configs...")

errors = []
for seed in range(start,end):
    print(seed)

    # get model forces

    model_forces = calc_model_forces(x0, seed)

    # gGet finite difference forces

    fd_forces = calc_fd_forces(x0, seed)

    # calc difference and error

    diff = model_forces - fd_forces
    percent_error = np.divide(diff, fd_forces)*100
    percent_error = percent_error.flatten()
    errors.append(percent_error)

errors = np.abs(np.array(errors))
errors = errors.flatten()
errors[errors == -np.inf] = 100.0
errors[errors == np.inf] = 100.0
errors[errors == 0] = 100.0
errors[errors == 0] = 100.0

n_bins = 50

# histogram on linear scale

hist, bins, _ = plt.hist(errors, bins=n_bins)
logbins = np.logspace(np.log10(bins[0]),np.log10(bins[-1]),len(bins))


fig, axs = plt.subplots(1, 1, sharey=True, tight_layout=True)
axs.hist(errors, bins=logbins)
axs.set_xscale('log')
axs.set_xlabel("Force component percent error (%)")

fig.savefig("fd_force_check.png", dpi=500)
