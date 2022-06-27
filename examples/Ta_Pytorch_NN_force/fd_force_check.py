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
        #print(x_derivatives.size())
        nbatch = int(x.size()[0]/natoms)
        #print(f"{nbatch} configs in this batch")

        # Calculate energies
        predicted_energy_total = torch.zeros(atoms_per_structure.size()).double()
        predicted_energy_total.index_add_(0, indices, self.network_architecture(x).squeeze())


        # Calculate forces
        x_indices = force_indices[0::3]
        y_indices = force_indices[1::3]
        z_indices = force_indices[2::3]
        #print(np.shape(force_indices))
        atom_indices = torch.tensor(force_indices[0::3,1].astype(int),dtype=torch.long) # Atoms i are repeated for each cartesian direction
        neigh_indices = torch.tensor(force_indices[0::3,0].astype(int),dtype=torch.long) # Neighbors j are repeated for each cartesian direction
        #print(neigh_indices.size())
        #print(int(neigh_indices))
        #dEdD = torch.autograd.grad(self.network_architecture(x), x, grad_outputs=torch.ones_like(self.network_architecture(x)))
        dEdD = torch.autograd.grad(self.network_architecture(x), x, grad_outputs=torch.ones_like(self.network_architecture(x)))
        #print(dEdD[0])
        dEdD = dEdD[0][neigh_indices,:].double() # These need to be dotted with dDdR in the x, y, and z directions.
        #print(dEdD)
        dDdRx = xd[0::3]
        dDdRy = xd[1::3]
        dDdRz = xd[2::3]
        #print(dDdRx)
        #print(x)
        #print(dEdD.size())
        #print(dDdRx.size())
        elementwise_x = torch.mul(dDdRx, dEdD)
        elementwise_y = torch.mul(dDdRy, dEdD)
        elementwise_z = torch.mul(dDdRz, dEdD)
        #print(elementwise)
        # Need to contract these along rows with indices given by force_indices[:,1]
        #print(atom_indices)
        fx_components = torch.zeros(natoms,nd).double()
        fy_components = torch.zeros(natoms,nd).double()
        fz_components = torch.zeros(natoms,nd).double()
        #print(fx_components.size())
        contracted_x = fx_components.index_add_(0,atom_indices,elementwise_x)
        contracted_y = fy_components.index_add_(0,atom_indices,elementwise_y)
        contracted_z = fz_components.index_add_(0,atom_indices,elementwise_z)
        #print(contracted.size())
        # Sum along bispectrum components to get force on each atom.
        predicted_fx = torch.sum(contracted_x, dim=1)
        predicted_fy = torch.sum(contracted_y, dim=1)
        predicted_fz = torch.sum(contracted_z, dim=1)
        # Reshape to get 2D tensor
        predicted_fx = torch.reshape(predicted_fx, (natoms,1))
        predicted_fy = torch.reshape(predicted_fy, (natoms,1))
        predicted_fz = torch.reshape(predicted_fz, (natoms,1))
        # Concatenate along the columns
        predicted_forces = torch.cat((predicted_fx,predicted_fy,predicted_fz), dim=1)
        #print(predicted_forces.size())
        #print(x)
        #print(dEdD)
        #predicted_forces = torch.zeros(nconfigs*natoms)
        """
        # Loop over all configs given by number of rows in descriptors array
        for m in range(0,nbatch):
            for i in range(0,natoms):
                # Loop over neighbors of i
                numneighs_i = len(neighlists[m,i])
                for jj in range(0,numneighs_i):
                    j = neighlists[m,i,jj]
                    jtag = tags[m,j]
                    for k in range(0,nd):
                        predicted_forces[natoms*m + i] -= x_derivatives[natoms*m + i,(jj*nd)+k]*dEdD[0][natoms*m + jtag,k]
        """

        return (predicted_energy_total, predicted_forces)
        #return predicted_energy_total

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

# LAMMPS setup commands
def prepare_lammps(seed):

    lmp.command("clear")
    lmp.command("units metal")
    lmp.command("boundary	p p p")
    lmp.command("atom_modify	map hash")
    lmp.command(f"lattice         bcc {latparam}")
    lmp.command(f"region		box block 0 {nx} 0 {ny} 0 {nz}")
    lmp.command(f"create_box	{ntypes} box")
    lmp.command(f"create_atoms	{ntypes} box")
    lmp.command("mass 		* 180.88")
    #lmp.command("displace_atoms 	all random 0.1 0.1 0.1 123456")
    lmp.command(f"displace_atoms 	all random 0.1 0.1 0.1 {seed}")
    lmp.command(f"pair_style zero 7.0")
    lmp.command(f"pair_coeff 	* *")
    #lmp.command(f"compute 	snap all snap {snap_options}")
    lmp.command(f"compute 	snap all snap {snap_options}")
    #lmp.command(f"compute snapneigh all snapneigh {snap_options}")
    lmp.command(f"thermo 		100")


# Get model force
def calc_model_forces(x0, seed):
    natoms = int(np.shape(x0)[0]/3)
    prepare_lammps(seed)
    x0 = lmp.numpy.extract_atom("x").flatten()
    for indx in range(0,3*natoms):
        x[indx]=x0[indx]
    lmp.scatter_atoms("x",1,3,x)
    lmp.command(f"run 0")
    #blah = lmp.numpy.extract_atom("x").flatten()
    #print(blah)
    lmp_snap = lmp.numpy.extract_compute("snap",0, 2)
    #force_indices = lmp.numpy.extract_compute("snapneigh", 0, 2).astype(np.int32)
    #print(lmp_snap[16:,:])
    #print(force_indices)
    # Calculate energy
    #descriptors = lmp_snap[:natoms, 0:nd]
    descriptors = lmp_snap[:natoms, 3:(nd+3)]
    dDdR_length = np.shape(lmp_snap)[0]-natoms-1 #6
    #dDdR = lmp_snap[natoms:(natoms+dDdR_length), 0:nd]
    dDdR = lmp_snap[natoms:(natoms+dDdR_length), 3:(nd+3)]
    #force_indices = lmp_snap[natoms:(natoms+dDdR_length), nd:(nd+3)].astype(np.int32)
    force_indices = lmp_snap[natoms:(natoms+dDdR_length), 0:3].astype(np.int32)
    descriptors = torch.from_numpy(descriptors).double().requires_grad_()
    dDdR = torch.from_numpy(dDdR).double().requires_grad_()
    #print(descriptors)
    #print(np.shape(descriptors))
    #print(np.shape(dDdR))
    (energies, forces) = model(descriptors, dDdR, indices, num_atoms, force_indices)
    #print(energies)
    #e1 = energies.detach().numpy()[0]
    forces = forces.detach().numpy()
    return forces

# Get finite difference force
def calc_fd_forces(x0, seed):

    natoms = int(np.shape(x0)[0]/3)

    a = 0
    forces = np.zeros((natoms,3)) # Only x direction for now.
    for i in range(0,natoms):
        for a in range(0,3):

            atomindx = 3*i + a

            # +h
            prepare_lammps(seed)
            x0 = lmp.numpy.extract_atom("x").flatten()
            #print(x0)
            #xtmp = x0
            for indx in range(0,3*natoms):
                x[indx]=x0[indx]
            x[atomindx] += h
            x1 = x0
            x1[atomindx] += h
            #if (i==7):
            #    print(x[atomindx])
            lmp.scatter_atoms("x",1,3,x)
            lmp.command(f"run 0")
            #blah = lmp.numpy.extract_atom("x").flatten()
            #print(blah)
            lmp_snap = lmp.numpy.extract_compute("snap",0, 2)
            lmp_snap1 = lmp_snap
            #print(lmp_snap[16:,:])
            #print(force_indices)
            # Calculate energy
            #descriptors = lmp_snap[:natoms, 0:nd]
            descriptors = lmp_snap[:natoms, 3:(nd+3)]
            d1 = descriptors
            #if (i==7):
            #    print(descriptors)
            dDdR_length = np.shape(lmp_snap)[0]-natoms-1 #6
            #dDdR = lmp_snap[natoms:(natoms+dDdR_length), 0:nd]
            dDdR = lmp_snap[natoms:(natoms+dDdR_length), 3:(nd+3)]
            #force_indices = lmp_snap[natoms:(natoms+dDdR_length), nd:(nd+3)].astype(np.int32)
            force_indices = lmp_snap[natoms:(natoms+dDdR_length), 0:3].astype(np.int32)
            descriptors = torch.from_numpy(descriptors).double().requires_grad_()
            dDdR = torch.from_numpy(dDdR).double().requires_grad_()
            #print(descriptors)
            (energies, force_junk) = model(descriptors, dDdR, indices, num_atoms, force_indices)
            #print(energies)
            e1 = energies.detach().numpy()[0]

            # -h
            prepare_lammps(seed)
            x0 = lmp.numpy.extract_atom("x").flatten()
            for indx in range(0,3*natoms):
                x[indx]=x0[indx]
            x[atomindx] -= h
            x2 = x0
            x2[atomindx] -= h
            #if (i==7):
            #    print(x[atomindx])
            lmp.scatter_atoms("x",1,3,x)
            lmp.command(f"run 0")
            #blah = lmp.numpy.extract_atom("x").flatten()
            #print(blah)
            lmp_snap = lmp.numpy.extract_compute("snap",0, 2)
            lmp_snap2 = lmp_snap
            #print(lmp_snap[16:,:])
            #print(force_indices)
            # Calculate energy
            #descriptors = lmp_snap[:natoms, 0:nd]
            descriptors = lmp_snap[:natoms, 3:(nd+3)]
            d2 = descriptors
            #if ((i==7) and (a==2)):
            #    #print(descriptors)
            #    d_diff = np.abs(d1-d2)
            #    print(d1)
            #    print(d2)
            #    print(d_diff)
            #print(descriptors)
            dDdR_length = np.shape(lmp_snap)[0]-natoms-1 #6
            #dDdR = lmp_snap[natoms:(natoms+dDdR_length), 0:nd]
            dDdR = lmp_snap[natoms:(natoms+dDdR_length), 3:(nd+3)]
            #force_indices = lmp_snap[natoms:(natoms+dDdR_length), nd:(nd+3)].astype(np.int32)
            force_indices = lmp_snap[natoms:(natoms+dDdR_length), 0:3].astype(np.int32)
            descriptors = torch.from_numpy(descriptors).double().requires_grad_()
            dDdR = torch.from_numpy(dDdR).double().requires_grad_()
            (energies, force_junk) = model(descriptors, dDdR, indices, num_atoms, force_indices)
            e2 = energies.detach().numpy()[0]

            #if (i==7):
            #    print(f"{e1} {e2}")
            """
            if ((i==7) and (a==2)):
                #print(descriptors)
                d_diff = np.abs(d1-d2)
                print("d1:")
                #print(d1)
                print(lmp_snap1)
                print("x1:")
                print(x1)
                print("d2:")
                #print(d2)
                print(lmp_snap2)
                print("x2:")
                print(x2)
                print("d_diff:")
                print(d_diff)
                print(f"e1, e2: {e1} {e2}")
            """

            force = (e1-e2)/(2*h)
            forces[i,a]=force

    return forces

# Finite difference parameters
h = 1e-4
# Other parameters
nconfigs=1

# Simulation parameters
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
#snap_options=f'{rcutfac} {rfac0} {twojmax} {radelem1} {radelem2} {wj1} {wj2} rmin0 {rmin0} quadraticflag {quadratic} bzeroflag {bzero} switchflag {switch}'
snap_options=f'{rcutfac} {rfac0} {twojmax} {radelem1} {radelem2} {wj1} {wj2} rmin0 {rmin0} quadraticflag {quadratic} bzeroflag {bzero} switchflag {switch} bikflag {bikflag} dgradflag {dgradflag}'

lmp = lammps(cmdargs=["-log", "none", "-screen", "none"])

# Get positions, natoms, number descriptors, length of dgrad
prepare_lammps(1)
lmp.command(f"run 0")
# These need to be run after run 0 otherwise you'll get a segfault since compute variables don't get initialized.
lmp_snap = lmp.numpy.extract_compute("snap",0, 2)
natoms = lmp.get_natoms()
dDdR_length = np.shape(lmp_snap)[0]-natoms-1 #6
#dDdR = lmp_snap[natoms:(natoms+dDdR_length), :]
dDdR = lmp_snap[natoms:(natoms+dDdR_length), 3:(nd+3)]
#print(lmp_snap)
#force_indices = lmp.numpy.extract_compute("snapneigh", 0, 2).astype(np.int32)
#force_indices = lmp_snap[natoms:(natoms+dDdR_length), nd:(nd+3)].astype(np.int32)
#print(lmp_snap[16:,:])
#print(np.shape(force_indices))
#print(force_indices[0:34,:])
x0 = lmp.numpy.extract_atom("x").flatten()
#natoms = lmp.get_natoms()
#descriptors = lmp_snap[:natoms, :]
descriptors = lmp_snap[:natoms, 3:(nd+3)]
#force_indices = lmp_snap[natoms:(natoms+dDdR_length), nd:(nd+3)].astype(np.int32)
force_indices = lmp_snap[natoms:(natoms+dDdR_length), 0:3].astype(np.int32)
x_indices = force_indices[0::3]
y_indices = force_indices[1::3]
z_indices = force_indices[2::3]
#nd = np.shape(descriptors)[1]
#print(np.shape(dDdR)) # Should be same as force_indices
#print(np.shape(force_indices))
# Define indices upon which to contract per-atom energies
indices = []
for m in range(0,nconfigs):
    for i in range(0,natoms):
        indices.append(m)
indices = torch.tensor(indices, dtype=torch.int64)
# Number of atoms per config is needed for future energy calculation.
num_atoms = natoms*torch.ones(nconfigs,dtype=torch.int32)

#Define the network parameters based on number of descriptors
#layer_sizes = ['num_desc', '10', '8', '6', '1'] # FitSNAP style
#nd=K
print(f"number descriptors: {nd}")
layer_sizes = [nd, nd, nd, 1]

# Build the model
network_architecture = create_torch_network(layer_sizes)
"""
for name, param in network_architecture.named_parameters():
    print("-----")
    print(name)
    print(param)
"""
model = FitTorch(network_architecture, nd).double()

#i = 0
#a = 0
#atomindx = 3*i+a
n3 = 3*natoms
# Allocate c array for positions.
x = (n3*c_double)()

start = 1
#end = start+1
end = 101

errors = []
for seed in range(start,end):
    print(seed)
    # Get model forces
    model_forces = calc_model_forces(x0, seed)
    #print(model_forces)
    # Get finite difference forces
    fd_forces = calc_fd_forces(x0, seed)
    #print(fd_forces)
    # Calc difference
    diff = model_forces - fd_forces
    #print(type(diff))
    #print(type(fd_forces))
    percent_error = np.divide(diff, fd_forces)*100
    #percent_error = torch.div(diff,fd_forces)*100
    #print(percent_error)
    percent_error = percent_error.flatten()
    #print(percent_error)
    errors.append(percent_error)

errors = np.abs(np.array(errors))
errors = errors.flatten()
errors[errors == -np.inf] = 0.1
errors[errors == np.inf] = 0.1
errors[errors == 0] = 0.1
errors[errors == 0] = 0.1
#print(errors)

n_bins = 50

# histogram on linear scale
#plt.subplot(111)
hist, bins, _ = plt.hist(errors, bins=n_bins)
logbins = np.logspace(np.log10(bins[0]),np.log10(bins[-1]),len(bins))
#plt.hist(errors, bins=logbins)
#plt.xscale('log')
#plt.show()


fig, axs = plt.subplots(1, 1, sharey=True, tight_layout=True)
#logbins = np.logspace(np.log10(bins[0]),np.log10(bins[-1]),len(bins))
# We can set the number of bins with the *bins* keyword argument.
axs.hist(errors, bins=logbins)
axs.set_xscale('log')
axs.set_xlabel("Force component percent error (%)")

fig.savefig("fd_force_check.png", dpi=500)
