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

def calculate_bessel(r, n, rc):
    """
    Calculate radial bessel functions for all pairs

    Attributes
    ----------

    rij: torch.Tensor.float
        Pairwise distance tensor with size (number_neigh, 1)

    n: torch.Tensor.float
        Integer in float form representing Bessel radial parameter n

    Returns
    -------

    rbf: torch.Tensor.float
        Radial Bessel function calculation for base n, has size (number_neigh, 1)
    """

    # calculate Bessel

    pi = np.pi
    rbf = np.divide(np.sqrt(2./rc)*np.sin(((n*pi)/rc)*r), r)     

    return rbf

rc = 4.2
r = np.linspace(0.0001,rc,100)
h = r[1]-r[0]
r = np.expand_dims(r, axis=1)
num_rbf = 3
basis = np.concatenate([calculate_bessel(r, n, rc) for n in range(1,num_rbf+1)], axis=1)

print(np.shape(basis))
print(basis)
print(f"h: {h}")

fig, ax = plt.subplots()
for i in range(0,num_rbf):
    ax.plot(r, basis[:,i])

ax.set(xlabel='Radius r (A)', ylabel='B(r)',
       title='Radial Bessel Basis')
ax.grid()

fig.show()
fig.savefig("basis.png", dpi=500)
plt.show()