import numpy as np
from matplotlib import pyplot as plt

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

def calc_g3b(x, mu):
    """
    Calculate Gaussian 3 body descriptor for a given value x and mu, which represents the cosine angle 
    between a triplet of atoms ijk.
    """

    eta = 4.
    value = np.exp(-eta*(x-mu)**2)

    return(value)

num_descriptors = 21
x = np.linspace(-1,1,100)
mu = np.linspace(-1,1,num_descriptors)

for u in mu:
    plt.plot(x, calc_g3b(x,u), 'k-', markersize=1)

plt.xlabel(r"$cos \theta$")
#plt.show()
plt.savefig("plot_g3b.png", dpi=500)
