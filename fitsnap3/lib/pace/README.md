# Wigner_ACE

### This code can be used to define ACE descriptors

The primary usage will allow one to generate a lexicographically ordered list of
n,l combinations (nu vectors) allowed for scalar ACE descriptors. For these
descriptor labels, the code can calculate generalized Wigner-3j symbols up to
rank 8 (reduce the product of 8 spherical harmonics). As of right now, the code
only produces coupling coefficients needed to generate descriptors that
transform as scalars. The descriptors themselves may be generated internally
(for testing purposes only) or through exsisting procedures in the LAMMPS
ML-PACE package.

### SETUP

Add this module folder to your $PYTHONPATH and execute the coupling_coefficients.py script.
Upon first use, the code will generate pickled libraries of Wigner 3-j symbols and
Clebsch-Gordan coefficients. This is time-consuming but only happens once. After this,
the functions in the module should load and run quickly.

### USAGE

To generate the descriptor labels, (nu), the generate_nl function may be used.
As input, it takes the rank of the descriptor labels to be generated, the
maximum n and maximum l quantum numbers that are allowed. All descriptor labels
will be generated up to these maximum values.

<pre><code>
from gen_labels import *
rank = 4
nmax = 2
lmax = 2

rank4_nus = generate_nl(rank,nmax,lmax)

</code></pre>
and returns a list of nu vectors in string format. Note that it will not
generate the labels for each rank up to the rank specified. Descriptor labels
for each rank must be generated separately.

For a certain descriptor, the generalized Wigner-3j symbols may be obtained.
Only the l quantum numbers are needed to generate the coefficient:

<pre><code>
from gen_labels import *
from wigner_couple import *

nu_test = rank4_nus[0]

n,l = get_n_l(nu_test)

w3js = rank_4(l)

</code></pre>
and returns a dictionary of generalized Wigner-3j symbols with magnetic quantum
number vectors as keys and coupling coefficients as values. The dictionary
contains all m combinations needed to construct a scalar invariant descriptor
(and have already been summed over intermediate angular momenta - See Yutsis
1962). 

### Generating coupling_coefficients.ace files for lammps computes

As mentioned before, the descriptors may be constructed internally for testing
purposes, or may be constructed by the LAMMPS ML-PACE package. To do the latter,
use the write_pot function:

<pre><code>
from gen_labels import *
from coupling_coeffs import *
from wigner_couple import *

#dictionary of lmax and nmax for all ranks
#note that without editing the pickled coefficient libraries,
#  the maximum l allowed for rank 8 descriptors is 2
lmax_dict = {1:6,2:6,3:4,4:2,5:2,6:2,7:1,8:1}
nradmax_dict = {1:16,2:6,3:4,4:2,5:2,6:2,7:2,8:2}

#list of descriptor ranks (maximum allowed is 8)
maxrank = 8
ranks = range(1,maxrank+1)
#Lexicographically ordered list of n,l tuples that obey appropriate triangle conditions
ranked_nus = [generate_nl(rank,nradmax_dict[rank],lmax_dict[rank]) for rank in ranks]
nus = [item for sublist in ranked_nus for item in sublist]

#get the generalized wigner-3j symbols for all descriptor labels
couplings, = get_coupling(nus)

#initialize an array of dummy ctilde coefficients (must be initialized to 1 for potential fitting)
coeff_arr = np.ones(len(nus))

coeffs = {nu:coeff for nu,coeff in zip(nus,coeff_arr)}

#write a "potential" containing all ACE model information (compatible with LAMMPS)

# ACE model parameters
radial_cutoff = 7.5
# Exponential lambda (decay parameter for scaled position in Drautz 2019)
lambd = 5.0
# reference energy
e0 = 0.0


filname = 'coupling_coefficients'
element = 'W'

write_pot(filname,element,ranks,lmax=max(lmax_dict.values()),nradbase=nradmax_dict[1],\ 
nradmax=max(nradmax_dict.values()), rcut=radial_cutoff, exp_lambda=lambd,\
nus=nus,coupling=coupling,coeffs=coeffs,E_0=e0)

</code></pre>
This will write a 'coupling_coefficients.ace' file that is compatible with the
LAMMPS ML-PACE package. See separate documentation for the LAMMPS compute,
compute pace , for details.

### Validation

You may test the rotational invariance of the descriptors in the 'unit_tests' folder. 
The 'unit_tests.py' script in this folder will calculate descriptors up to the
maximum possible rank implemented, and will report the difference between 
descriptors evaluated under some random initial coordinates and those evaluated
from those coordinates with an asymmetric rotation applied. See the reference
file for an example of the expected value range.
