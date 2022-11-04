# Permutation-adapted descriptor labels ACE and n-bond functions of R3

Requirements for library:
sympy 1.9 +


# RI code usage (bond angle basis)
Returns the PA-R(P)I descriptor labels for a bond angle basis. These labels
are generated for a bond angle basis that **has** been symmetrized with 
respect to permutations as well. In the case that no $S_N$ symmetrization
is done, the PA angular basis labels are essentially the same as those from
the canonical angular basis (c.f. Table I of Goff et al.). 

<pre><code>
from rpi_lib import *
l=[1,1,2,2]
varsigma_lL=permutation_adapted_lL(l)
</code></pre>
These are the angular basis descriptor labels that are unique up to 
automorphism.

# Permutation-adapted RPI code usage (bond angle + radial basis)
Returns the PA-RPI descriptor labels a bond angle + bond length basis 
that **has** been symmetrized with respect to permutations. 

<pre><code>
from rpi_lib import *
n=[1,1,1,2]
l=[1,1,2,2]
varsigma_nlLs=permutation_adapted_nlL(n,l)
</code></pre>
These are the PA-RPI (radial + angular) descriptor labels that are unique
up to automorphism. In this simplified code, the smallest valid intermediate
is chosen for a given tree structure. The others are linearly dependent by
recursion relationships from p. 224 of Rose's "Elementary Theory of Angular 
Momentum" (1957).

# Exhaustive descriptor label generation
Generates all PA-RPI PA-RPI (radial + angular) descriptor labels up to a 
specified rank, maximum l quantum number, and maximum n quantum number.
This is currently hard coded up to rank 6. Higher rank descriptors and a 
recursive generation are under development.

<pre><code>
from rpi_lib import *
rank = 4
nmax = 4
lmax = 4
symmetric_set = descriptor_labels_YSG(rank,nmax,lmax)
</code></pre>

# Example descriptor calculations:
Examples for calculating descriptors and building .yace files are provided
in the examples folder.

## RPI procedure
Outline/example of the procedure is given in RPI_procedure.png

# FitSNAP usage
Fitsnap will need the generalized wigner symbols to plug into LAMMPS. 
To use this feature it is recommendend, but not required, that the
wigner coefficient pickle libraries be generated first in serial. This can
be done by running:
<pre><code>
python gen_lib.py generate
</code></pre>

The sym_ACE_settings.py file sets certain variables for the package, such
as the package name and the maximum l values per rank of generalized Wigner
symbols. Users that wish to use very large l values in their descriptors
or tensor product of spherical harmonics may need to adjust values in this
file. For optimal speed of the library, it is recommended that the default
settings be used.

## Citations:
Goff, J., Sievers, C., Wood, M. Thompson, A., "Permutation-adapted atomic cluster expansion descriptors" (in preparation)
GAP group. "Gap–Groups." Algorithms, and Programming, version 4.10 (2008).
