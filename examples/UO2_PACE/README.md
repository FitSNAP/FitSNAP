# Multielement ACE input

# ACE section 

Note that multielement hyperparameters are assigned per bond.
Your types must be sorted alphabetically: (H,O), etc
Bond types are assigned using a native python combination tool:

<pre><code>
```
In [1]: import itertools
   ...: types = ['H' , 'O']
   ...: bonds =[ bond for bond in itertools.product(types,types) ]

In [2]: bonds
Out[2]: [('H', 'H'), ('H', 'O'), ('O', 'H'), ('O', 'O')]
```
</code></pre>

You may specify a radial cutoff per bond type:

<pre><code>
```
rcutfac = 5.0 5.5 5.5 6.0
```
</code></pre>
but note that permutations of bonds should have the same  
rcutfac (as well as any other hyperparameter). For example,
 ('H', 'O') and ('O', 'H') both have an rcutfac of 5.5. This 
will be simplified in a future update.

<pre><code>
```
[ACE]
numTypes = 2
rcutfac = 5.0 5.5 5.5 6.0
lambda = 3.3 3.3 3.3 3.3
rcinner = 0.1 0.1 0.1 0.1 
drcinner = 0.01 0.01 0.01 0.01
ranks = 1 2 3 4
lmax =  1 2 2 1
nmax = 16 2 2 1
mumax = 2
nmaxbase = 16
type = O U
bzeroflag = 0
```
</code></pre>

The only new parameters from the single element examples
are the rcinner and drcinner, which correspond to the
radius for hard-core repulsion cutoffs (use in place of ZBL)
and the gradient step for the hard core region, respectively.
This acts as a reference potential so that you dont have to 
train extreme nuclear repulsion. To disable, set the rcinner
parameters for all bonds to 0.
