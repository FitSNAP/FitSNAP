from yamlpace_tools.potential import *


# potential parameters. NOTE these MUST match the FitSNAP input
reference_ens = [0.,0.]
rcutfac = [5.0, 5.5, 5.5, 6.0]
lmbda = [3.3, 3.3, 3.3, 3.3]
rcinner = [0.1, 0.1, 0.1, 0.1]
drcinner = [0.01, 0.01, 0.01, 0.01]
elements=["O","U"]
ranks = [1, 2, 3, 4]
lmax =  [1, 2, 2, 1]
nmax = [16, 2, 2, 1]
nradbase=max(nmax)


Apot = AcePot(elements,reference_ens,ranks,nmax,lmax,nradbase,rcutfac,lmbda,rcinner,drcinner,RPI_heuristic='root_SO3_span')
# read the potential file to get expansion coefficients
betas = Apot.read_acecoeff('UO2_pot')
# set the expansion coefficients and multiply them by coupling coefficients to get c_tilde coefficients
Apot.set_betas(betas)
Apot.set_funcs()
# write a new potential file
Apot.write_pot('UO2_potential')
coeffs = flatten( [b.values() for b in betas.values()])
print (coeffs)
np.save('coefficient_array.npy',np.array(coeffs))
