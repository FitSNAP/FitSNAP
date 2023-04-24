from ase.data import *
import numpy as np
import itertools

#SEE BOTTOM OF SCRIPT FOR EXAMPLE


# ionic radii from mendeleev (ionic radii in crystals from slater 1964) in angstroms
ionic_radii =  {'H': 0.25, 'He': 1.2, 'Li': 1.45, 'Be': 1.05, 'B': 0.85, 'C': 0.7, 'N': 0.65, 'O': 0.6, 'F': 0.5, 'Ne': 1.6, 'Na': 1.8, 'Mg': 1.5, 'Al': 1.25, 'Si': 1.1, 'P': 1.0, 'S': 1.0, 'Cl': 1.0, 'Ar': 0.71, 'K': 2.2, 'Ca': 1.8, 'Sc': 1.6, 'Ti': 1.4, 'V': 1.35, 'Cr': 1.4, 'Mn': 1.4, 'Fe': 1.4, 'Co': 1.35, 'Ni': 1.35, 'Cu': 1.35, 'Zn': 1.35, 'Ga': 1.3, 'Ge': 1.25, 'As': 1.15, 'Se': 1.15, 'Br': 1.15, 'Kr': np.nan, 'Rb': 2.35, 'Sr': 2.0, 'Y': 1.8, 'Zr': 1.55, 'Nb': 1.45, 'Mo': 1.45, 'Tc': 1.35, 'Ru': 1.3, 'Rh': 1.35, 'Pd': 1.4, 'Ag': 1.6, 'Cd': 1.55, 'In': 1.55, 'Sn': 1.45, 'Sb': 1.45, 'Te': 1.4, 'I': 1.4, 'Xe': np.nan, 'Cs': 2.6, 'Ba': 2.15, 'La': 1.95, 'Ce': 1.85, 'Pr': 1.85, 'Nd': 1.85, 'Pm': 1.85, 'Sm': 1.85, 'Eu': 1.85, 'Gd': 1.8, 'Tb': 1.75, 'Dy': 1.75, 'Ho': 1.75, 'Er': 1.75, 'Tm': 1.75, 'Yb': 1.75, 'Lu': 1.75, 'Hf': 1.55, 'Ta': 1.45, 'W': 1.35, 'Re': 1.35, 'Os': 1.3, 'Ir': 1.35, 'Pt': 1.35, 'Au': 1.35, 'Hg': 1.5, 'Tl': 1.9, 'Pb': 1.8, 'Bi': 1.6, 'Po': 1.9, 'At': np.nan, 'Rn': np.nan, 'Fr': np.nan, 'Ra': 2.15, 'Ac': 1.95, 'Th': 1.8, 'Pa': 1.8, 'U': 1.75, 'Np': 1.75, 'Pu': 1.75, 'Am': 1.75, 'Cm': np.nan, 'Bk': np.nan, 'Cf': np.nan, 'Es': np.nan, 'Fm': np.nan, 'Md': np.nan, 'No': np.nan, 'Lr': np.nan, 'Rf': np.nan, 'Db': np.nan, 'Sg': np.nan, 'Bh': np.nan, 'Hs': np.nan, 'Mt': np.nan, 'Ds': np.nan, 'Rg': np.nan, 'Cn': np.nan, 'Nh': np.nan, 'Fl': np.nan, 'Mc': np.nan, 'Lv': np.nan, 'Ts': np.nan, 'Og': np.nan}

metal_list = ['Li','Be','Na','Mg','K','Ca',
'Sc','Ti','V','Cr','Mn','Fe','Co','Ni','Cu','Zn',
'Rb','Sr','Y','Zr','Nb','Mo','Tc','Ru','Rh','Pd',
'Ag','Cd','Cs','Ba','Lu','Hf','Ta','W','Re','Os','Ir',
'Pt','Au','Hg','Fr','Ha',
'La','Ce','Pr','Nd','Pm','Sm','Eu','Gd','Tb','Dy','Ho','Er','Tb','Yb',
'Ac','Th','Pa','U','Np','Pu','Am']

def default_rc(elms,nshell=2.2,use_vdw=False,metal_max=True):
    # nshell (float): number of bond shells to include in radial cutoff
    # use_vdw (logical): flag to use the vdw bond lengths ( a decent estimate of maximum cutoff needed for given number of shells)
    # metal_max (logical): flag to use ionic/crystal radii rather than a hybrid bond length (default is true)
    elms = sorted(elms)
    elm1,elm2 = tuple(elms)
    atnum1 = atomic_numbers[elm1]
    covr1 = covalent_radii[atnum1]
    vdwr1 = vdw_radii[atnum1]
    ionr1 = ionic_radii[elm1]
    if np.isnan(vdwr1):
        print ('NOTE: using dummy VDW radius of 2* covalent radius for %s' % elm1)
        vdwr1 = 2*covr1
    atnum2 = atomic_numbers[elm2]
    covr2 = covalent_radii[atnum2]
    vdwr2 = vdw_radii[atnum2]
    ionr2 = ionic_radii[elm2]
    if np.isnan(vdwr2):
        print ('NOTE: using dummy VDW radius of 2* covalent radius for %s' %elm2)
        vdwr2 = 2*covr2
    #minrc = min([ionr1,ionr2])
    #maxrc = max([vdwr1,vdwr2])
    minbond = ionr1 + ionr2
    if metal_max:
        if elm1 not in metal_list and elm2 not in metal_list:
            maxbond = vdwr1 + vdwr2
        elif elm1 in metal_list and elm2 not in metal_list:
            maxbond = ionr1 + vdwr2
            minbond = (ionr1 + ionr2)*0.8
        elif elm1 in metal_list and elm2 in metal_list:
            maxbond = ionr1 + ionr2
            minbond = (ionr1 + ionr2)*0.8
        else:
            maxbond = ionr1 + ionr2
    else:
            maxbond = vdwr1 + vdwr2
    midbond = (maxbond +minbond)/2
    # by default, return the ionic bond length * number of bonds for minimum
    returnmin = minbond
    if use_vdw:
        #return vdw bond length if requested
        returnmax = maxbond
    elif not use_vdw:
        #return hybrid vdw/ionic bonds by default
        returnmax = midbond
    return round(returnmin,3),round(returnmax,3)

def get_default_settings(elems,nshell=1.0,return_range=True,apply_shift=False,metal_max=True,inner_fraction=0.25):
    # elems (list): sorted list of element symbols
    # nshell (float): number of bond shells to include in estimation of radial cutoff ranges
    # return_range: logical to return a dictionary of possible ranges of rcut for different bond types (e.g. for dakota)
    # apply_shift: logical to shift disparate rcs due to highly mixed bond types (e.g. in a W-H system)
    # metal_max (logical): flag to use ionic/crystal radii rather than a hybrid bond length for estimation of radial cutoff (default is true)
    # inner_fraction (float): factor to reduce smallest bond length by to get inner cutoff radius. 

    def reference_printer(bonds,inners, rc):
        # default inner = 0.25 * ionic bond length
        # default outer = 0.25/0.8 * ionic bond length
        elemsi = sorted(list(set(np.array(bonds).flatten().tolist())))
        print (elemsi)
        elem_inds = {e:ii + 1 for ii,e in enumerate(elemsi)}
        ubonds = [bond for bond in bonds if tuple(sorted(bond)) == bond]
        inner_per_u = {b:i for b,i in zip(bonds,inners)}
        outer_per_u = {b:(i*4*(1.)) for b,i in zip(bonds,inners)}
        zbl_str_i = 'zbl %f %f'
        zbl_lst = []
        for ubond in ubonds:
            zbl_lst.append(zbl_str_i % (inner_per_u[ubond],outer_per_u[ubond]))
        zbl_str = ' '.join([zb for zb in zbl_lst])
        pair_coeff_i = 'pair_coeff%d = %d %d zbl %d %d'
        pair_coeff_ii = 'pair_coeff%d = %d %d zbl %d    %d %d'
        pair_coeff_strs = []

        for ib, ubond in enumerate(ubonds):
            pind = ib + 2
            atnum1 = atomic_numbers[ubond[0]]
            atnum2 = atomic_numbers[ubond[1]]
            if len(ubonds) == 1:
                pair_coeff_strs.append(pair_coeff_i % (pind,elem_inds[ubond[0]],elem_inds[ubond[1]],atnum1,atnum2))
            elif len(ubonds) > 1:
                pair_coeff_strs.append(pair_coeff_ii % (pind,elem_inds[ubond[0]],elem_inds[ubond[1]],pind-1, atnum1,atnum2))

        print ('Suggested starting point for reference:\n')
        print ('pair_style = hybrid/overlay zero %2.6f %s' % (rc + 0.01, zbl_str))
        print ('pair_coeff1 = * * zero')
        for pair_coeff_str in pair_coeff_strs:
            print(pair_coeff_str)
        
        print ('\n')


    assert tuple(elems) == tuple(sorted(elems)), " elements must be listed alphabetically"
    bonds =[bp for bp in itertools.product(elems,elems)]
    rc_range = {bp:None for bp in bonds}
    rin_def = {bp:None for bp in bonds}
    rc_def = {bp:None for bp in bonds}
    for bond in bonds:
        rcmini,rcmaxi = default_rc(bond)
        defaultri = (rcmaxi+rcmini)/1.8
        defaultrcinner = (rcmini)*0.25 # 1/4 of shortest ionic bond length
        rc_range[bond] = [rcmini,rcmaxi]
        rc_def[bond] = defaultri*nshell
        rin_def[bond] = defaultrcinner
    #shift = ( max(rc_def.values()) - min(rc_def.values())  )/2
    shift = np.std(list(rc_def.values()))
    #if apply_shift:
    #print (rc_def)
    for bond in bonds:
        if rc_def[bond] != max(rc_def.values()):
            if apply_shift:
                rc_def[bond] = (rc_def[bond] + shift)#*nshell
            else:
                rc_def[bond] = rc_def[bond]#*nshell
    #print (rc_def)
    default_lmbs = [i*0.3 for i in list(rc_def.values())]
    rc_def_lst = ['%1.3f']* len(bonds)
    rc_def_str = 'rcutfac = ' + '  '.join(b for b in rc_def_lst) % tuple(list(rc_def.values()))
    lmb_def_lst = ['%1.3f']* len(bonds)
    lmb_def_str = 'lambda = ' + '  '.join(b for b in lmb_def_lst) % tuple(default_lmbs)
    rcin_def_lst = ['%1.3f']* len(bonds)
    rcin_def_str = 'rcinner = ' + '  '.join(b for b in rcin_def_lst) % tuple(list(rin_def.values()))
    print (rc_range)
    reference_printer(bonds,list(rin_def.values()), max(list(rc_def.values())))
    return rc_range,rc_def_str,lmb_def_str,rcin_def_str


#uncomment for different examples

#elems = ['Ta']
#elems = ['H','W']
elems = ['H','O']
#elems = ['W','Zr','C']
elems = sorted(elems) # sort element types alphabetically
rc_range,rc_default,lmb_default,rcin_default = get_default_settings(elems,nshell=2.2,return_range=True,apply_shift=False)
print ('recommended starting hyperparameters\n')
print (rc_default)
print (lmb_default)
print (rcin_default)
