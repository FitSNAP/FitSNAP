from fitsnap3lib.lib.sym_ACE.pa_lib import *
import json

def pa_labels_raw(rank,nmax,lmax,mumax,lmin=1):
    nadd = 4
    if rank > 3:
        lmax_strs = generate_l_LR(range(lmin,lmax+1),rank,L_R=0,M_R=0,use_permutations=False)
        lvecs = [tuple([int(k) for k in lmax_str.split(',')]) for lmax_str in lmax_strs]
        nvecs = [i for i in itertools.combinations_with_replacement(range(0,nmax),rank)]
        nvecs_gen = [i for i in itertools.combinations_with_replacement(range(0,nmax+nadd),rank)]
        muvecs = [i for i in itertools.combinations_with_replacement(range(mumax),rank)]
        reduced_nvecs=get_mapped_subset(nvecs_gen)

        fs_labs = []
        all_nl = []

        all_PA_tabulated = []
        PA_per_nlblock = {}
        for nin in reduced_nvecs:
            for lin in lvecs:
                max_labs,all_labs,labels_per_block,original_spans = tree_labels(nin,lin)
                combined_labs = combine_blocks(labels_per_block,lin,original_spans)
                nl = (nin,lin)
                lspan_perm = list(original_spans.keys())[0]
                parity_span = [p for p in original_spans[lspan_perm] if np.sum(lspan_perm[:2] + p[2][:1]) %2 == 0 and np.sum(lspan_perm[2:4] + p[2][1:2]) %2 == 0]
                PA_labels = apply_ladder_relationships(lin, nin, combined_labs, parity_span, parity_span_labs = max_labs, full_span=original_spans[lspan_perm])
                mustrlst = ['%d']*rank
                nstrlst = ['%d']*rank
                lstrlst = ['%d']*rank
                Lstrlst = ['%d']*(rank-2)
                nl_simple_labs = []
                nlstr = ','.join(nstrlst) % tuple(nin) + '_' + ','.join(lstrlst) % tuple(lin)
                for lab in PA_labels:
                    mu0,mu,n,l,L = get_mu_n_l(lab,return_L=True)
                    if L != None:
                        nlL = (tuple(n),tuple(l),L)
                    else:
                        nlL = (tuple(n),tuple(l),tuple([]))
                    simple_str = ','.join(nstrlst) % tuple(n) + '_' + ','.join(lstrlst) % tuple(l) + '_' + ','.join(Lstrlst) % L
                    all_PA_tabulated.append(simple_str)
                    nl_simple_labs.append(simple_str)
                PA_per_nlblock[nlstr] = nl_simple_labs

        #dct = {'labels':all_PA_tabulated}
        dct = {'labels':PA_per_nlblock}
        with open('all_labels_n%d_l%d_r%d.json' % (nmax,lmax,rank),'w') as writejson:
            json.dump(dct,writejson, sort_keys=False, indent=2)
        with open('all_labels_n%d_l%d_r%d.json' % (nmax,lmax,rank),'r') as readjson:
            data = json.load(readjson)

        all_lammps_labs = []
        all_not_compat = []
        possible_mus = list(range(mumax))
        for muvec in muvecs:
            muvec = tuple(muvec)
            for nlblockstr in list(PA_per_nlblock.keys()):
                nstr,lstr = tuple(nlblockstr.split('_'))
                nvec = tuple([int(k) + 1 for k in nstr.split(',')])
                nvec_raw = tuple([int(k) for k in nstr.split(',')])
                lvec = tuple([int(k) for k in lstr.split(',')])
                #nus = from_tabulated((0,0,0,0),(1,1,1,1),(4,4,4,4),allowed_mus = possible_mus, tabulated_all = data)
                if nvec_raw in nvecs:
                    nus = from_tabulated(muvec,nvec,lvec,allowed_mus = possible_mus, tabulated_all = data)
                    lammps_ready,not_compatible = lammps_remap(nus,rank=rank,allowed_mus=possible_mus)
                    all_lammps_labs.extend(lammps_ready)
                    all_not_compat.extend(not_compatible)

                #print ('raw PA-RPI',nus)
                #print ('lammps ready PA-RPI',lammps_ready)
                #print ('not compatible with lammps (PA-RPI with a nu vector that cannot be reused)',not_compatible)
        return all_lammps_labs,all_not_compat
    else:
        return generate_nl(rank,nmax,lmax,mumax,lmin), []
"""
print ('all_final')

PA_lammps, not_compat = pa_labels_raw(rank=4,nmax=2,lmax=2,mumax=1,lmin=1)
for lab in PA_lammps:
    print (lab)

print('not compat',not_compat)

print (len(PA_lammps),len(not_compat))
"""
