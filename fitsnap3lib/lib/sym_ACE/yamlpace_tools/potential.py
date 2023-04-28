import itertools
from fitsnap3lib.lib.sym_ACE.wigner_couple import *
from fitsnap3lib.lib.sym_ACE.rpi_lib import *
from fitsnap3lib.lib.sym_ACE.yamlpace_tools.acecoeff_tools import *
import json


class AcePot():
    def __init__(self,
            elements,
            reference_ens,
            ranks,
            nmax,
            lmax,
            nradbase,
            rcut,
            lmbda,
            rcutinner=0.0,
            drcutinner=0.01,
            lmin = 1,
            RPI_heuristic='root_SO3_span',
            **kwargs):
        if kwargs != None:
            self.__dict__.update(kwargs)
      
        self.global_ccs = generate_ccs()
        self.E0 = reference_ens
        self.ranks =ranks
        self.elements = elements
        self.betas = None
        self.nus = None
        self.deltaSplineBins=0.001
        self.global_ndensity=1
        self.global_FSparams=[1.0, 1.0]
        self.global_rhocut = 100000
        self.global_drhocut = 250
        #assert the same nmax,lmax,nradbase (e.g. same basis) for each bond type
        self.radbasetype = 'ChebExpCos'
        self.global_nmax=nmax
        self.global_lmax=lmax
        assert len(nmax) == len(lmax),'nmax and lmax arrays must be same size'
        self.global_nradbase=nradbase
        if type(rcut) != dict and type(rcut) != list:
            self.global_rcut = rcut
            self.global_lmbda = lmbda
            self.global_rcutinner = rcutinner
            self.global_drcutinner = drcutinner
            self.global_lmin = lmin
        else:
            self.rcut = rcut
            self.lmbda = lmbda
            self.rcutinner = rcutinner
            self.drcutinner = drcutinner
            self.lmin = lmin
        self.RPI_heuristic = RPI_heuristic
        self.set_embeddings()
        self.set_bonds()
        self.set_bond_base()

        lmax_dict = {rank:lv for rank,lv in zip(self.ranks,self.global_lmax)}
        try:
            lmin_dict = {rank:lv for rank,lv in zip(self.ranks,self.lmin)}
        except AttributeError:
            lmin_dict = {rank:lv for rank,lv in zip(self.ranks,self.global_lmin*len(self.ranks))}
        nradmax_dict = {rank:nv for rank,nv in zip(self.ranks,self.global_nmax)}
        mumax_dict={rank:len(self.elements) for rank in self.ranks}
        if self.RPI_heuristic == 'lexicographic':
            nulst_1 = [generate_nl(rank,nradmax_dict[rank],lmax_dict[rank],mumax_dict[rank]) for rank in self.ranks]
        else:
            nulst_1 = [descriptor_labels_YSG(rank,nradmax_dict[rank],lmax_dict[rank],mumax_dict[rank],lmin_dict[rank]) for rank in self.ranks]
        nus_unsort = [item for sublist in nulst_1 for item in sublist]
        nus = nus_unsort.copy()
        mu0s = []
        mus =[]
        ns = []
        ls = []
        for nu in nus_unsort:
            mu0ii,muii,nii,lii = get_mu_n_l(nu)
            mu0s.append(mu0ii)
            mus.append(tuple(muii))
            ns.append(tuple(nii))
            ls.append(tuple(lii))
        nus.sort(key = lambda x : mus[nus_unsort.index(x)],reverse = False)
        nus.sort(key = lambda x : ns[nus_unsort.index(x)],reverse = False)
        nus.sort(key = lambda x : ls[nus_unsort.index(x)],reverse = False)
        nus.sort(key = lambda x : mu0s[nus_unsort.index(x)],reverse = False)
        nus.sort(key = lambda x : len(x),reverse = False)
        nus.sort(key = lambda x : mu0s[nus_unsort.index(x)],reverse = False)
        self.nus = nus
        self.set_funcs(nus)

        return None


    def set_embeddings(self,npoti='FinnisSinclair',FSparams=[1.0,1.0]):#default for linear models in lammps PACE
        #embeddings =dict()#OrderedDict() #{ind:None for ind in range(len(self.elements))}
        embeddings ={ind:None for ind in range(len(self.elements))}
        for elemind in range(len(self.elements)):
            embeddings[elemind] = {'ndensity':self.global_ndensity,
                                   'FS_parameters':FSparams,'npoti':npoti, 
                                   'rho_core_cutoff':self.global_rhocut, 
                                   'drho_core_cutoff':self.global_drhocut}
        self.embeddings = embeddings

    def set_bonds(self):
        bondinds=range(len(self.elements))
        bond_lsts = [list(b) for b in itertools.product(bondinds,bondinds)]
        self.bondlsts = bond_lsts

    def set_bond_base(self):
        bondstrs = ['[%d, %d]' %(b[0],b[1]) for b in self.bondlsts]
        bonds = {bondstr:None for bondstr in bondstrs}

        #radial basis function expansion coefficients
        #saved in n,l,k shape
        # defaults to orthogonal delta function [g(n,k)] basis of drautz 2019   
        try:
            nradmax = max(self.global_nmax[:])
        except ValueError:
            nradmax = max(self.global_nmax)
        lmax= max(self.global_lmax)
        nradbase = self.global_nradbase
        crad = np.zeros((nradmax,lmax+1,nradbase),dtype=int)
        for n in range(nradmax):
            for l in range(lmax+1):
                crad[n][l] = np.array([1 if k==n else 0 for k in range(nradbase)]) 

        cnew = np.zeros((nradbase,nradmax,lmax+1))
        for n in range(1,nradmax+1):
            for l in range(lmax+1):
                for k in range(1,nradbase+1):
                    cnew[k-1][n-1][l] = crad[n-1][l][k-1]

        for bondind,bondlst in enumerate(self.bondlsts):
            bstr = '[%d, %d]' %(bondlst[0],bondlst[1])
            
            try:
                bonds[bstr] = {'nradmax':nradmax, 
                               'lmax':max(self.global_lmax), 
                               'nradbasemax':self.global_nradbase,
                               'radbasename':self.radbasetype,
                               'radparameters':[self.global_lmbda], 
                               'radcoefficients':crad.tolist(), 
                               'prehc':0, 
                               'lambdahc':self.global_lmbda,
                               'rcut':self.global_rcut, 
                               'dcut':0.01, 
                               'rcut_in':self.global_rcutinner, 
                               'dcut_in':self.global_drcutinner, 
                               'inner_cutoff_type':'distance'}
            except AttributeError:
                if type(self.rcut) == dict:
                    bonds[bstr] = {'nradmax':nradmax, 
                                   'lmax':max(self.global_lmax), 
                                   'nradbasemax':self.global_nradbase,
                                   'radbasename':self.radbasetype,
                                   'radparameters':[self.lmbda[bstr]], 
                                   'radcoefficients':crad.tolist(), 
                                   'prehc':0, 
                                   'lambdahc':self.lmbda[bstr],
                                   'rcut':self.rcut[bstr], 
                                   'dcut':0.01, 
                                   'rcut_in':self.rcutinner[bstr], 
                                   'dcut_in':self.drcutinner[bstr], 
                                   'inner_cutoff_type':'distance'}
                elif type(self.rcut) == list:
                    bonds[bstr] = {'nradmax':nradmax, 
                                   'lmax':max(self.global_lmax), 
                                   'nradbasemax':self.global_nradbase,
                                   'radbasename':self.radbasetype,
                                   'radparameters':[self.lmbda[bondind]], 
                                   'radcoefficients':crad.tolist(), 
                                   'prehc':0, 
                                   'lambdahc':self.lmbda[bondind],
                                   'rcut':self.rcut[bondind], 
                                   'dcut':0.01, 
                                   'rcut_in':self.rcutinner[bondind], 
                                   'dcut_in':self.drcutinner[bondind], 
                                   'inner_cutoff_type':'distance'}
          
        self.bonds = bonds

    def set_funcs(self,nulst=None,muflg=True,print_0s=True):
        if nulst == None:
            if self.nus != None:
                nulst = self.nus.copy()
            else:
                raise AttributeError("No list of descriptors found/specified")
          
        muflg = True
        permu0 = {b:[] for b in range(len(self.elements))}
        permunu = {b:[] for b in range(len(self.elements))}
        if self.betas != None:
            betas = self.betas
        else:
            #betas = {ind:{nu:1.0 for nu in nulst} for ind in range(len(self.elements))}
            betas = {ind:{} for ind in range(len(self.elements))}
            for nu in nulst:
                mu0,mu,n,l,L = get_mu_n_l(nu,return_L=True)
                betas[mu0][nu] = 1.0
        for nu in nulst:
            mu0,mu,n,l,L = get_mu_n_l(nu,return_L=True)
            rank = get_mu_nu_rank(nu)

            llst = ['%d']*rank
            #print (nu,l,oldfmt,muflg)
            lstr = ','.join(b for b in llst) % tuple(l)
            if L != None:
                ccs = self.global_ccs[rank][lstr][tuple(L)]
            elif L == None:
                try:
                    ccs = self.global_ccs[rank][lstr][()]
                except KeyError:
                    ccs = self.global_ccs[rank][lstr]
            ms = list(ccs.keys())
            mslsts = [[int(k) for k in m.split(',')] for m in ms]
            msflat= [item for sublist in mslsts for item in sublist]
            if print_0s or betas[mu0][nu] != 0.:
                ccoeffs =  list ( np.array(list(ccs.values())) * betas[mu0][nu] )
                permu0[mu0].append({'mu0':mu0,
                                    'rank':rank,
                                    'ndensity':self.global_ndensity,
                                    'num_ms_combs':len(ms),
                                    'mus':mu, 
                                    'ns':n,
                                    'ls':l,
                                    'ms_combs':msflat, 
                                    'ctildes':ccoeffs})
                permunu[mu0].append(nu)
            elif betas[mu0][nu] == 0. and not print_0s:
                print ('Not printing descriptor: %s, coefficient is 0' % nu)
            
        #for b in range(len(self.elements)):
        #   for i in permunu[b]:
        #       print (b,i)
        #for b in range(len(self.elements)):
        #   print (b,len(permu0[b]))
        self.funcs = permu0
        self.permunu = permunu

    def set_betas(self,betas,has_zeros=False):
        if type(betas) != dict:
            if not has_zeros:
                assert len(betas) == len(self.nus), "list of betas must be the same size as list of descriptors (0th order coefficient should NOT be included in this list"
            elif has_zeros:
                with_nu_inds = len(self.nus) + len(self.elements)
                base_N_nu_per_ind = int(len(self.nus)/len(self.elements))
                e0inds =[]
                for i in range(len(self.elements)):
                    e0ind = (i*base_N_nu_per_ind)  + i 
                    e0inds.append(e0ind)
                e0s = [betas[e0ind] for e0ind in e0inds]
                self.E0 = e0s
                betas = [b for i,b in enumerate(betas) if i not in e0inds]
        
            betas_dict = {ind:{} for ind in range(len(self.elements))}
            for nu,beta in zip(self.nus,betas):
                mu0,mu,n,l,L = get_mu_n_l(nu,return_L=True)
                betas_dict[mu0][nu] = beta
            self.betas = betas_dict
        elif type(betas) == dict:
            self.betas = betas

    def read_acecoeff(self,name,remove_0s=False):
        f = '%s.acecoeff' % name
        coeff_dict = process_acepot(f,self.elements)
        e0s = []
        beta_dict = {mu0: {} for mu0 in range(len(self.elements))}
        remove_keys = {mu0: [] for mu0 in range(len(self.elements))}
        for ind,element in enumerate(self.elements):
            for key  in coeff_dict.keys():
                if key == '%d_0' % ind:
                    e0s.append(coeff_dict[key])
                    remove_keys[ind].append(key)
                else:
                    mu0,mu,n,l = get_mu_n_l(key)
                    beta_dict[mu0][key] = coeff_dict[key]
      
        for ind,element in enumerate(self.elements):
            for key in remove_keys[ind]:
                try:
                    del beta_dict[ind][key]
                except KeyError:
                    pass

        self.E0 = e0s
        return beta_dict
          
    def write_pot(self,name):
        srt_nus = []
        with open('%s.yace'%name,'w') as writeout:
            e0lst = ['%f']*len(self.elements)
            e0str = ', '.join(b for b in e0lst) % tuple(self.E0)
            elemlst =['%s']*len(self.elements)
            elemstr = ', '.join(b for b in elemlst) % tuple(self.elements)
            writeout.write('elements: [%s] \n' % elemstr)
            writeout.write('E0: [%s] \n' % e0str)
            writeout.write('deltaSplineBins: %f \n' % self.deltaSplineBins)
            writeout.write('embeddings:\n')
            for mu0, embed in self.embeddings.items():
                writeout.write('  %d: ' % mu0)
                ystr = json.dumps(embed) + '\n'
                ystr = ystr.replace('"','')
                writeout.write(ystr)
            writeout.write('bonds:\n')
            bondstrs=['[%d, %d]' %(b[0],b[1]) for b in self.bondlsts]
            for bondstr in bondstrs:
                writeout.write('  %s: ' % bondstr)
                bstr = json.dumps(self.bonds[bondstr]) + '\n'
                bstr = bstr.replace('"','')
                writeout.write(bstr)
            writeout.write('functions:\n')
            for mu0 in range(len(self.elements)):
                writeout.write('  %d:\n'%(mu0))
                mufuncs = self.funcs[mu0]
                for mufunc in mufuncs:
                    mufuncstr = '    - ' +json.dumps(mufunc) + '\n'
                    mufuncstr = mufuncstr.replace('"','')
                    writeout.write(mufuncstr)
