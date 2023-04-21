import numpy as np


def munlLinds_per_rank(rank):
    n_inters = 0
    if rank == 1:
        ln = 4
    elif rank == 2:
        ln = 1
        ln += (rank*3)
    else:
        ln = 1 
        ln += (rank *3)
        ln += (rank - 2)
        n_inters = rank - 2
    return ln,n_inters

def build_str(munlL_flat):
    munl_per_rank = {}
    L_per_rank = {}
    munlL_per_rank = {}
    for rank in range(1,8):
        full,inters = munlLinds_per_rank(rank)
        munlL_per_rank[rank] = full
        munl_per_rank[rank] = full-inters
        L_per_rank[rank] = inters

    rank_per_munl = {val:key for key,val in munl_per_rank.items()}
    rank_per_L = {val:key for key,val in L_per_rank.items()}
    rank_per_munlL = {val:key for key,val in munlL_per_rank.items()}


    if len(munlL_flat) in [4,7]:
        if len(munlL_flat) == 4:
            mu0 = munlL_flat[0]
            mu = munlL_flat[1]
            n = munlL_flat[2]
            l = munlL_flat[3]
            nu = '%d_%d,%d,%d_' % (mu0,mu,n,l)
        else:
            rank = rank_per_munlL[len(munlL_flat)]
            mu0 = munlL_flat[0]
            munl_tup = np.array(munlL_flat[1:])
            munl_tup = munl_tup.reshape(3,rank)
            print (munl_tup)
            munl_tup = tuple([tuple(v) for v in munl_tup])
            vecstrlst = ['%d']*rank
            vecstr = ','.join(b for b in vecstrlst)
            mustr = vecstr % munl_tup[0]
            nstr = vecstr % munl_tup[1]
            lstr = vecstr % munl_tup[2]
            nu = '%d_%s,%s,%s_' % (mu0,mustr,nstr,lstr)

    else:
        rank = rank_per_munlL[len(munlL_flat)]
        mu0 = munlL_flat[0]
        leninters = L_per_rank[rank]
        munl_tup = np.array(munlL_flat[1:-leninters])
        munl_tup = munl_tup.reshape(3,rank)
        munl_tup = tuple([tuple(v) for v in munl_tup])
        vecstrlst = ['%d']*rank
        Lstrlst = ['%d']*leninters
        vecstr = ','.join(b for b in vecstrlst)
        Lvecstr = '-'.join(b for b in Lstrlst)
        mustr = vecstr % munl_tup[0]
        nstr = vecstr % munl_tup[1]
        lstr = vecstr % munl_tup[2]
        Lstr = Lvecstr % tuple([i for i in munlL_flat[-leninters:]])
        nu = '%d_%s,%s,%s_%s' % (mu0,mustr,nstr,lstr,Lstr)
        
    return nu

def process_acepot(f,elems):

    nzs = {}
    nzcs = {}
    nzlst = []
    with open(f,'r') as readin:
        lines = readin.readlines()
        elem_lines = [ [line for line in lines if '%s\n' % elem in line] for elem in elems]
        elem_lines = [item for sublist in elem_lines for item in sublist]
        elem_lines_itr = iter([lines.index(eline) for eline in elem_lines])
        for ind, elem in enumerate(elems):
            if ind == 0:
                this_bound = next(elem_lines_itr)
            try:
                next_bound = next(elem_lines_itr)
            except StopIteration:
                if ind == len(elems) - 1:
                    next_bound = len(lines) -2
            for line in lines[this_bound + 1:next_bound]:
                ltmp = line.split('#  B[')
                c = float(ltmp[0])
                bstr = ltmp[1]
                bstr = bstr.replace(']','')
                bstr = bstr.split()
                nzlst.append(c)
                if bstr[0] == '0':
                    munlLs = [ind, 0]
                else:
                    #bstr = bstr.strip('[')
                    #bstr = bstr.strip(']')
                    munlLs = [int(k.split(',')[0]) for k in bstr[1:] ]
                    munlLs[0] = ind
                #strb = ','.join(str(b) for b in munlLs)
                if bstr[0] == '0':
                    strb = '%d_0' % ind
                else:
                    strb = build_str(munlLs)
                nzcs[strb] = c
            this_bound = next_bound
    return nzcs
