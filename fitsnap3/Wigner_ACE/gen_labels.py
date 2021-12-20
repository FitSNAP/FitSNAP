import copy
import itertools

def ind_vec(lrng,size):
     uniques = []
     combs = itertools.combinations_with_replacement(lrng,size)
     for comb in combs:
         perms = itertools.permutations(comb)
         for p in perms:
             pstr = ','.join(str(k) for k in p)
             if pstr not in uniques:
                 uniques.append(pstr)
     return uniques

def get_nu_rank(nu):
    nu_splt = [int(k) for k in nu.split(',')]
    if len(nu_splt) == 3:
        if nu_splt[1] ==0 and nu_splt[2] ==0:
            return 1
        elif nu_splt[1] !=0 or nu_splt[2] !=0:
            return 2
    elif len(nu_splt) >3:
        return int(len(nu_splt)/2)

def get_intermediates(l):
    try:
        l = l.split(',')
        l1 = int(l[0])
        l2 = int(l[1])
    except AttributeError:
        l1 = l[0]
        l2 = l[1]

    tris = [i for i in range(abs(l1-l2),l1+l2+1)]

    ints = [i for i in tris]
    return ints

#wrapper
def get_intermediates_w(l1,l2):
    l = [l1,l2]
    return get_intermediates(l)

def get_n_l(nu,**kwargs):
    try:
        rank = kwargs['rank']
    except KeyError:
        rank = get_nu_rank(nu)
    #print (nu,rank)
    if rank ==1:
        nusplt = [int(k) for k in nu.split(',')]
        n = [nusplt[0]]
        l = [0]
    elif rank ==2:
        nusplt = [int(k) for k in nu.split(',')]
        n = [nusplt[0],nusplt[1]]
        l = [nusplt[-1],nusplt[-1]] #same l for each rank A in a rank 2 invariant       a
    elif rank >2:
        nusplt = [int(k) for k in nu.split(',')]
        n = nusplt[:rank]
        l = nusplt[rank:]

    return n,l


#this function is used to generate all (reverse) lexicographically ordered \bold{nl} (aka \nu) vectors
#  that produce the ACE basis for scalar properties
def generate_nl(rank,nmax,lmax,enforce_dorder=True):
    # rank: int  - basis function rank to evaluate nl combinations for
    # nmax: int  - maximum value of the n quantum numbers in the nl vectors
    # lmax: int  - maximum value of the l quantum numbers in the nl vectors
    # enforce_dorder: logical  - flag to specify whether reverse ordering should be used NOTE this is necessary for rank 7 + descriptors!
    # RETURN: list of nl vectors in string format n1,n2,..n_k,l1,l2,..l_k 


    #nl vectors
    nl=[]
    if   rank ==1:
        for n in range(1,nmax+1):
            nl.append('%d,0,0'%n)
    elif rank ==2:
        for n1 in range(1,nmax+1):
            for n2 in range(1,nmax+1):
                for l in range(lmax+1):
                    x = [(l,n1),(l,n2)]
                    srt = sorted(x)
                    if x == srt:
                        nl.append('%d,%d,%d' %(n1,n2,l))
    elif rank ==3:
        for n1 in range(1,nmax+1):
            for n2 in range(1,nmax+1):
                for n3 in range(1,nmax+1):
                    for l1 in range(lmax+1):
                        for l2 in range(lmax+1):
                            for l3 in range(abs(l1-l2),l1+l2+1):
                                if (l1+l2+l3) % 2 == 0 and l3 <=lmax:
                                    x = [(l1,n1),(l2,n2),(l3,n3)]
                                    srt = sorted(x)
                                    if x == srt:
                                        stmp = "%d,%d,%d,%d,%d,%d" % (n1,n2,n3,l1,l2,l3)
                                        stmp_rev = "%d,%d,%d,%d,%d,%d" % (n3,n2,n1,l3,l2,l1)
                                        if enforce_dorder:
                                            if stmp_rev not in nl:
                                                nl.append(stmp_rev)
                                        else:
                                            if stmp not in nl:
                                                nl.append(stmp)
    elif rank ==4:
        myl4s=[]

        for l1 in range(lmax+1):
            for l2 in range(lmax+1):
                for l12 in get_intermediates('%d,%d' %(l1,l2)):
                    for l3 in range(lmax+1):
                        for l4 in get_intermediates('%d,%d' %(l12,l3)):
                            #enforce parity of all l_i and intermediates
                            if (l1+l2+l3+l4) %2==0 and l4 <= lmax and (l1 +l2 +l12) %2 ==0:
                                myl4s.append([l1,l2,l3,l4])

        for n1 in range(1,nmax+1):
            for n2 in range(1,nmax+1):
                for n3 in range(1,nmax+1):
                    for n4 in range(1,nmax+1):
                        for ind,l in enumerate(myl4s):
                            l1,l2,l3,l4 = l[0],l[1],l[2],l[3]
                            x = [(l1,n1),(l2,n2),(l3,n3),(l4,n4)]
                            srt = sorted(x)
                            if x == srt:
                                stmp = "%d,%d,%d,%d,%d,%d,%d,%d" % (n1,n2,n3,n4,l1,l2,l3,l4)
                                stmp_rev = "%d,%d,%d,%d,%d,%d,%d,%d" % (n4,n3,n2,n1,l4,l3,l2,l1)
                                if enforce_dorder:
                                    if stmp_rev not in nl:
                                        nl.append(stmp_rev)
                                else:
                                    if stmp not in nl:
                                        nl.append(stmp)
    elif rank ==5:
        myl5s = []
        for l1 in range(lmax+1):
            for l2 in range(lmax+1):
                for l12 in get_intermediates('%d,%d' % (l1,l2)):
                    for l3 in range(lmax+1):
                        for l123 in get_intermediates('%d,%d' % (l12,l3)):
                            for l4 in range(lmax+1):
                                for l5 in range(abs(l123-l4),l123+l4+1):
                                    #enforce parity of l_i
                                    if (l1+l2+l3+l4+l5) %2==0 and l5<=lmax:
                                        #enforce parity of intermediates
                                        if (l12 + l1+l2) %2 ==0 and (l12+ l3+l123 )%2 ==0:
                                            myl5s.append([l1,l2,l3,l4,l5])
                                                
        for n1 in range(1,nmax+1):
            for n2 in range(1,nmax+1):
                for n3 in range(1,nmax+1):
                    for n4 in range(1,nmax+1):
                        for n5 in range(1,nmax+1):
                            for ind,l in enumerate(myl5s):
                                l1,l2,l3,l4,l5 = l[0],l[1],l[2],l[3],l[4]
                                x = [(l1,n1),(l2,n2),(l3,n3),(l4,n4),(l5,n5)]
                                srt = sorted(x)
                                if x == srt:
                                    stmp = "%d,%d,%d,%d,%d,%d,%d,%d,%d,%d" % (n1,n2,n3,n4,n5,l1,l2,l3,l4,l5)
                                    stmp_rev = "%d,%d,%d,%d,%d,%d,%d,%d,%d,%d" % (n5,n4,n3,n2,n1,l5,l4,l3,l2,l1)
                                    if enforce_dorder:
                                        if stmp_rev not in nl:
                                            nl.append(stmp_rev)
                                    else:
                                        if stmp not in nl:
                                            nl.append(stmp)

    elif rank==6:
        myl6s = []
        for l1 in range(lmax+1):
            for l2 in range(lmax+1):
                for l12 in get_intermediates('%d,%d' % (l1,l2)):
                    for l3 in range(lmax+1):
                        for l123 in get_intermediates('%d,%d' % (l12,l3)):
                            for l4 in range(lmax+1):
                                for l1234 in get_intermediates('%d,%d' % (l123,l4)):
                                    for l5 in range(lmax+1):
                                        for l6 in range(abs(l1234-l5),l1234+l5+1):
                                            #enforce parity of all l_i
                                            if (l1+l2+l3+l4+l5+l6) %2==0 and l6<=lmax:
                                                # enforce parity of all intermediates
                                                if (l1234 +l4 +l123) %2 ==0 and (l123 +l3 +l12)%2==0 and (l12 +l1+l2) %2 ==0:
                                                    myl6s.append([l1,l2,l3,l4,l5,l6])
                                                    
        for n1 in range(1,nmax+1):
            for n2 in range(1,nmax+1):
                for n3 in range(1,nmax+1):
                    for n4 in range(1,nmax+1):
                        for n5 in range(1,nmax+1):
                            for n6 in range(1,nmax+1):
                                for ind,l in enumerate(myl6s):
                                    l1,l2,l3,l4,l5,l6 = l[0],l[1],l[2],l[3],l[4],l[5]
                                    x = [(l1,n1),(l2,n2),(l3,n3),(l4,n4),(l5,n5),(l6,n6)]
                                    srt = sorted(x)
                                    if x == srt:
                                        stmp = "%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d" % (n1,n2,n3,n4,n5,n6,l1,l2,l3,l4,l5,l6)
                                        stmp_rev = "%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d" % (n6,n5,n4,n3,n2,n1,l6,l5,l4,l3,l2,l1)
                                        if enforce_dorder:
                                            if stmp_rev not in nl:
                                                nl.append(stmp_rev)
                                        else:
                                            if stmp not in nl:
                                                nl.append(stmp)
    elif rank==7:
        myl7s = []
        for l1 in range(lmax+1):
            for l2 in range(lmax+1):
                for l12 in get_intermediates_w(l1,l2):
                    for l3 in range(lmax+1):
                        for l123 in get_intermediates_w(l12,l3):
                            for l4 in range(lmax+1):
                                for l1234 in get_intermediates_w(l123,l4):
                                    for l5 in range(lmax+1):
                                        for l12345 in get_intermediates_w(l1234,l5):
                                            for l6 in range(lmax+1):
                                                for l7 in get_intermediates_w(l12345,l6):
                                                    if (l1+l2+l3+l4+l5+l6 +l7) %2==0 and l7<=lmax:
                                                        if (l12345 + l5 +l1234) %2 ==0  and (l1234 +l4 +l123) %2 ==0 and (l123 +l3 +l12)%2==0 and (l12 +l1+l2) %2 ==0:
                                                                myl7s.append([l1,l2,l3,l4,l5,l6,l7])
        for n1 in range(1,nmax+1):
            for n2 in range(1,nmax+1):
                for n3 in range(1,nmax+1):
                    for n4 in range(1,nmax+1):
                        for n5 in range(1,nmax+1):
                            for n6 in range(1,nmax+1):
                                for n7 in range(1,nmax+1):
                                    for ind,l in enumerate(myl7s):
                                        l1,l2,l3,l4,l5,l6,l7 = l[0],l[1],l[2],l[3],l[4],l[5],l[6]
                                        x = [(l1,n1),(l2,n2),(l3,n3),(l4,n4),(l5,n5),(l6,n6),(l7,n7)]
                                        srt = sorted(x)
                                        if x == srt:
                                            stmp = "%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d" % (n1,n2,n3,n4,n5,n6,n7,l1,l2,l3,l4,l5,l6,l7)
                                            stmp_rev = "%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d" % (n7,n6,n5,n4,n3,n2,n1,l7,l6,l5,l4,l3,l2,l1)
                                            if enforce_dorder:
                                                if stmp_rev not in nl:
                                                    nl.append(stmp_rev)
                                            else:
                                                if stmp not in nl:
                                                    nl.append(stmp)
    elif rank ==8:

        myl8s = []
        for l1 in range(lmax+1):
            for l2 in range(lmax+1):
                for l12 in get_intermediates_w(l1,l2):
                    for l3 in range(lmax+1):
                        for l123 in get_intermediates_w(l12,l3):
                            for l4 in range(lmax+1):
                                for l1234 in get_intermediates_w(l123,l4):
                                    for l5 in range(lmax+1):
                                        for l12345 in get_intermediates_w(l1234,l5):
                                            for l6 in range(lmax+1):
                                                for l123456 in get_intermediates_w(l12345,l6):
                                                    for l7 in range(lmax+1):
                                                        for l8 in get_intermediates_w(l123456,l7):
                                                            if (l1+l2+l3+l4+l5+l6+l7+l8) %2==0 and l8<=lmax:
                                                                if (l123456 + l6 + l12345) %2 ==0 and (l12345 + l5 +l1234) %2 ==0  and (l1234 +l4 +l123) %2 ==0 and (l123 +l3 +l12)%2==0 and (l12 +l1+l2) %2 ==0:
                                                                    myl8s.append([l1,l2,l3,l4,l5,l6,l7,l8])
        for n1 in range(1,nmax+1):
            for n2 in range(1,nmax+1):
                for n3 in range(1,nmax+1):
                    for n4 in range(1,nmax+1):
                        for n5 in range(1,nmax+1):
                            for n6 in range(1,nmax+1):
                                for n7 in range(1,nmax+1):
                                    for n8 in range(1,nmax+1):
                                        for ind,l in enumerate(myl8s):
                                            l1,l2,l3,l4,l5,l6,l7,l8 = l[0],l[1],l[2],l[3],l[4],l[5],l[6],l[7]
                                            x = [(l1,n1),(l2,n2),(l3,n3),(l4,n4),(l5,n5),(l6,n6),(l7,n7),(l8,n8)]
                                            srt = sorted(x)
                                            if x == srt:
                                                stmp = "%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d" % (n1,n2,n3,n4,n5,n6,n7,n8,l1,l2,l3,l4,l5,l6,l7,l8)
                                                stmp_rev = "%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d" % (n8,n7,n6,n5,n4,n3,n2,n1,l8,l7,l6,l5,l4,l3,l2,l1)
                                                if enforce_dorder:
                                                    if stmp_rev not in nl:
                                                        nl.append(stmp_rev)
                                                else:
                                                    if stmp not in nl:
                                                        nl.append(stmp)
    elif rank>8:
        raise ValueError("nl combinations can only be calculated for rank 7 functions and below (currently)")

    return nl
