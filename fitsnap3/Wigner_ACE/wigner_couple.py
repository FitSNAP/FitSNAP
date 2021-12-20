import numpy as np
from .coupling_coeffs import *
from .gen_labels import *
from functools import partial


def rank_4(l):
    l1,l2,l3,l4 = l[0],l[1],l[2],l[3]

    l12s = get_intermediates_w(l1,l2)
    couplings = {}
    for m1 in range(-l1,l1+1):
        for m2 in range(-l2,l2+1):
            for m3 in range(-l3,l3+1):
                for m4 in range(-l4,l4+1):
                    mflag = (m1+m2+m3+m4)==0
                    if mflag:
                        couplings['%d,%d,%d,%d' % (m1,m2,m3,m4)] = 0.

    for l12 in l12s:
        for m1 in range(-l1,l1+1):
            for m2 in range(-l2,l2+1):
                for m12 in range(-l12,l12+1):
                    if (m1+m2) == m12:
                        for m3 in range(-l3,l3+1):
                            for m4 in range(-l4,l4+1):
                                if (m1+m2+m3+m4) ==0:
                                    mstr = '%d,%d,%d,%d' % (m1,m2,m3,m4)
                                    w1 = Wigner_3j['%d,%d,%d,%d,%d,%d' %(l1,m1,l2,m2,l12,-m12)]
                                    w2 = Wigner_3j['%d,%d,%d,%d,%d,%d' %(l3,m3,l4,m4,l12,m12)]
                                    w = w1*w2 * ((-1)**m12)
                                    try:
                                        #comment is example for C_L12 normalization (See drautz 2019 erratum)
					#intermediate weights are intentially left unnormalized - See James Notes
                                        couplings[mstr] += w #*(1/np.sqrt(len(l12s)))
                                    except KeyError:
                                        couplings[mstr] = w
    return couplings


def rank_5(l):
    l1,l2,l3,l4,l5 = l[0],l[1],l[2],l[3],l[4]

    l12s = get_intermediates_w(l1,l2)
    l123s = {l12:get_intermediates_w(l12,l3) for l12 in l12s}
    couplings = {}
    for l12 in l12s:
        for l123 in l123s[l12]:
            for m1 in range(-l1,l1+1):
                for m2 in range(-l2,l2+1):
                    for m12 in range(-l12,l12+1):
                        if (m1+m2) == m12:
                            for m3 in range(-l3,l3+1):
                                for m123 in range(-l123,l123+1):
                                    if (m12 + m3) == m123:
                                        for m4 in range(-l4,l4+1):
                                            for m5 in range(-l5,l5+1):
                                                if (m1+m2+m3+m4+m5) ==0:
                                                    mstr = '%d,%d,%d,%d,%d' % (m1,m2,m3,m4,m5)
                                                    w1 = Wigner_3j['%d,%d,%d,%d,%d,%d' %(l1,m1,l2,m2,l12,-m12)]
                                                    w2 = Wigner_3j['%d,%d,%d,%d,%d,%d' %(l12,m12,l3,m3,l123,-m123)]
                                                    w3 = Wigner_3j['%d,%d,%d,%d,%d,%d' %(l4,m4,l5,m5,l123,m123)]
                                                    w = w1*w2*w3* ((-1)**(m123+m12))
                                                    try:
                                                        couplings[mstr] += w
                                                    except KeyError:
                                                        couplings[mstr] = w
    return couplings

def rank_6(l,**kwargs):
    l1,l2,l3,l4,l5,l6 = l[0],l[1],l[2],l[3],l[4],l[5]

    l12s = get_intermediates_w(l1,l2)
    l123s = {l12:get_intermediates_w(l12,l3) for l12 in l12s}
    l1234s = { l12:{l123:get_intermediates_w(l123,l4) for l123 in l123s[l12]} for l12 in l12s}
    couplings = {}
    for l12 in l12s:
        for l123 in l123s[l12]:
            for l1234 in l1234s[l12][l123]:
                for m1 in range(-l1,l1+1):
                    for m2 in range(-l2,l2+1):
                        for m12 in range(-l12,l12+1):
                            if (m1+m2) == m12:
                                for m3 in range(-l3,l3+1):
                                    for m123 in range(-l123,l123+1):
                                        if (m12 + m3) == m123:
                                            for m4 in range(-l4,l4+1):
                                                for m1234 in range(-l1234,l1234+1):
                                                    if (m123 + m4) == m1234:
                                                        for m5 in range(-l5,l5+1):
                                                            for m6 in range(-l6,l6+1):
                                                                if (m1+m2+m3+m4+m5+m6) ==0:
                                                                    mstr = '%d,%d,%d,%d,%d,%d' % (m1,m2,m3,m4,m5,m6)
                                                                    w1 = Wigner_3j['%d,%d,%d,%d,%d,%d' %(l1,m1,l2,m2,l12,-m12)]
                                                                    w2 = Wigner_3j['%d,%d,%d,%d,%d,%d' %(l12,m12,l3,m3,l123,-m123)]
                                                                    w3 = Wigner_3j['%d,%d,%d,%d,%d,%d' %(l123,m123, l4,m4,l1234,-m1234)]
                                                                    w4 = Wigner_3j['%d,%d,%d,%d,%d,%d' %(l5,m5,l6,m6,l1234,m1234)]
                                                                    w = w1*w2*w3*w4* ((-1)**(m1234+m123+m12))
                                                                    try:
                                                                        couplings[mstr] += w
                                                                    except KeyError:
                                                                        couplings[mstr] = w
    return couplings

def rank_7(l):
    l1,l2,l3,l4,l5,l6,l7 = l[0],l[1],l[2],l[3],l[4],l[5],l[6]

    l12s = get_intermediates_w(l1,l2)
    l123s = {l12:get_intermediates_w(l12,l3) for l12 in l12s}
    l1234s = {l12:{l123:get_intermediates_w(l123,l4) for l123 in l123s[l12]} for l12 in l12s}
    l12345s =  {l12: {l123: { l1234:get_intermediates_w(l1234,l5) for l1234 in l1234s[l12][l123]} for l123 in l123s[l12]} for l12 in l12s}
    #print (l12345s)
    couplings = {}
    for l12 in l12s:
        for l123 in l123s[l12]:
            for l1234 in l1234s[l12][l123]:
                for l12345 in l12345s[l12][l123][l1234]:
                    for m1 in range(-l1,l1+1):
                        for m2 in range(-l2,l2+1):
                            for m12 in range(-l12,l12+1):
                                if (m1+m2) == m12:
                                    for m3 in range(-l3,l3+1):
                                        for m123 in range(-l123,l123+1):
                                            if (m12 + m3) == m123:
                                                for m4 in range(-l4,l4+1):
                                                    for m1234 in range(-l1234,l1234+1):
                                                        if (m123 + m4) == m1234:
                                                            for m5 in range(-l5,l5+1):
                                                                for m12345 in range(-l12345,l12345 +1):
                                                                    if (m1234 + m5) == m12345:
                                                                        for m6 in range(-l6,l6+1):
                                                                            for m7 in range(-l7,l7+1):
                                                                                if (m1+m2+m3+m4+m5+m6+m7) ==0:
                                                                                    mstr = '%d,%d,%d,%d,%d,%d,%d' % (m1,m2,m3,m4,m5,m6,m7)
                                                                                    w1 = Wigner_3j['%d,%d,%d,%d,%d,%d' %(l1,m1,l2,m2,l12,-m12)]
                                                                                    w2 = Wigner_3j['%d,%d,%d,%d,%d,%d' %(l12,m12,l3,m3,l123,-m123)]
                                                                                    w3 = Wigner_3j['%d,%d,%d,%d,%d,%d' %(l123,m123,l4,m4,l1234,-m1234)]
                                                                                    w4 = Wigner_3j['%d,%d,%d,%d,%d,%d' %(l1234,m1234,l5,m5,l12345,-m12345)]
                                                                                    w5 = Wigner_3j['%d,%d,%d,%d,%d,%d' %(l6,m6,l7,m7,l12345,m12345)]
                                                                                    #both phase factors work so far....
																											#Based on CG conversion
                                                                                    #w = w1*w2*w3*w4*w5* ((-1)**((l1-l2)+ (l12-l3) + (l123-l4) + (l1234-l5) + (l12345-l6) + m12345+m1234+m123+m12))
																											#Based on magnetic quantum number
                                                                                    w = w1*w2*w3*w4*w5* ((-1)**(( m12345+m1234+m123+m12)))
                                                                                    try:
                                                                                        couplings[mstr] += w
                                                                                    except KeyError:
                                                                                        couplings[mstr] = w

    return couplings

def rank_8(l):
    l1,l2,l3,l4,l5,l6,l7,l8 = l[0],l[1],l[2],l[3],l[4],l[5],l[6],l[7]

    l12s = get_intermediates_w(l1,l2)
    l123s = {l12:get_intermediates_w(l12,l3) for l12 in l12s}
    l1234s = {l12:{l123:get_intermediates_w(l123,l4) for l123 in l123s[l12]} for l12 in l12s}
    l12345s =  {l12: {l123: { l1234:get_intermediates_w(l1234,l5) for l1234 in l1234s[l12][l123]} for l123 in l123s[l12]} for l12 in l12s}
    l123456s =  {l12: {l123: { l1234:{l12345: get_intermediates_w(l12345,l6) for l12345 in l12345s[l12][l123][l1234] } for l1234 in l1234s[l12][l123]} for l123 in l123s[l12]} for l12 in l12s}
    couplings = {}
    for l12 in l12s:
        for l123 in l123s[l12]:
            for l1234 in l1234s[l12][l123]:
                for l12345 in l12345s[l12][l123][l1234]:
                    for l123456 in l123456s[l12][l123][l1234][l12345]:
                        for m1 in range(-l1,l1+1):
                            for m2 in range(-l2,l2+1):
                                for m12 in range(-l12,l12+1):
                                    if (m1+m2) == m12:
                                        for m3 in range(-l3,l3+1):
                                            for m123 in range(-l123,l123+1):
                                                if (m12 + m3) == m123:
                                                    for m4 in range(-l4,l4+1):
                                                        for m1234 in range(-l1234,l1234+1):
                                                            if (m123 + m4) == m1234:
                                                                for m5 in range(-l5,l5+1):
                                                                    for m12345 in range(-l12345,l12345 +1):
                                                                        if (m1234 + m5) == m12345:
                                                                            for m6 in range(-l6,l6+1):
                                                                                for m123456 in range(-l123456,l123456+1):
                                                                                    for m7 in range(-l7,l7+1):
                                                                                        for m8 in range(-l8,l8+1):
                                                                                            if (m1+m2+m3+m4+m5+m6+m7+m8) ==0:
                                                                                                mstr = '%d,%d,%d,%d,%d,%d,%d,%d' % (m1,m2,m3,m4,m5,m6,m7,m8)
                                                                                                w1 = Wigner_3j['%d,%d,%d,%d,%d,%d' %(l1,m1,l2,m2,l12,-m12)]
                                                                                                w2 = Wigner_3j['%d,%d,%d,%d,%d,%d' %(l12,m12,l3,m3,l123,-m123)]
                                                                                                w3 = Wigner_3j['%d,%d,%d,%d,%d,%d' %(l123,m123,l4,m4,l1234,-m1234)]
                                                                                                w4 = Wigner_3j['%d,%d,%d,%d,%d,%d' %(l1234,m1234,l5,m5,l12345,-m12345)]
                                                                                                w5 = Wigner_3j['%d,%d,%d,%d,%d,%d' %(l12345,m12345,l6,m6,l123456,-m123456)]
                                                                                                w6 = Wigner_3j['%d,%d,%d,%d,%d,%d' %(l7,m7,l8,m8,l123456,m123456)]
                                                                                                w = w1*w2*w3*w4*w5*w6* ((-1)**(( m123456+m12345+m1234+m123+m12)))
                                                                                                try:
                                                                                                    couplings[mstr] += w
                                                                                                except KeyError:
                                                                                                    couplings[mstr] = w

    return couplings



def get_coupling(nus,ranks,**kwargs):
    coupling = {str(rank):{} for rank in ranks}
    l_only = {}
    weights = [1.]
    wigner_flag =True
    try:
        winger_flag = kwargs['wigner_flag']
    except KeyError:
        print ('using default generalized Wigner 3j couplings')
        print ('generalized CG couplings have been removed')

    for nu in nus:
        rank = get_nu_rank(nu)
        rnk = str(rank)
        n,l = get_n_l(nu)
        if rank ==1:
            coupling[rnk][nu] = rank_1_ccs()
        elif rank ==2:
            coupling[rnk][nu] = rank_2_ccs(n,l)
        elif rank ==3:
            coupling[rnk][nu] = rank_3_ccs(n,l)
        elif rank ==4:
            llst = ['%d']*rank
            lstr = ','.join(str(i) for i in llst ) % tuple(l)
            try:
                coupling[rnk][nu] = l_only[lstr]
            except KeyError:
                coupling[rnk][nu] = rank_4(l)
                l_only[lstr] = coupling[rnk][nu]
        elif rank ==5:
            llst = ['%d']*rank
            lstr = ','.join(str(i) for i in llst ) % tuple(l)
            try:
                coupling[rnk][nu] = l_only[lstr]
            except KeyError:
                coupling[rnk][nu] = rank_5(l)
                l_only[lstr] = coupling[rnk][nu]
        elif rank ==6:
            llst = ['%d']*rank
            lstr = ','.join(str(i) for i in llst ) % tuple(l)
            try:
                coupling[rnk][nu] = l_only[lstr]
            except KeyError:
                coupling[rnk][nu] = rank_6(l)
                l_only[lstr] = coupling[rnk][nu]
        elif rank ==7:
            llst = ['%d']*rank
            lstr = ','.join(str(i) for i in llst ) % tuple(l)
            try:
                coupling[rnk][nu] = l_only[lstr]
            except KeyError:
                coupling[rnk][nu] = rank_7(l)
                l_only[lstr] = coupling[rnk][nu]
        elif rank ==8:
            llst = ['%d']*rank
            lstr = ','.join(str(i) for i in llst ) % tuple(l)
            try:
                coupling[rnk][nu] = l_only[lstr]
            except KeyError:
                coupling[rnk][nu] = rank_8(l)
                l_only[lstr] = coupling[rnk][nu]
        elif rank >8:
            raise ValueError("Cannot generate couplings for rank %d. Couplings up to rank 8 have been implemented" % rank)
    return coupling,weights

