import sys
import yaml
from operator import itemgetter

def scrape_yace(f):
    with open(f,'r') as readin:
        lines = readin.readlines()
        funcline = [line for line in lines if 'functions:' in line][0]
        funclineidx = lines.index(funcline)
        mu0lines = [line for line in lines[funclineidx+1:] if len(line.split()) ==1] 
        mu0lineidxs = [lines.index(line) for line in mu0lines]
        mu0s = [line.split(':')[0] for line in mu0lines ]
        nu_bymu0 = {mu0:[] for mu0 in mu0s}
        for lstind,mu0idx in enumerate(mu0lineidxs[:]):
            try:
                nextind = mu0lineidxs[lstind+1]
            except IndexError:
                nextind = None
            for line in lines[mu0idx+1:nextind]:
                dctstr = line.split(' - ')[-1]
                d = yaml.safe_load(dctstr)
                print (d)
                mus = d['mus']
                ns = d['ns']
                ls = d['ls']

                rank = d['rank']
                nulst = ['%d']*(rank*3)
                nustr = ','.join(b for b in nulst)
                nu = nustr % tuple(mus + ns + ls)
                #nus.append(nu)
                nu_bymu0[mu0s[lstind]].append(nu) 
                #print (nu)
    print(nu_bymu0)
    #print (nus)
    return nu_bymu0
    
#scrape_yace('potential.yace')
