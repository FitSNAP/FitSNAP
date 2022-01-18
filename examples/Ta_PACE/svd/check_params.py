import numpy as np

old = []
new = []

descnew = []
descold = []

with open('Ta_pot.acecoeff','r') as readin:
	lines = readin.readlines()
	ndescs = int(lines[2].split()[-1])
	for descind in range(ndescs):
		line = lines[4 + descind]
		l = line.split()
		descnew.append(line.split('#')[-1])
		new.append(float(l[0]))

with open('./myrun/Ta_pot.acecoeff','r') as readin:
	lines = readin.readlines()
	ndescs = int(lines[2].split()[-1])
	for descind in range(ndescs):
		line = lines[4 + descind]
		l = line.split()
		descold.append(line.split('#')[-1])
		old.append(float(l[0]))
comp = [(a-b)/np.average([a,b]) for a,b in zip(new,old)]


for ind,c in enumerate(comp):
	print (c, descold[ind],descnew[ind])
