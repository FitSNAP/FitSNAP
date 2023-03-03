import numpy as np
from ase import Atom,Atoms
from ase.io import read,write

class fsnap_atoms():
    def __init__(self):
        topkeys = ['Data', 'PositionsStyle', 'AtomTypeStyle', 'Label', 'StressStyle', 'LatticeStyle', 'EnergyStyle', 'ForcesStyle']
        self.data = None
        self.dataset = {key:None for key in topkeys}
        return None

    def read_json(self,jfile):
        import json5
        with open(jfile,'r') as readin:
            d = json5.load(readin)
            dsetkeys = d['Dataset'].keys()
            for key in dsetkeys:
                if key != 'Data':
                    self.dataset[key] = d['Dataset'][key]
                elif key == 'Data':
                    self.dataset[key] = d['Dataset'][key]
            self.data = d['Dataset']['Data'][0]

    def set_ASE(self,atoms,magmomflag = False,chargeflag=False, **kwargs):
        self.dataset['AtomTypeStyle'] = "chemicalsymbol"
        self.dataset['EnergyStyle'] = "electronvolt"
        self.dataset['ForceStyle'] = "electronvoltperangstrom"
        self.dataset['Label'] = "ASE generated"
        self.dataset['LatticeStyle'] = "angstrom"
        self.dataset['PositionsStyle'] = "angstrom"
        self.dataset['StressStyle'] = "kB"
        dkeys = ['Stress', 'Positions', 'Energy', 'AtomTypes', 'Lattice', 'NumAtoms', 'Forces']
        data = {dkey:None for dkey in dkeys}
        data['Positions'] = atoms.positions.copy().tolist()
        try:
            en = kwargs['Energy']
        except KeyError:
            en = 0.
        try:
            frc = kwargs['Forces']
        except KeyError:
            frc = [[0.0,0.0,0.0]]*len(atoms)
        try:
            stress = kwargs['Stress']
        except KeyError:
            stress= [[0.0,0.0,0.0],
                [0.0,0.0,0.0],
                [0.0,0.0,0.0] ]

        data['Stress']=stress
        data['Energy'] = en
        data['AtomTypes'] = atoms.get_chemical_symbols()
        data['Lattice'] = atoms.get_cell().tolist()
        data['NumAtoms'] = len(atoms)
        data['Forces'] = frc
        if magmomflag:
            data['MagneticMoments'] = atoms.magmoms.copy().tolist()
        if chargeflag:
            data['Charges'] = atoms.charges.copy().tolist()

        self.data = data
        self.dataset['Data'] = [data]

    def get_ASE(self):
        atoms = Atoms(self.data['AtomTypes'])
        atoms.set_positions(self.data['Positions'])
        atoms.set_cell(self.data['Lattice'])
        atoms.set_pbc(True)
        return atoms

    def reset_chems(self,chemmap):
        symbols = self.data['AtomTypes']
        new_symbols = [chemmap[symbol] for symbol in symbols]
        self.data['AtomTypes'] = new_symbols
        self.dataset['Data'] = [self.data]

    def write_JSON(self,name):
        import json
        with open('%s.json'%name,'w') as writeout:
            writeout.write('# file %s\n' % name)
            json.dump({'Dataset': self.dataset}, writeout,sort_keys=True,indent=2)


fsats = fsnap_atoms()
fsats.read_json("FitSnap_struct.json")
aseats = fsats.get_ASE()

aseats.set_chemical_symbols(['Ta','Xe','Ta','Xe'])

fsats.set_ASE(aseats)
mp = {'Ta':'Cu1', 'Xe': 'Cu2'}
fsats.reset_chems(mp)
fsats.write_JSON('chemtest')
