
from fitsnap3lib.parallel_tools import ParallelTools
from fitsnap3lib.io.input import Config
from fitsnap3lib.calculators.inq import INQ
import json
import numpy as np

bond_scan = [{
    "Positions":[[-d/2,0.0,0.0], [d/2,0.0,0.0]], 
    "AtomTypes":['H','F']
    } for d in np.arange(0.6,1.401,.05)]

settings = { 
    'CALCULATOR': { 'calculator': 'INQ', 'energy': 1, 'force': 1, 'dipole': 1 },
    'INQ': { 'theory': 'PBE', 'cell': 'cubic 8.0 A finite' }
}

pt = ParallelTools()
config = Config(pt, settings)
inq = INQ('inq', pt, config)
inq.process_configs(bond_scan)

for i, b in enumerate(bond_scan):
    with open(f'JSON/HF/HF_{i}.json', 'w') as json_file:
        json.dump({"Dataset": {"Data": [b]}}, json_file)

