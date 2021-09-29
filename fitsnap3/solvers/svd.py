from .solver import Solver
from ..parallel_tools import pt
from ..io.input import config
from scipy.linalg import lstsq
import numpy as np


class SVD(Solver):

    def __init__(self, name):
        super().__init__(name)

    def perform_fit(self):
        if pt.shared_arrays['configs_per_group'].testing_elements != 0:
            testing = -1*pt.shared_arrays['configs_per_group'].testing_elements
        else:
            testing = len(pt.shared_arrays['w'].array)
        #print(testing)
        w = pt.shared_arrays['w'].array[:testing]
        aw, bw = w[:, np.newaxis] * pt.shared_arrays['a'].array[:testing], w * pt.shared_arrays['b'].array[:testing]
        #print(w.shape)
        #print(aw.shape)
        #print(bw.shape)
        #print(pt.shared_arrays['a'].array.shape)
        #print(pt.shared_arrays['b'].array.shape)
#        Transpose method does not work with Quadratic SNAP (why?)
#        We need to revisit this preconditioning of the linear problem, we can make this a bit more elegant. 
#        Since this breaks some examples this will stay as a 'secret' feature. 
#        Need to chat with some mathy people on how we can profile A and find good preconditioners. 
#        Will help when we want to try gradient based linear solvers as well. 
        if config.sections['EXTRAS'].apply_transpose:
            bw = aw.T@bw
            aw = aw.T@aw
        self.fit, residues, rank, s = lstsq(aw, bw, 1.0e-13)
        #print(aw.T@bw)
        #print(pt.shared_arrays['atb'].array)
        #print(self.fit)
        #print(self.fit.shape)

        fileA = open('A.bin', 'wb');        
        pt.shared_arrays['a'].array.flatten(order = 'F').astype('float64').tofile(fileA);
        fileA.close();
        fileb = open('b.bin', 'wb');        
        pt.shared_arrays['b'].array.flatten(order = 'F').astype('float64').tofile(fileb);
        fileb.close();        
        filec = open('c.bin', 'wb');        
        self.fit.flatten(order = 'F').astype('float64').tofile(filec);
        filec.close();        
        fileE = open('eindex.bin', 'wb');        
        eindex = np.array(pt.shared_arrays['a'].energy_index)
        eindex.flatten(order = 'F').astype('float64').tofile(fileE)
        fileE.close()
        fileF = open('findex.bin', 'wb');        
        findex = np.array(pt.shared_arrays['a'].force_index)
        findex.flatten(order = 'F').astype('float64').tofile(fileF);
        fileF.close();        
        fileS = open('fspacing.bin', 'wb');        
        num_forces = np.array(pt.shared_arrays['a'].num_atoms)*3
        num_forces.flatten(order = 'F').astype('float64').tofile(fileS);
        fileS.close();        
        fileV = open('vindex.bin', 'wb');        
        sindex = np.array(pt.shared_arrays['a'].stress_index)
        sindex.flatten(order = 'F').astype('float64').tofile(fileV);
        fileV.close();

    def perform_atafit(self):
        ata = 0.5*(pt.shared_arrays['ata'].array + pt.shared_arrays['ata'].array.T)
        atb = pt.shared_arrays['atb'].array               
        self.coeff, residues, rank, s = lstsq(ata, atb, 1.0e-13)

    def _dump_a(self):
        np.savez_compressed('a.npz', a=pt.shared_arrays['a'].array)

    def _dump_x(self):
        np.savez_compressed('x.npz', x=self.fit)

    def _dump_b(self):
        b = pt.shared_arrays['a'].array @ self.fit
        np.savez_compressed('b.npz', b=b)
