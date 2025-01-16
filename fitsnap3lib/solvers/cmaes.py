from fitsnap3lib.solvers.solver import Solver
from fitsnap3lib.calculators.lammps_reaxff import LammpsReaxff
import cma
import numpy as np
from pprint import pprint
from sys import exit

"""Methods you may or must override in new solvers"""

class CMAES(Solver):

    def __init__(self, name, pt, config):
        super().__init__(name, pt, config, linear=False)
        self.popsize = self.config.sections['CMAES'].popsize
        self.sigma = self.config.sections['CMAES'].sigma
        self.parameters = self.config.sections["REAXFF"].parameters
        

    def loss_function(self, x):

        LammpsReaxff.change_parameters(self.fs.calculator,x)
        LammpsReaxff.process_all_configs(self.fs.calculator, self.fs.data)

        # Good practice after a large parallel operation is to impose a barrier.
        self.pt.all_barrier()

        #for d in self.data:

        #self.pt.shared_arrays['b'].array -= self.pt.shared_arrays['b'].array[self._reference_index]

        #sse = np.sum(self._weights*(np.square((self.pt.shared_arrays['b'].array - self._energies)/1.255018947555075)))
        
        sse = 99999
        return sse


    def parallel_loss_function(self, x_list):
        return [self.loss_function(x) for x in self.pt.split_by_node(x_list)]


    def cmaes_constraints(self, x):

        #print("cmaes_constraints... self=" ,self)
        #print(type(args))
        #print(f'cmaes_constraints... {x}\nargs... {args}')

        constraints = []
        constraints.append(x[1]-x[0])
        constraints.append(x[2]-x[1])

        return constraints


    def perform_fit(self, fs):
        """
        Base class function for performing a fit.
        """

        #if( self.popsize % self.pt.get_size() == 0 ):
        #    self.pt.single_print(f"! WARNING: For optimal performance, please choose a population size which is multiple of MPI ranks.")

        self.fs = fs
        x0 = [p['value'] for p in self.parameters]

        #options={'maxiter': 99, 'maxfevals': 999, 'popsize': 3}
        options={
          'popsize': self.popsize, 'maxiter': 10,
          'bounds': [[p['range'][0] for p in self.parameters],[p['range'][1] for p in self.parameters]]
          }

        x_best, es = cma.fmin2( None, x0, self.sigma,
          parallel_objective=self.parallel_loss_function, options=options)

        LammpsReaxff.change_parameters(self.fs.calculator,x_best)
        self.fit = self.fs.calculator.force_field_string
        self.errors = es.pop_sorted

        #cfun = cma.ConstrainedFitnessAL(self.loss_function, self.cmaes_constraints)
        #x, es = cma.fmin2( cfun, x0, self.sigma, options=options, callback=cfun.update)

        #print("======== es ========")
        #pprint(vars(es))
        #print("======== es ========")
        #c = es.countiter
        #x = cfun.find_feasible(es)
        #print("find_feasible took {} iterations".format(es.countiter - c))
        #print(x,self.cmaes_constraints(x))

    def error_analysis(self):
        pass



