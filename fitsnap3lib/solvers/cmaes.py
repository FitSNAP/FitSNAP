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

        #print(f'x={x}')
        #print(f'len(self.configs)={len(self.configs)}')

        LammpsReaxff.change_parameters(self.calculator,x)
        LammpsReaxff.process_all_configs(self.calculator, self._data)

        # Good practice after a large parallel operation is to impose a barrier.
        self.pt.all_barrier()

        self.pt.shared_arrays['b'].array -= self.pt.shared_arrays['b'].array[self._reference_index]

        sse = np.sum(self._weights*(np.square((self.pt.shared_arrays['b'].array - self._energies)/1.255018947555075)))
        #exit()
        return sse


    def cmaes_constraints(self, x):

        #print("cmaes_constraints... self=" ,self)
        #print(type(args))
        #print(f'cmaes_constraints... {x}\nargs... {args}')

        constraints = []
        constraints.append(x[1]-x[0])
        constraints.append(x[2]-x[1])

        return constraints

    #def cmaes_update(self, es):
    #    self._stddev = es.stds


    def perform_fit(self, calculator, data):
        """
        Base class function for performing a fit.
        """

        self._data = sorted(data, key=lambda d: d['File'])
        self._energies = np.array([d['Energy'] for d in self._data])
        self._weights = np.array([d['Weight'] for d in self._data])
        self._reference_index  = 3
        self.calculator = calculator
        self.calculator.allocate_per_config(self._data)
        self.calculator.create_a()
        x0 = [p['value'] for p in self.parameters]

        #options={'maxiter': 99, 'maxfevals': 999, 'popsize': 3}
        options={
          'popsize': self.popsize, #'maxiter': 1,
          'bounds': [[p['range'][0] for p in self.parameters],[p['range'][1] for p in self.parameters]]
          }

        cfun = self.loss_function
        x, es = cma.fmin2( cfun, x0, self.sigma, options=options)

        #cfun = cma.ConstrainedFitnessAL(self.loss_function, self.cmaes_constraints)
        #x, es = cma.fmin2( cfun, x0, self.sigma, options=options, callback=cfun.update)

        print(es)
        #c = es.countiter
        #x = cfun.find_feasible(es)
        #print("find_feasible took {} iterations".format(es.countiter - c))
        #print(x,self.cmaes_constraints(x))
        print(self.pt.shared_arrays['b'].array)

    def error_analysis(self):
        pass























    def parallel_loss_function(parameter_arrays, self):

        # parallel_objective
        # an objective function that accepts a list of numpy.ndarray as input and returns a list, which is mostly used instead of objective_function, but for the initial (also initial elitist) and the final evaluations unless not callable(objective_function). If parallel_objective is given, the objective_function (first argument) may be None.

        print(f'parameter_arrays={parameter_arrays}')
        #print(f'args={args}')

        return [0.0 for p in parameter_arrays]
