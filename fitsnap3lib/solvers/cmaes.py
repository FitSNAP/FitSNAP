from fitsnap3lib.solvers.solver import Solver
from fitsnap3lib.calculators.lammps_reaxff import LammpsReaxff

import cma, itertools, functools
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
        

    def loss_function(self, x, d):

        shared_index = d["shared_index"]
        LammpsReaxff.change_parameters(self.fs.calculator,x)
        LammpsReaxff.process_configs(self.fs.calculator, d, shared_index)
        computed_energy = float(self.pt.shared_arrays['energy'].array[shared_index])
        return {**{k: d[k] for k in ['Energy','relative_energy_index']},**{"computed_energy": computed_energy}}


    def parallel_loss_function(self, x_arrays):

        # list.index() doesnt work with numpy arrays, convert them to lists
        x_list = [list(a) for a in x_arrays]
        #pprint(x_list)
        x_data_pairs = self.pt.split_by_node(list(itertools.product(x_list, self.pt.fitsnap_dict["Data"])))
        tmp = list([(x_list.index(x),self.loss_function(x, d)) for x, d in x_data_pairs])
        #pprint(tmp)
        #self.pt.all_barrier()

        answer = []

        for k, g in itertools.groupby(tmp, key=lambda t: t[0]):
          #print("k=",k)

          # k= 0
          # 0 (0, {'Energy': 14.72554246214304, 'relative_energy_index': 3, 'computed_energy': -246.83747238283183})
          # 1 (0, {'Energy': 5.628891278123319, 'relative_energy_index': 2, 'computed_energy': -251.02248371565693})
          # 2 (0, {'Energy': 1.1565879346369456, 'relative_energy_index': 1, 'computed_energy': -252.26329117497153})

          g_list = list(g)

          pred = np.array([t['computed_energy']-tmp[i+t['relative_energy_index']][1]['computed_energy'] for i, (_,t) in enumerate(g_list)])

          reference = np.array([t['Energy'] for (_,t) in g_list])
          answer.append(float(np.sum((pred - reference)**2)))

        #pprint(answer)
        return answer


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



