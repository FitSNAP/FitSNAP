from fitsnap3lib.solvers.solver import Solver
from fitsnap3lib.calculators.lammps_reaxff import LammpsReaxff

import cma, itertools, functools
import numpy as np
from pprint import pprint
from sys import exit


def loss_function_subgroup(i_x_j):

    subgroup = reaxff_calculator.pt.fitsnap_dict["Data"][i_x_j[2]]
    reaxff_calculator.change_parameters(i_x_j[1])
    configs = subgroup['configs']
    for c in configs: reaxff_calculator.process_configs(c, i_x_j[0])

    if reaxff_calculator.energy:
      ground_predicted_energy = configs[subgroup['ground_index']]['predicted_energy']
      for c in configs: c['predicted_energy'] -= ground_predicted_energy
      predicted_energy = np.array([c['predicted_energy'] for c in configs])
      #pprint(predicted_energy)
      weighted_residuals = subgroup['weights'] * np.square((predicted_energy - subgroup['reference_energy']))

    if reaxff_calculator.force:
      pass

    return (i_x_j[0], float(np.sum(weighted_residuals)))


class CMAES(Solver):

    def __init__(self, name, pt, config):

        super().__init__(name, pt, config, linear=False)
        self.popsize = self.config.sections['SOLVER'].popsize
        self.sigma = self.config.sections['SOLVER'].sigma


    def parallel_loss_subgroup(self, x_arrays):

        x_list = x_arrays if isinstance(x_arrays, list) else [x_arrays]
        all_data = self.pt.fitsnap_dict["Data"]
        x_subgroup_pairs = itertools.product(range(len(x_list)),range(len(all_data)))
        tuples = [(i,x_list[i],j) for i, j in x_subgroup_pairs]
        answer = [0.0] * len(x_list)
        for p in self.executor.map(loss_function_subgroup, tuples, unordered=True): answer[p[0]] += p[1]
        print(answer)
        return answer


    def perform_fit(self, fs):
        """
        Base class function for performing a fit.
        """

        # "Avoid global variables that are declared outside class methods or attributes"
        # no way around this sorry
        global reaxff_calculator
        reaxff_calculator = fs.calculator

        x0 = reaxff_calculator.values
        print( x0 )

        bounds = np.empty([len(reaxff_calculator.parameters),2])

        for i, p in enumerate(reaxff_calculator.parameters):
            if 'range' in p:
                bounds[i] = p['range'] # FIXME 'range' config parser
            elif p[0]==0:
                bounds[i] = [0.0, 99.99]
            else:
                delta = 0.2*np.abs(x0[i])
                delta = delta if delta>0.0 else 1.0
                bounds[i] = [x0[i]-delta, x0[i]+delta]

        #print(bounds)

        #options={'maxiter': 99, 'maxfevals': 999, 'popsize': 3}
        options={
          'popsize': self.popsize, 'seed': 12345, 'maxiter': 3,
          'bounds': list(np.transpose(bounds))
        }

        if self.pt.stubs == 0:
            from mpi4py import MPI
            from mpi4py.futures import MPICommExecutor, wait

            # SAFER TO USE *MPICommExecutor* INSTEAD OF *MPIPoolExecutor*
            # "Legacy MPI-1 implementations (as well as some vendor MPI-2 implementations) do not support the dynamic process management features introduced in the MPI-2 standard. Additionally, job schedulers and batch systems in supercomputing facilities may pose additional complications to applications using the MPI_Comm_spawn() routine.
            # [https://mpi4py.readthedocs.io/en/stable/mpi4py.futures.html#mpicommexecutor]

            with MPICommExecutor(MPI.COMM_WORLD, root=0) as self.executor:
                if self.executor is not None:
                    x_best, es = cma.fmin2( None, x0, self.sigma,
                        parallel_objective=self.parallel_loss_subgroup, options=options)

        if self.pt.stubs == 1:
            x_best, es = cma.fmin2( None, x0, self.sigma,
                parallel_objective=self.parallel_loss_function, options=options)

        if self.pt._rank == 0:
            self.fit = reaxff_calculator.change_parameters_string(x_best)
            self.errors = es.pop_sorted

        #cfun = cma.ConstrainedFitnessAL(self.loss_function, self.cmaes_constraints)
        #x, es = cma.fmin2( cfun, x0, self.sigma, options=options, callback=cfun.update)

        #c = es.countiter
        #x = cfun.find_feasible(es)
        #print("find_feasible took {} iterations".format(es.countiter - c))
        #print(x,self.cmaes_constraints(x))

    def error_analysis(self):
        pass


    def cmaes_constraints(self, x):

        #print("cmaes_constraints... self=" ,self)
        #print(type(args))
        #print(f'cmaes_constraints... {x}\nargs... {args}')

        constraints = []
        constraints.append(x[1]-x[0])
        constraints.append(x[2]-x[1])

        return constraints



