from fitsnap3lib.solvers.solver import Solver
from fitsnap3lib.calculators.lammps_reaxff import LammpsReaxff

import cma, itertools, functools
import numpy as np
from pprint import pprint
from sys import exit

# MPICommExecutor Legacy MPI-1 implementations (as well as some vendor MPI-2 implementations) do not support the dynamic process management features introduced in the MPI-2 standard. Additionally, job schedulers and batch systems in supercomputing facilities may pose additional complications to applications using the MPI_Comm_spawn() routine. [https://mpi4py.readthedocs.io/en/stable/mpi4py.futures.html#mpicommexecutor]


def force_field_string(x):

    return LammpsReaxff.change_parameters(reaxff_calculator,x)


def loss_function_tuple(i_x_j):

  #print(f"reaxff_calculator.pt.get_rank()={reaxff_calculator.pt.get_rank()} index_x_data={index_x_data}")

  shared_index = i_x_j[2]
  d = reaxff_calculator.pt.fitsnap_dict["Data"][shared_index]
  reaxff_calculator.force_field_string = i_x_j[1]
  LammpsReaxff.process_reaxff_config(reaxff_calculator, d, shared_index)
  computed_energy = float(reaxff_calculator.pt.shared_arrays['energy'].array[shared_index])
  d['Weight'] = 1.0

  return (i_x_j[0],d['Weight'],d['Energy'],d['relative_energy_index'],computed_energy)


class CMAES(Solver):

    def __init__(self, name, pt, config):
        super().__init__(name, pt, config, linear=False)
        self.popsize = self.config.sections['CMAES'].popsize
        self.sigma = self.config.sections['CMAES'].sigma
        self.parameters = self.config.sections["REAXFF"].parameters


    def parallel_loss_function(self, x_arrays):

        ff_strings = list(self.executor.map(force_field_string, x_arrays))

        x_data_pairs = itertools.product(range(len(x_arrays)), range(len(self.pt.fitsnap_dict["Data"])))
        tuples = [(i,ff_strings[i],j) for i, j in x_data_pairs]
        #print(tuples)
        tmp = list(self.executor.map(loss_function_tuple, tuples, chunksize=7, unordered=True))

        #print(f"self.pt.get_rank()={self.pt.get_rank()}")

        answer = []

        for k, g in itertools.groupby(tmp, key=lambda t: t[0]):

          # (index_x_data[0],d['Weight'],d['Energy'],d['relative_energy_index'],computed_energy)

          g_list = list(g)

          pred = np.array([t[4]-tmp[i+t[3]][4] for i, t in enumerate(g_list)])

          reference = np.array([t[2] for t in g_list])
          weighted_residuals = [g_list[i][1]*((pred[i]-reference[i])/1.2550189475550748)**2 for i in range(len(g_list))]
          answer.append(float(np.sum(weighted_residuals)))

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

        global reaxff_calculator
        reaxff_calculator = fs.calculator

        x0 = [p['value'] for p in self.parameters]

        #options={'maxiter': 99, 'maxfevals': 999, 'popsize': 3}
        options={
          'popsize': self.popsize, 'seed': 12345,
          'bounds': [[p['range'][0] for p in self.parameters],[p['range'][1] for p in self.parameters]]
        }

        if self.pt.stubs == 0:
            from mpi4py import MPI
            from mpi4py.futures import MPICommExecutor

            with MPICommExecutor(MPI.COMM_WORLD, root=0) as self.executor:
                if self.executor is not None:
                    x_best, es = cma.fmin2( None, x0, self.sigma,
                        parallel_objective=self.parallel_loss_function, options=options)

        if self.pt.stubs == 1:
            x_best, es = cma.fmin2( None, x0, self.sigma,
                parallel_objective=self.parallel_loss_function, options=options)

        LammpsReaxff.change_parameters(reaxff_calculator,x_best)
        self.fit = reaxff_calculator.force_field_string
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



