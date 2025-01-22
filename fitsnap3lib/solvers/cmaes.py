from fitsnap3lib.solvers.solver import Solver
from fitsnap3lib.calculators.lammps_reaxff import LammpsReaxff

import cma, itertools, functools
import numpy as np
from pprint import pprint
from sys import exit


def force_field_string(x):
    return LammpsReaxff.change_parameters(reaxff_calculator,x)


def force_field_strings(strings):
    global ff_strings
    ff_strings = strings


def loss_function_tuple(i_j_x):

    d = reaxff_calculator.pt.fitsnap_dict["Data"][i_j_x[0]]
    #reaxff_calculator.change_parameters(i_j_x[2])
    parameters = reaxff_calculator.pt.shared_arrays["parameters"].array[i_j_x[1]]
    reaxff_calculator.change_parameters(parameters)
    reaxff_calculator.process_reaxff_config(d)
    return (d['ground_shared_index'], d['ground_relative_index'], i_j_x[0], i_j_x[1], d['predicted_energy'])


class CMAES(Solver):

    def __init__(self, name, pt, config):
        super().__init__(name, pt, config, linear=False)
        self.popsize = self.config.sections['CMAES'].popsize
        self.sigma = self.config.sections['CMAES'].sigma
        self.parameters = self.config.sections["REAXFF"].parameters


    def parallel_loss_function(self, x_arrays):

        #print("\n\nx_arrays="); pprint(x_arrays)
        self.pt.shared_arrays["parameters"].array[:] = np.row_stack(x_arrays)
        #print("\n\nshared_array="); pprint(self.pt.shared_arrays["parameters"].array)
        all_data = self.pt.fitsnap_dict["Data"]
        x_data_pairs = itertools.product(range(len(x_arrays)),range(len(all_data)))
        #tuples = [(i,j,x_arrays[j]) for j, i in x_data_pairs]
        tuples = [(i,j) for j, i in x_data_pairs]
        parallel_results = list(self.executor.map(loss_function_tuple, tuples,unordered=True))
        sorted_results = sorted(parallel_results, key=lambda t: (t[0],t[3],t[2]))

        #print(f"self.pt.get_rank()={self.pt.get_rank()}")

        answer = [0.0] * len(x_arrays)
        #answer = []

        for k, g in itertools.groupby(sorted_results, key=lambda t: (t[0],t[3])):

            # (ground_shared_index, ground_relative_index, shared_index, x_index, computed_energy)
            group = list(g)
            #print(f"\nk={k} group="); pprint(group)
            predicted = np.array([t[4]-group[i+t[1]][4] for i, t in enumerate(group)])
            reference = np.array([all_data[t[2]]["Energy"] for t in group])
            weights = np.array([all_data[t[2]]["Weight"] for t in group])
            #weighted_residuals = weights * np.square((predicted - reference)/1.2550189475550748)
            weighted_residuals = np.square((predicted - reference))
            answer[k[1]] += float(np.sum(weighted_residuals))

        #print(answer)
        return answer


    def perform_fit(self, fs):
        """
        Base class function for performing a fit.
        """

        # "Avoid global variables that are declared outside class methods or attributes"
        # no way around this sorry
        global reaxff_calculator
        reaxff_calculator = fs.calculator

        x0 = [p['value'] for p in self.parameters]

        #options={'maxiter': 99, 'maxfevals': 999, 'popsize': 3}
        options={
          'popsize': self.popsize, 'seed': 12345, #'maxiter': 5,
          'bounds': [[p['range'][0] for p in self.parameters],[p['range'][1] for p in self.parameters]]
        }

        self.pt.create_shared_array('parameters', self.popsize, len(self.parameters))

        if self.pt.stubs == 0:
            from mpi4py import MPI
            from mpi4py.futures import MPICommExecutor, wait

            # SAFER TO USE *MPICommExecutor* INSTEAD OF *MPIPoolExecutor*
            # "Legacy MPI-1 implementations (as well as some vendor MPI-2 implementations) do not support the dynamic process management features introduced in the MPI-2 standard. Additionally, job schedulers and batch systems in supercomputing facilities may pose additional complications to applications using the MPI_Comm_spawn() routine.
            # [https://mpi4py.readthedocs.io/en/stable/mpi4py.futures.html#mpicommexecutor]

            with MPICommExecutor(MPI.COMM_WORLD, root=0) as self.executor:
                if self.executor is not None:
                    x_best, es = cma.fmin2( None, x0, self.sigma,
                        parallel_objective=self.parallel_loss_function, options=options)

        if self.pt.stubs == 1:
            x_best, es = cma.fmin2( None, x0, self.sigma,
                parallel_objective=self.parallel_loss_function, options=options)

        if self.pt._rank == 0:
            self.fit = reaxff_calculator.change_parameters_string(x_best)
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


    def cmaes_constraints(self, x):

        #print("cmaes_constraints... self=" ,self)
        #print(type(args))
        #print(f'cmaes_constraints... {x}\nargs... {args}')

        constraints = []
        constraints.append(x[1]-x[0])
        constraints.append(x[2]-x[1])

        return constraints



