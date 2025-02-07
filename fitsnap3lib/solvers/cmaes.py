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
        residuals = np.nan_to_num(predicted_energy - subgroup['reference_energy'], nan=99)
        weighted_residuals = subgroup['weights'] * np.square(residuals)

    if reaxff_calculator.force:
      pass

    if reaxff_calculator.dipole:
        predicted_dipole = np.array(c['predicted_dipole'])
        reference_dipole = np.array(c["Dipole"])
        #print(f"predicted_dipole {predicted_dipole} reference_dipole {reference_dipole}")
        dipole_residuals = np.nan_to_num([predicted_dipole - reference_dipole for c in configs], nan=99)
        #print(f"dipole_residuals {dipole_residuals}")

    return (i_x_j[0], float(np.sum(weighted_residuals) + np.sum(np.square(dipole_residuals))))


def loss_function_data_index(i):

    d = reaxff_calculator.set_data_index(i)

    for x in self.pt.fitsnap_dict["x_arrays"]:
        reaxff_calculator.process_data_for_parameter_values(x)

    if reaxff_calculator.energy:
        pass


class CMAES(Solver):

    def __init__(self, name, pt, config):

        super().__init__(name, pt, config, linear=False)
        self.popsize = self.config.sections['SOLVER'].popsize
        self.sigma = self.config.sections['SOLVER'].sigma


    def parallel_loss_function(self, x_arrays):

        self.pt.shared_arrays['x_arrays'].array = x_arrays
        answer = [0.0] * len(x_arrays)
        #for p in self.executor.map(loss_function_config, range(len(all_data)), unordered=True):
        #   answer[p[0]] += p[1]

        self.executor.map(loss_function_data_index, self.range_all_data, unordered=True)
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
        #print( x0 )
        self.range_all_data = range(len(self.pt.fitsnap_dict["Data"]))
        bounds = np.empty([len(reaxff_calculator.parameters),2])

        for i, p in enumerate(reaxff_calculator.parameters):
            if 'range' in p:
                bounds[i] = p['range'] # FIXME 'range' config parser
            else:
                delta = 0.2*np.abs(x0[i])
                delta = delta if delta>0.0 else 1.0
                bounds[i] = [x0[i]-delta, x0[i]+delta]

        #pprint(bounds)

        #options={'maxiter': 99, 'maxfevals': 999, 'popsize': 3}
        options={
          'popsize': self.popsize, 'seed': 12345, #'maxiter': 1,
          'bounds': list(np.transpose(bounds))
        }

        self.pt.create_shared_array('x_arrays', self.popsize, len(reaxff_calculator.parameters))

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
            reaxff_calculator.change_parameters(x_best)
            all_data = reaxff_calculator.pt.fitsnap_dict["Data"]
            for subgroup in all_data:
              for c in subgroup['configs']:
                  reaxff_calculator.process_configs(c, 99)

            self.errors = all_data

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



