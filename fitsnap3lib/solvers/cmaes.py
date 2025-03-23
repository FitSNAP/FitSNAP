from fitsnap3lib.solvers.solver import Solver
from fitsnap3lib.calculators.lammps_reaxff import LammpsReaxff
import cma, itertools, functools
import numpy as np
from pprint import pprint

# ------------------------------------------------------------------------------------------------

def loss_function(x_arrays):

    print(f"*** rank {reaxff_calculator.pt._rank} x_arrays {x_arrays}")
    #d = reaxff_calculator.set_data_index(index)

    return reaxff_calculator.process_configs_with_values(x_arrays)

# ------------------------------------------------------------------------------------------------

class CMAES(Solver):

    # --------------------------------------------------------------------------------------------

    def __init__(self, name, pt, config):

        super().__init__(name, pt, config, linear=False)
        self.popsize = self.config.sections['SOLVER'].popsize
        self.sigma = self.config.sections['SOLVER'].sigma

    # --------------------------------------------------------------------------------------------

    def parallel_loss_function(self, x_arrays):

        #self.pt.shared_arrays['x_arrays'].array[:] = x_arrays
        #self.pt.shared_arrays['x_arrays'].win.Sync()
        # FIXME: no-op ? is it even needed ??
        #self.pt.shared_arrays['x_arrays'].win.Flush(MPI.PROC_NULL)
        #for r in range(1, ): self.executor.submit(fence_x_arrays, r)
        #self.pt.shared_arrays['x_arrays'].win.Fence()

        futures = [self.executor.submit(loss_function, x_arrays) for _ in self.range_workers]
        results = [f.result() for f in futures]
        print(results)

        def sum_weighted_residual(i):
            ground_index = self.pt.shared_arrays['ground_index'].array
            reference_energy = self.pt.shared_arrays['reference_energy'].array
            predicted_energy = self.pt.shared_arrays['predicted_energy'].array[i]
            weights = self.pt.shared_arrays['weights'].array
            residual = np.square(predicted_energy - predicted_energy[ground_index] - reference_energy)
            return np.sum(weights * np.nan_to_num(residual,99))

        answer = [sum_weighted_residual(i) for i in range(len(x_arrays))]
        #print(answer)
        return answer

    # --------------------------------------------------------------------------------------------

    def perform_fit(self, fs):

        # "Avoid global variables that are declared outside class methods or attributes"
        # no way around this sorry
        global reaxff_calculator
        reaxff_calculator = fs.calculator

        x0 = self.config.sections['REAXFF'].values
        #print( x0 )
        self.range_workers = range(1, self.pt.get_size())

        import warnings
        warnings.simplefilter("ignore", category=UserWarning)

        options={
          'popsize': self.popsize, 'seed': 12345, 'maxiter': 3,
          'bounds': list(np.transpose(self.config.sections['REAXFF'].parameter_bounds))
        }

        if self.pt.stubs == 0:
            # SAFER TO USE *MPICommExecutor* INSTEAD OF *MPIPoolExecutor*
            # [https://mpi4py.readthedocs.io/en/stable/mpi4py.futures.html#mpicommexecutor]
            from mpi4py import MPI
            from mpi4py.futures import MPICommExecutor

            with MPICommExecutor(MPI.COMM_WORLD, root=0) as self.executor:
                if self.executor is not None:
                    self.x_best, es = cma.fmin2( None, x0, self.sigma,
                        parallel_objective=self.parallel_loss_function, options=options)
        else:
            self.x_best, es = cma.fmin2( None, x0, self.sigma,
                parallel_objective=self.parallel_loss_function, options=options)

        if self.pt._rank == 0:
            self.fit = reaxff_calculator.change_parameters_string(self.x_best)

            print("----------------------------------------")
            print(self.config.sections['CALCULATOR'].charge_fix)
            print("PARAMETER_NAME          BEFORE     AFTER")
            for p, x0i, xbi in zip(reaxff_calculator.parameter_names, x0, self.x_best):
                print(f"{p:<20} {x0i:9.4f} {xbi:9.4f}")
            print("----------------------------------------")

    # --------------------------------------------------------------------------------------------

    def error_analysis(self):

        pass

        #for i in self.range_all_data:
        #    d = reaxff_calculator.set_data_index(i)
        #    reaxff_calculator.process_data_for_parameter_values(0, self.x_best)


        #all_data = reaxff_calculator.pt.fitsnap_dict["Data"]
        #for subgroup in all_data:
        #  for c in subgroup['configs']:
        #      reaxff_calculator.process_configs(c, 99)
        #self.errors = all_data

    # --------------------------------------------------------------------------------------------










































################################ SCRATCH ################################



    def cmaes_constraints(self, x):

        #print("cmaes_constraints... self=" ,self)
        #print(type(args))
        #print(f'cmaes_constraints... {x}\nargs... {args}')

        #cfun = cma.ConstrainedFitnessAL(self.loss_function, self.cmaes_constraints)
        #x, es = cma.fmin2( cfun, x0, self.sigma, options=options, callback=cfun.update)

        #c = es.countiter
        #x = cfun.find_feasible(es)
        #print("find_feasible took {} iterations".format(es.countiter - c))
        #print(x,self.cmaes_constraints(x))

        constraints = []
        constraints.append(x[1]-x[0])
        constraints.append(x[2]-x[1])

        return constraints



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

    if reaxff_calculator.dipole:
        predicted_dipole = np.array(c['predicted_dipole'])
        reference_dipole = np.array(c["Dipole"])
        #print(f"predicted_dipole {predicted_dipole} reference_dipole {reference_dipole}")
        dipole_residuals = np.nan_to_num([predicted_dipole - reference_dipole for c in configs], nan=99)
        #print(f"dipole_residuals {dipole_residuals}")

    return (i_x_j[0], float(np.sum(weighted_residuals) + np.sum(np.square(dipole_residuals))))

