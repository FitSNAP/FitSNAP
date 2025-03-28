from fitsnap3lib.solvers.solver import Solver
from fitsnap3lib.calculators.lammps_reaxff import LammpsReaxff
import cma, itertools, functools
import numpy as np
from pprint import pprint

# ------------------------------------------------------------------------------------------------

def _loss_function(x_arrays):

    return reaxff_calculator.process_configs_with_values(x_arrays)

# ------------------------------------------------------------------------------------------------

class CMAES(Solver):

    # --------------------------------------------------------------------------------------------

    def __init__(self, name, pt, config):

        super().__init__(name, pt, config, linear=False)
        self.popsize = self.config.sections['SOLVER'].popsize
        self.sigma = self.config.sections['SOLVER'].sigma
        self.reaxff_io = self.config.sections['REAXFF']

    # --------------------------------------------------------------------------------------------

    def _parallel_loss_function(self, x_arrays):

        # print(f"*** rank {self.pt._rank} ok 2a")

        np.set_printoptions(precision=4, linewidth=2000)
        
        if self.pt.stubs==1:
            answer = _loss_function(x_arrays)
        else:
            futures = [self.executor.submit(_loss_function, x_arrays) for _ in self.range_workers]
            results = np.vstack([np.nan_to_num(f.result(), nan=8e8) for f in futures])
            answer = np.sum(results, axis=0)
            #print(f"*** rank {self.pt._rank} results {results}")

        print(f"*** rank {self.pt._rank} answer {answer}")
        return answer.tolist()

    # --------------------------------------------------------------------------------------------

    def perform_fit(self, fs):

        # "Avoid global variables that are declared outside class methods or attributes"
        # no way around this sorry
        global reaxff_calculator
        reaxff_calculator = fs.calculator
        x0 = self.reaxff_io.values
        #print( x0 )
        self.output = fs.output
        self.range_workers = range(1, self.pt.get_size())

        import warnings
        warnings.simplefilter("ignore", category=UserWarning)

        options={
          'popsize': self.popsize, 'seed': 12345, #'maxiter': 1,
          'bounds': list(np.transpose(self.config.sections['REAXFF'].parameter_bounds))
        }

        if self.pt.stubs==0:
            # SAFER TO USE *MPICommExecutor* INSTEAD OF *MPIPoolExecutor*
            # [https://mpi4py.readthedocs.io/en/stable/mpi4py.futures.html#mpicommexecutor]
            from mpi4py import MPI
            from mpi4py.futures import MPICommExecutor

            with MPICommExecutor(MPI.COMM_WORLD, root=0) as self.executor:
                if self.executor is not None:
                    _, es = cma.fmin2( None, x0, self.sigma, callback=self._log_progress,
                        parallel_objective=self._parallel_loss_function, options=options)
        else:
            _, es = cma.fmin2( None, x0, self.sigma,
                parallel_objective=self._parallel_loss_function, options=options)

        if self.pt._rank == 0:
            self.fit = self.reaxff_io.change_parameters_string(es.best.x)
            self._log_best(es)

    # --------------------------------------------------------------------------------------------

    def _log_progress(self, es):

        if es.countiter % 1 == 0:
            self._log_best(es)

        if es.countiter % 10 == 0:
            current_fit = self.reaxff_io.change_parameters_string(es.best.x)
            self.output.output(current_fit, None)


    # --------------------------------------------------------------------------------------------

    def _log_best(self, es):

        print("--------------------------------------------------------------------")
        print(self.config.sections['CALCULATOR'].charge_fix)
        print("PARAMETER_NAME        INITIAL  LOWER_BOUND       NOW     UPPER_BOUND")
        for p, x0i, xbi in zip(self.reaxff_io.parameter_names, es.x0, es.best.x):
            p_bounds = self.reaxff_io.bounds[p.split('.')[-1]]
            print(f"{p:<19} {x0i: > 9.4f}  [  {p_bounds[0]: > 6.2f} {xbi: > 13.8f} {p_bounds[1]: > 8.2f}  ]")
        print(f"------------------------ {es.countiter:<7} {es.best.f:9.4g} ------------------------")

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


