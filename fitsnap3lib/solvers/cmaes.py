from fitsnap3lib.solvers.solver import Solver
import cma

"""Methods you may or must override in new solvers"""

class CMAES(Solver):

    def __init__(self, name, pt, config):
        super().__init__(name, pt, config)
        self.sigma = self.config.sections['CMAES'].sigma

    def loss_function(x, *args):

        print(f'x={x}')
        print(f'args={args}')


    def perform_fit(self):
        """
        Base class function for performing a fit.
        """

        x0 = [p['value'] for p in eval(self.config.sections['REAXFF'].parameters)]

        print(x0)

        x, es = cma.fmin2(self.loss_function, x0, self.sigma,
          options={'maxiter': 3,'maxfevals': 3,'popsize':2})
