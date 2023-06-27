from fitsnap3lib.solvers.solver import Solver
import numpy as np


try:
    from fitsnap3lib.lib.scalapack_solver.scalapack import lstsq, dummy_lstsq

    class ScaLAPACK(Solver):

        def __init__(self, name, pt, config):
            super().__init__(name, pt, config)

        def perform_fit(self):
            # This dictionary is NoneType on other subranks.
            if self.pt.get_subrank() == 0:
                if any(self.pt.fitsnap_dict['Testing']):
                    raise NotImplementedError("Testing w/ the ScaLAPACK solver is not implemented!")
            self.pt.split_by_node(self.pt.shared_arrays['w'])
            self.pt.split_by_node(self.pt.shared_arrays['a'])
            self.pt.split_by_node(self.pt.shared_arrays['b'])
            total_length = self.pt.shared_arrays['a'].get_total_length()
            node_length = self.pt.shared_arrays['a'].get_node_length()
            scraped_length = self.pt.shared_arrays['a'].get_scraped_length()
            lengths = [total_length, node_length, scraped_length]
            if self.pt.get_subrank() == 0:
                w = self.pt.shared_arrays['w'].array[:]
                aw, bw = w[:, np.newaxis] * self.pt.shared_arrays['a'].array[:], w * self.pt.shared_arrays['b'].array[:]
                # TODO:
                """
                Transpose method does not work with Quadratic SNAP (why?)
                We need to revisit this preconditioning of the linear problem, we can make this a bit more elegant.
                Since this breaks some examples this will stay as a 'secret' feature.
                Need to chat with some mathy people on how we can profile A and find good preconditioners.
                Will help when we want to try gradient based linear solvers as well.
                """
                self.fit = lstsq(aw, bw, lengths=lengths)
                if self.pt.get_subrank() == 0:
                    self.fit = self.pt.gather_to_head_node(self.fit)[0]
                # self.fit, residues, rank, s = lstsq(aw, bw, 1.0e-13)
            else:
                self.fit = dummy_lstsq()

        def _dump_a(self):
            np.savez_compressed('a.npz', a=self.pt.shared_arrays['a'].array)

        def _dump_x(self):
            np.savez_compressed('x.npz', x=self.fit)

        def _dump_b(self):
            b = self.pt.shared_arrays['a'].array @ self.fit
            np.savez_compressed('b.npz', b=b)

except ModuleNotFoundError:

    class ScaLAPACK(Solver):

        def __init__(self, name, pt, config):
            super().__init__(name, pt, config)
            raise ModuleNotFoundError("ScaLAPACK module not installed in lib")

except ImportError:

    class ScaLAPACK(Solver):

        def __init__(self, name, pt, config):
            super().__init__(name, pt, config)
            raise ImportError("ScaLAPACK module not installed in lib")
