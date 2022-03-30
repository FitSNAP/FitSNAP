from .sections import Section
from ...parallel_tools import pt


class Extras(Section):

    def __init__(self, name, config, args):
        super().__init__(name, config, args)
        self.allowedkeys = ['multinode_testing', 'apply_transpose', 'only_test', 'dump_descriptors', 'dump_truth',
                            'dump_weights', 'dump_dataframe']
        self._check_section()

        self.multinode_testing = self.get_value("EXTRAS", "multinode_testing", "0", "bool")
        self.apply_transpose = self.get_value("EXTRAS", "apply_transpose", "0", "bool")
        self.only_test = self.get_value("EXTRAS", "only_test", "0", "bool")
        self.dump_a = self.get_value("EXTRAS", "dump_descriptors", "0", "bool")
        self.dump_b = self.get_value("EXTRAS", "dump_truth", "0", "bool")
        self.dump_w = self.get_value("EXTRAS", "dump_weights", "0", "bool")
        self.dump_dataframe = self.get_value("EXTRAS", "dump_dataframe", "0", "bool")
        self.descriptor_file = \
            self.check_path(self.get_value("OUTFILE", "descriptors", "Descriptors.npy"))
        self.truth_file = \
            self.check_path(self.get_value("OUTFILE", "truth", "Truth-Ref.npy"))
        self.weights_file = \
            self.check_path(self.get_value("OUTFILE", "weights", "Weights.npy"))
        self.dataframe_file = \
            self.check_path(self.get_value("OUTFILE", "dataframe", "FitSNAP.df"))
        self.delete()
