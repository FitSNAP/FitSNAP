from .sections import Section
from ...parallel_tools import pt


class Extras(Section):

    def __init__(self, name, config, args):
        super().__init__(name, config, args)
        allowedkeys = ['multinode_testing','apply_transpose','only_test','dump_descriptors','dump_truth','dump_weights']
        for value_name in config['EXTRAS']:
            if value_name in allowedkeys: continue
            else: pt.single_print(">>> Found unmatched variable in EXTRAS section of input: ",value_name)
        self.multinode_testing = self.get_value("EXTRAS", "multinode_testing", "0", "bool")
        self.apply_transpose = self.get_value("EXTRAS", "apply_transpose", "0", "bool")
        self.only_test = self.get_value("EXTRAS", "only_test", "0", "bool")
        self.dump_a = self.get_value("EXTRAS", "dump_descriptors", "0", "bool")
        self.dump_b = self.get_value("EXTRAS", "dump_truth", "0", "bool")
        self.dump_w = self.get_value("EXTRAS", "dump_weights", "0", "bool")
        self.delete()
