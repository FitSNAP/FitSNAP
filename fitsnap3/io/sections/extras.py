from .sections import Section


class Extras(Section):

    def __init__(self, name, config, args):
        super().__init__(name, config, args)
        self.multinode_testing = self.get_value("EXTRAS", "multinode_testing", "0", "bool")
        self.apply_transpose = self.get_value("EXTRAS", "apply_transpose", "0", "bool")
        self.only_test = self.get_value("EXTRAS", "only_test", "0", "bool")
        self.dump_a = self.get_value("EXTRAS", "dump_descriptors", "0", "bool")
        self.dump_b = self.get_value("EXTRAS", "dump_truth", "0", "bool")
        self.dump_w = self.get_value("EXTRAS", "dump_weights", "0", "bool")
        self.delete()
