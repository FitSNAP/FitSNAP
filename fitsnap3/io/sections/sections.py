from fitsnap3.parallel_tools import pt


class Section:

    def __init__(self, name, config, args):
        self.name = name
        self._config = config
        self._args = args

    def delete(self):
        del self._config
        del self._args

    def print_name(self):
        pt.single_print(self.name)

