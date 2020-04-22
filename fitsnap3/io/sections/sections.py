from fitsnap3.parallel_tools import pt
from distutils.util import strtobool


class Section:
    parameters = []

    def __init__(self, name, config, args):
        self.name = name
        self._config = config
        self._args = args
        try:
            on = self._on
        except AttributeError:
            on = "True"
        self._on = self.get_value(self.name.upper(), "on", on, "bool")
        if self._on is False:
            self.delete()
            return

    def delete(self):
        del self._config
        del self._args

    def print_name(self):
        pt.single_print(self.name)

    def get_value(self, section, key, fallback, interpreter="str"):
        if self._args == "verbose" and section.lower() == self.name.lower():
            Section.add_parameter(section, key, fallback, interpreter)
        if interpreter == "str" or interpreter == "string":
            convert = str
        elif interpreter == "bool":
            convert = strtobool
        elif interpreter == "float":
            convert = float
        elif interpreter == "int" or interpreter == "integer":
            convert = int
        else:
            raise ValueError("{} is not an implemented interpreter.")

        if section not in self._config:
            return convert(fallback)
        else:
            return convert(self._config.get(section, key, fallback=fallback))

    @classmethod
    def add_parameter(cls, section, key, fallback, interpreter):
        cls.parameters.append([section, key, fallback, interpreter])
