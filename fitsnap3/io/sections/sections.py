from fitsnap3.parallel_tools import pt, output
from distutils.util import strtobool
from os import getcwd


class Section:
    parameters = []
    relative_directory = None

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
        output.screen(self.name)

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

    def get_section(self, section):
        if section not in self._config:
            return None
        return self._config.items(section)

    @classmethod
    def add_parameter(cls, section, key, fallback, interpreter):
        cls.parameters.append([section, key, fallback, interpreter])

    @classmethod
    def _get_relative_directory(cls, self):
        if cls.relative_directory is None:
            cls._set_relative_directory(self)
        return cls.relative_directory

    @classmethod
    def _set_relative_directory(cls, self):
        paths = getcwd().split('/') + self._args.infile.split('/')[:-1]
        relative_directory = ''
        for directory in paths[:-1]:
            relative_directory += directory + '/'
        relative_directory += paths[-1]
        cls.relative_directory = relative_directory
