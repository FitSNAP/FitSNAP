from fitsnap3.parallel_tools import ParallelTools
from fitsnap3.parallel_output import Output
from fitsnap3.io.error import ExitFunc
from distutils.util import strtobool
from os import getcwd


pt = ParallelTools()
output = Output()


class Section:
    parameters = []
    relative_directory = None

    def __init__(self, name, config, args=None):
        self.name = name
        self._config = config
        self._args = args
        self.allowedkeys = None
        try:
            on = self._on
        except AttributeError:
            on = "True"
        self._on = self.get_value(self.name.upper(), "on", on, "bool")
        self._on = config.has_section(self.name.upper())
        if self._on is False:
            self.delete()
            raise ExitFunc

    def delete(self):
        del self._config
        del self._args

    def _check_section(self):
        for value_name in self._config[self.name]:
            if value_name in self.allowedkeys:
                continue
            else:
                raise RuntimeError(">>> Found unmatched variable in {} section of input: {}".format(self.name,
                                                                                                    value_name))

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

    def _check_if_used(self, from_sec, sec_type, default, name=None):
        if not name:
            name = self.__class__.__name__
        if self.get_value(from_sec, sec_type, default).upper() != name.upper():
            raise UserWarning("{0} {1} section is in input, but not set as {1}".format(name, sec_type))

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
