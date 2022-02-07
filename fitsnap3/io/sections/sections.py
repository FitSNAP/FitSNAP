from ...parallel_tools import output, pt
from ..error import ExitFunc
from distutils.util import strtobool
from os import getcwd, path


class Section:
    parameters = []
    _infile_directory = None
    _outfile_directory = None

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

    def check_path(self, name):
        name = path.join(Section.get_outfile_directory(self), name)
        if self._args.overwrite is None:
            return name
        names = [name, name + '.snapparam', name + '.snapcoeff']
        for element in names:
            if not self._args.overwrite and path.exists(element):
                raise FileExistsError(f"File {element} already exists.")
        return name

    def _check_if_used(self, from_sec, sec_type, default, name=None):
        if not name:
            name = self.__class__.__name__
        if self.get_value(from_sec, sec_type, default).upper() != name.upper():
            raise UserWarning("{0} {1} section is in input, but not set as {1}".format(name, sec_type))

    @classmethod
    def add_parameter(cls, section, key, fallback, interpreter):
        cls.parameters.append([section, key, fallback, interpreter])

    @classmethod
    def get_infile_directory(cls, self):
        if cls._infile_directory is None:
            cls._set_infile_directory(self)
        return cls._infile_directory

    @classmethod
    def get_outfile_directory(cls, self):
        if cls._outfile_directory is None:
            cls._set_outfile_directory(self)
        return cls._outfile_directory

    @classmethod
    def _set_infile_directory(cls, self):
        """ Set path to input file directory """
        cwd = getcwd().split('/')
        path_to_file = self._args.infile.split('/')[:-1]
        if not path_to_file:
            cls._infile_directory = ''
        elif not path.isabs(self._args.infile):
            cls._infile_directory = '/'.join(path_to_file)
        else:
            count = 0
            while path_to_file[count] == cwd[count]:
                count += 1
            cwd = (['..'] * (len(cwd)-count)) + path_to_file[count:]
            cls._infile_directory = '/'.join(cwd)

    @classmethod
    def _set_outfile_directory(cls, self):
        """ Set current working directory, if args.relative == True: cwd = input file directory else: cwd = cd ./ """
        if self._args.relative:
            cls._outfile_directory = cls.get_infile_directory(self)
        else:
            cls._outfile_directory = getcwd()
