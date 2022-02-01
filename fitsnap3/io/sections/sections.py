from ...parallel_tools import pt, output
from distutils.util import strtobool
from os import getcwd, path


class Section:
    parameters = []
    _infile_directory = None
    _outfile_directory = None

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

    def check_path(self, name):
        name = path.join(Section.get_outfile_directory(self), name)
        if self._args.overwrite is None:
            return name
        names = [name, name + '.snapparam', name + '.snapcoeff']
        for element in names:
            if not self._args.overwrite and path.exists(element):
                raise FileExistsError(f"File {element} already exists.")
        return name

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
