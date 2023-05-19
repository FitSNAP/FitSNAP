from fitsnap3lib.io.error import ExitFunc
from distutils.util import strtobool
from os import getcwd, path


class Section:
    parameters = []
    _infile_directory = None
    _outfile_directory = None
    sections = {}
    dependencies = {}
    num_desc = 0

    def __init__(self, name, config, pt, infile, args=None):
        self.name = name
        self.pt = pt
        self.infile = infile
        Section.sections[name] = self
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
        self._check_dependencies()
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
        self.pt.single_print(self.name)

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
            value = convert(fallback)
        else:
            value = convert(self._config.get(section, key, fallback=fallback))

        return value

    def get_section(self, section):
        if section not in self._config:
            return None
        return self._config.items(section)

    def check_path(self, name):
        if name == 'None':
            return None
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

    def _check_dependencies(self):
        """
        Run at end of section creation to check if any dependencies fail assertion
        """
        if self.name in Section.dependencies:
            for attribute in Section.sections[self.name].__dict__.keys():
                if attribute in Section.dependencies[self.name]:
                    for dependency in Section.dependencies[self.name][attribute]:
                        og_section, og_attribute, dependent_value = dependency
                        og_section._assert_dependency(og_attribute, self.name, attribute, dependent_value)
                    del Section.dependencies[self.name][attribute]

    def _assert_dependency(self, this, dependent_section, dependent_attribute, dependent_value=True):
        """
        Add dependency onto section attribute from separate section
        """
        if dependent_section not in Section.sections:
            if dependent_section not in Section.dependencies:
                Section.dependencies[dependent_section] = {}
                if dependent_attribute not in Section.dependencies[dependent_section]:
                    Section.dependencies[dependent_section][dependent_attribute] = []
            Section.dependencies[dependent_section][dependent_attribute].append([self, this, dependent_value])
            return
        try:
            assert Section.sections[dependent_section].__dict__[dependent_attribute] == dependent_value
        except AssertionError:
            raise AssertionError('config[{}].{} depends on config[{}].{} being equal to {}'.format(self.name,
                                                                                                   this,
                                                                                                   dependent_section,
                                                                                                   dependent_attribute,
                                                                                                   dependent_value))

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
        """ 
        Set path to input file directory.
        This is the directory that we read input from.
        If no infile is supplied (i.e. we have indict), then no need for this.
        """
        cwd = getcwd().split('/')
        path_to_file = self.infile.split('/')[:-1] if self.infile else None
        if not path_to_file:
            cls._infile_directory = ''
        elif not path.isabs(self.infile):
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
