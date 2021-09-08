import configparser
import argparse
from pickle import HIGHEST_PROTOCOL
from ..io.sections.section_factory import new_section
from ..parallel_tools import ParallelTools
from ..parallel_output import Output
from os import path, listdir


pt = ParallelTools()
output = Output()


class Singleton(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if (
            kwargs is not None
            and "config" in kwargs.keys()
            and kwargs["config"] is not None
        ):
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


class Config(metaclass=Singleton):

    def __init__(self, arguments_lst=None):
        self.default_protocol = HIGHEST_PROTOCOL
        self.args = None
        self.parse_cmdline(arguments_lst)
        self.sections = {}
        self.parse_config()

    def parse_cmdline(self, arguments_lst=None):
        parser = argparse.ArgumentParser(prog="FitSNAP3")

        parser.add_argument("infile", action="store",
                            help="Input file with bispectrum etc. options")
        # Not Implemented
        parser.add_argument("--verbose", "-v", action="store_true", dest="verbose",
                            default=False, help="Show more detailed information about processing")
        parser.add_argument("--lammpslog", "-l", action="store_true", dest="lammpslog",
                            help="Write logs from LAMMPS. Logs will appear in current working directory.")
        parser.add_argument("--printlammps", "-pl", action="store_true", dest="printlammps",
                            help="Print all lammps commands")
        parser.add_argument("--relative", "-r", action="store_true", dest="relative",
                            help="""Put output files in the directory of INFILE. If this flag
                             is not not present, the files are stored in the
                            current working directory.""")
        # Not Implemented
        parser.add_argument("--nofit", "-nf", action="store_false", dest="perform_fit",
                            help="Don't perform fit, just compute bispectrum data.")

        parser.add_argument("--overwrite", action="store_true", dest="overwrite",
                            help="Allow overwriting existing files")
        # Not Implemented
        parser.add_argument("--lammps_noexceptions", action="store_true",
                            help="Allow LAMMPS compiled without C++ exceptions handling enabled")
        # Not Implemented
        parser.add_argument("--keyword", "-k", nargs=3, metavar=("GROUP", "NAME", "VALUE"),
                            action="append", dest="keyword_replacements",
                            help="""Replace or add input keyword group GROUP, key NAME,
                            with value VALUE. Type carefully; a misspelled key name or value
                            may be silently ignored.""")
        parser.add_argument("--screen", "-sc", action="store_false", dest="screen",
                            help="Print to screen")
        parser.add_argument("--nscreen", "-ns", action="store_true", dest="nscreen",
                            help="Print each nodes screen")
        parser.add_argument("--pscreen", "-ps", action="store_true", dest="pscreen",
                            help="Print each processors screen")
        parser.add_argument("--log", action="store", dest="log",
                            default=None, help="Write fitsnap log to this file")

        self.args = parser.parse_args(arguments_lst)

    def parse_config(self):

        tmp_config = configparser.ConfigParser(inline_comment_prefixes='#')
        tmp_config.optionxform = str
        tmp_config.read(self.args.infile)

        vprint = output.screen if self.args.verbose else lambda *arguments, **kwargs: None
        if self.args.keyword_replacements:
            for kwg, kwn, kwv in self.args.keyword_replacements:
                if kwg not in tmp_config:
                    raise ValueError(f"{kwg} is not a valid keyword group")
                vprint(f"Substituting {kwg}:{kwn}={kwv}")
                tmp_config[kwg][kwn] = kwv

        self.set_sections(tmp_config)

    def set_sections(self, tmp_config):
        location = '/' + '/'.join(path.abspath(__file__).split("/")[:-1]) + '/sections'
        temp = {file: None for file in listdir(location)}
        allowedkeys = []
        for file in temp:
            if file.split('.')[-1] != 'py' or file == 'sections.py' or file == 'section_factory.py' or file == '__init__.py':
                continue
            else:
                section = '.'.join(file.split('.')[:-1]).upper()
                if section == "TEMPLATE":
                    section = "DEFAULT"
                if section == "BASIC_CALCULATOR":
                    section = "BASIC"
                self.sections[section] = new_section(section, tmp_config, self.args)
                allowedkeys.append(section)
        for property_name in tmp_config.keys():
            if property_name in allowedkeys: continue
            else: pt.single_print(">>> Found unmatched section in input: ",property_name)
        del temp
