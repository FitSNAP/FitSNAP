import configparser
import argparse
import sys
from pickle import HIGHEST_PROTOCOL
from fitsnap3lib.io.sections.section_factory import new_section
from pathlib import Path
import random


class Config():
    """ 
    Class for storing input settings in a `config` instance. The `config` instance is first created 
    in `io/output.py`. If given a path to an input script, we use Python's native ConfigParser 
    to parse the settings. If given a nested dictionary, the sections are determined from the 
    first keys and specific settings from the nested keys.

    Args:
        pt: A ParallelTools instance.
        input: Optional input can either be a filename or a dictionary.
        arguments_lst: List of args that can be supplied at the command line.

    Attributes:
        infile: String for optional input filename. Defaults to None.
        indict: Dictionary for optional input dictionary of settings, to replace input file. Defaults 
            to None.
        
    """

    def __init__(self, pt, input=None, arguments_lst: list = []):
        self.pt = pt
        self.input = input
        # Input file (infile) and dictionary (indict) set to None by default and get set in 
        # parse_config.
        self.infile = None
        self.indict = None
        self.default_protocol = HIGHEST_PROTOCOL
        self.args = None
        self.parse_cmdline(arguments_lst=arguments_lst)
        self.sections = {}
        self.parse_config()

        # Generate random 128 bit hash to identify this fit on rank 0.
        if self.pt._rank == 0:
            self.hash = f"{random.getrandbits(128):032x}"
        else:
            self.hash = None

    def parse_cmdline(self, arguments_lst=[]):
        """ Parse command line args. """
        parser = argparse.ArgumentParser(prog="fitsnap3")
        if (self.input is None):
            parser.add_argument("infile", action="store",
                                help="Input file with bispectrum etc. options")

        # Optional args.
        parser.add_argument("--lammpslog", "-l", action="store_true", dest="lammpslog",
                            help="Write logs from LAMMPS. Logs will appear in current working directory.")
        parser.add_argument("--nofit", "-nf", action="store_false", dest="perform_fit",
                            help="Don't perform fit, just compute bispectrum data.")
        parser.add_argument("--overwrite", action="store_true", dest="overwrite",
                            help="Allow overwriting existing files")
        parser.add_argument("--verbose", "-v", action="store_true", dest="verbose",
                            default=False, help="Show more detailed information about processing")
        parser.add_argument("--tarball", "-tb", action="store_true", dest="tarball",
                            default=False, help="Package SNAP fit files into a hashed tarball")
        parser.add_argument("--relative", "-r", action="store_true", dest="relative",
                            help='''Put output files in the directory of INFILE. If this flag
                             is not not present, the files are stored in the
                            current working directory.''')
        parser.add_argument("--keyword", "-k", nargs=3, metavar=("GROUP", "NAME", "VALUE"),
                            action="append", dest="keyword_replacements",
                            help='''Replace or add input keyword group GROUP, key NAME,
                            with value VALUE. Type carefully; a misspelled key name or value
                            may be silently ignored.''')
        # The following have questionable importance.
        parser.add_argument("--screen", "-sc", action="store_false", dest="screen",
                            help="Print fitsnap output to screen.")
        parser.add_argument("--nscreen", "-ns", action="store_true", dest="nscreen",
                            help="Print fitsnap output to each nodes screen.")
        parser.add_argument("--pscreen", "-ps", action="store_true", dest="pscreen",
                            help="Print fitsnap output to each processors screen.")
        parser.add_argument("--log", action="store", dest="log",
                            default=None, help="Write fitsnap log to this file.")
        parser.add_argument("--screen2file", "-s2f", action="store", dest="screen2file",
                            default=None, help="Print screen to a file")
        # Not Implemented.
        """
        parser.add_argument("--lammps_noexceptions", action="store_true",
                            help="Allow LAMMPS compiled without C++ exceptions handling enabled")
        parser.add_argument("--printlammps", "-pl", action="store_true", dest="printlammps",
                            help="Print all lammps commands")
        """

        # Check if building docs, in which case we revert to using Ta Linear example.

        for item in sys.argv:
            if (item=="build" or item=="html" or item=="source"):
                # We're building docs in this case.
                arguments_lst = arguments_lst=["../examples/Ta_Linear_JCP2014/Ta-example-nodump.in", "--overwrite"]
        self.args = parser.parse_args(arguments_lst)

    def parse_config(self):
        tmp_config = configparser.ConfigParser(inline_comment_prefixes='#')
        tmp_config.optionxform = str
        if self.input is not None:
            if (isinstance(self.input, str)):
                self.infile = self.input
            elif (isinstance(self.input, dict)):
                self.indict = self.input
        else:
            if not Path(self.args.infile).is_file():
                raise FileNotFoundError("Input file not found")
            self.infile = self.args.infile

        if (self.infile is not None):
            # We have an input file.
            tmp_config.read(self.infile)
            infile_folder = str(Path(self.infile).parent.absolute())
            file_name = self.infile.split('/')[-1]
            if not Path(infile_folder+'/'+file_name).is_file():
                raise RuntimeError("Input file {} not found in {}", file_name, infile_folder)

            #vprint = output.screen if self.args.verbose else lambda *arguments, **kwargs: None
            # This adds keyword replacements to the config.
            if self.args.keyword_replacements:
                for kwg, kwn, kwv in self.args.keyword_replacements:
                    if kwg not in tmp_config:
                        raise ValueError(f"{kwg} is not a valid keyword group")
                    tmp_config[kwg][kwn] = kwv

        elif (self.indict is not None):
            # We have an input dictionary  instead of a file.
            for key1, data1 in self.indict.items():
                tmp_config[key1] = {}
                for key2, data2 in data1.items():
                    tmp_config[key1]["{}".format(key2)] = str(data2)
            # Default missing sections to empty dicts which will prompt default values.
            names = ["ESHIFT", "EXTRAS", "GROUPS", "MEMORY"]
            for name in names:
                if name not in tmp_config:
                    tmp_config[name] = {}

        # Make sections based on input settings.
        self.set_sections(tmp_config)

    def set_sections(self, tmp_config):
        sections = tmp_config.sections()
        for section in sections:
            if section == "TEMPLATE":
                section = "DEFAULT"
            if section == "BASIC_CALCULATOR":
                section = "BASIC"
            self.sections[section] = new_section(section, tmp_config, self.pt, self.infile, self.args)
