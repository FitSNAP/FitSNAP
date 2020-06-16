import configparser
import argparse
from pickle import HIGHEST_PROTOCOL
from fitsnap3.io.sections.section_factory import new_section
from fitsnap3.parallel_tools import pt, output
from os import path, listdir


class Config:

    def __init__(self):
        self.default_protocol = HIGHEST_PROTOCOL
        self.args = None
        self.parse_cmdline()
        self.sections = {}
        self.parse_config()
        self._reset_chemflag()

    def parse_cmdline(self):
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
        # Not Implemented
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

        self.args = parser.parse_args()

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
        for file in temp:
            if file.split('.')[-1] != 'py' or file == 'sections.py' or file == 'section_factory.py':
                continue
            else:
                section = '.'.join(file.split('.')[:-1]).upper()
                if section == "TEMPLATE":
                    section = "DEFAULT"
                self.sections[section] = new_section(section, tmp_config, self.args)
        del temp

    def _reset_chemflag(self):
        if self.sections["CALCULATOR"].chemflag != 0:
            chemflag = "{}".format(self.sections['BISPECTRUM'].numtypes)
            for element in self.sections["BISPECTRUM"].type_mapping:
                element_type = self.sections["BISPECTRUM"].type_mapping[element]
                chemflag += " {}".format(element_type - 1)
            self.sections["CALCULATOR"].chemflag = "{}".format(chemflag)


if __name__ == "fitsnap3.io.input":
    config = Config()
