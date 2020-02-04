# <!----------------BEGIN-HEADER------------------------------------>
# ## FitSNAP3
# A Python Package For Training SNAP Interatomic Potentials for use in the LAMMPS molecular dynamics package
#
# _Copyright (2016) Sandia Corporation.
# Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
# the U.S. Government retains certain rights in this software.
# This software is distributed under the GNU General Public License_
# ##
#
# #### Original author:
#     Aidan P. Thompson, athomps (at) sandia (dot) gov (Sandia National Labs)
#     http://www.cs.sandia.gov/~athomps
#
# #### Key contributors (alphabetical):
#     Mary Alice Cusentino (Sandia National Labs)
#     Nicholas Lubbers (Los Alamos National Lab)
#     Maybe me ¯\_(ツ)_/¯
#     Adam Stephens (Sandia National Labs)
#     Mitchell Wood (Sandia National Labs)
#
# #### Additional authors (alphabetical):
#     Elizabeth Decolvenaere (D. E. Shaw Research)
#     Stan Moore (Sandia National Labs)
#     Steve Plimpton (Sandia National Labs)
#     Gary Saavedra (Sandia National Labs)
#     Peter Schultz (Sandia National Labs)
#     Laura Swiler (Sandia National Labs)
#
# <!-----------------END-HEADER------------------------------------->

import configparser
import argparse
from pickle import HIGHEST_PROTOCOL
from os import path
from parallel_tools import pt

DEFAULT_PROTOCOL = HIGHEST_PROTOCOL


def parse_cmdline():

    parser = argparse.ArgumentParser(prog="FitSNAP3")

    parser.add_argument("infile", action="store",
                        help="Input file with bispectrum etc. options")
    parser.add_argument("--jobs", "-j", action="store", type=int, dest="jobs",
                        default=1, help="Number of parallel LAMMPS processes")
    parser.add_argument("--verbose", "-v", action="store_true", dest="verbose",
                        default=False, help="Show more detailed information about processing")
    parser.add_argument("--lammpslog", "-l", action="store_true", dest="lammpslog",
                        help="Write logs from LAMMPS. Logs will appear in current working directory.")
    parser.add_argument("--relative", "-r", action="store_true", dest="relative",
                        help="""Put output files in the directory of INFILE. If this flag
                         is not not present, the files are stored in the
                        current working directory.""")
    parser.add_argument("--nofit", "-nf", action="store_false", dest="perform_fit",
                        help="Don't perform fit, just compute bispectrum data.")
    parser.add_argument("--overwrite", action="store_true", dest="overwrite",
                        help="Allow overwriting existing files")
    parser.add_argument("--lammps_noexceptions", action="store_true",
                        help="Allow LAMMPS compiled without C++ exceptions handling enabled")
    parser.add_argument("--keyword", "-k", nargs=3, metavar=("GROUP", "NAME", "VALUE"),
                        action="append", dest="keyword_replacements",
                        help="""Replace or add input keyword group GROUP, key NAME,
                        with value VALUE. Type carefully; a misspelled key name or value
                        may be silently ignored.""")

    args = parser.parse_args()

    return args


def parse_config(args):

    config = configparser.ConfigParser(inline_comment_prefixes='#')
    config.read(args.infile)

    vprint = pt.single_print if args.verbose else lambda *arguments, **kwargs: None
    if args.keyword_replacements:
        for kwg, kwn, kwv in args.keyword_replacements:
            if kwg not in config:
                raise ValueError(f"{kwg} is not a valid keyword group")
            vprint(f"Substituting {kwg}:{kwn}={kwv}")
            config[kwg][kwn] = kwv

    set_file_names(args, config)

    return config


def set_file_names(args, config):

    """Gets the outfile names and opening for fitsnap.
    If not allow_overwrite, check to make sure the files don't already exist and raise an error.
    """

    if args.relative:
        base_path, _ = path.split(args.infile)
    else:
        base_path = None

    config_file = config.get("OUTFILE", "configs", fallback="fitsnap_configs.pkl.gz")
    config_file = check_path(base_path, config_file, args.overwrite)
    config.set("OUTFILE", "configs", config_file)
    metric_file = config.get("OUTFILE", "metrics", fallback="fitsnap_metrics.csv")
    metric_file = check_path(base_path, metric_file, args.overwrite)
    config.set("OUTFILE", "metrics", metric_file)
    potential_name = config.get("OUTFILE", "potential", fallback="fitsnap_potential")
    potential_name = check_path(base_path, potential_name, args.overwrite)
    config.set("OUTFILE", "potential", potential_name)
    group_file = config.get("OUTFILE", "potential", fallback="grouplist.in")
    group_file = check_path(base_path, group_file)
    config.set("OUTFILE", "potential", group_file)


def check_path(base_path, name, overwrite=None):
    if base_path is not None:
        name = path.join(base_path, name)
    else:
        name = name
    if overwrite is None:
        return name
    names = [name, name + '.snapparam', name + '.snapcoeff']
    for element in names:
        if overwrite and path.exists(element):
            raise FileExistsError(f"File {element} already exists.")
    return name

