# <!----------------BEGIN-HEADER------------------------------------>
# ## FitSNAP3
# A Python Package For Training SNAP Interatomic Potentials for use in the LAMMPS molecular dynamics package
#
# _Copyright (2016) Sandia Corporation. Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains certain rights in this software. This software is distributed under the GNU General Public License_
# ##
#
# #### Original author:
#     Aidan P. Thompson, athomps (at) sandia (dot) gov (Sandia National Labs)
#     http://www.cs.sandia.gov/~athomps
#
# #### Key contributors (alphabetical):
#     Mary Alice Cusentino (Sandia National Labs)
#     Nicholas Lubbers (Los Alamos National Lab)
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
import sys
import os
import argparse
import configparser
import gzip
import pickle
import datetime

import contextlib

from distutils.util import strtobool

from . import bispecopt, scrape, deploy, serialize, linearfit

DEFAULT_PROTOCOL = pickle.HIGHEST_PROTOCOL
DEFAULT_CONFIGNAME = "fitsnap_configs.pkl.gz"
DEFAULT_POTNAME = "fitsnap_potential"
DEFAULT_METNAME =  "fitsnap_metrics.csv"

def parse_cmdline():

    parser = argparse.ArgumentParser(prog="FitSNAP3")

    parser.add_argument("infile", action="store",
                        help="Input file with bispectrum etc. options")
    parser.add_argument("--jobs","-j",action="store",type=int,dest="jobs",
                        default=1, help="Number of parallel LAMMPS processes")
    parser.add_argument("--verbose", "-v", action="store_true", dest="verbose",
                        default=False, help="Show more detailed information about processing")
    parser.add_argument("--lammpslog", "-l", action="store_true", dest="lammpslog",
                        help="Write logs from LAMMPS. Logs will appear in current working directory.")
    parser.add_argument("--relative", "-r", action="store_true",dest="relative",
                        help="""Put output files in the directory of INFILE. If this flag
                         is not not present, the files are stored in the
                        current working directory.""")
    parser.add_argument("--nofit","-nf",action="store_false",dest="perform_fit",
                        help="Don't perform fit, just compute bispectrum data.")
    parser.add_argument("--overwrite",action="store_true",dest="overwrite",
                        help="Allow overwriting existing files")
    parser.add_argument("--lammps_noexceptions",action="store_true",
                        help="Allow LAMMPS compiled without C++ exceptions handling enabled")
    parser.add_argument("--keyword","-k",nargs=3,metavar=("GROUP","NAME","VALUE"),
                        action="append",dest="keyword_replacements",
                        help="""Replace or add input keyword group GROUP, key NAME,
                        with value VALUE. Type carefully; a misspelled key name or value
                        may be silently ignored.""")
    parser.add_argument("--mpi", action="store_true", dest="mpi",
                        help="Use MPI for parallelizing lammps (control number of workers with --jobs argument)")

    args = parser.parse_args()

    return args


##### Filename setup  functions ####
def defaulted_name(name,default):
    "Returns name if the string is not boolable. If string falsy, returns None, if string true, returns default"
    try: return default if strtobool(name) else None
    except ValueError: return name # String couldn't be read
    except AttributeError: return None # Got non-string such as NoneType

def get_filenames(base_path,allow_exists,configs=None,metrics=None,potential=None):
    """Gets the outfile names and opening for fitsnap.
    If not allow_overwrite, check to make sure the files don't already exist and raise an error.
    """
    configfile = defaulted_name(configs,DEFAULT_CONFIGNAME)
    metricfile = defaulted_name(metrics,DEFAULT_METNAME)
    potentialname = defaulted_name(potential,DEFAULT_POTNAME)

    fnameinfo ={
        "config":(configfile,'b'),
        "metrics":(metricfile,'t'),
        "snapparam":(potentialname and potentialname + '.snapparam','t'),  # Keeps None if already None
        "snapcoeff":(potentialname and potentialname + '.snapcoeff','t'),
    }

    for key,(name,mode) in fnameinfo.copy().items():
        if base_path is not None and name is not None:
            name = os.path.join(base_path,name)
        if allow_exists:
            mode = 'w'+mode
        else:
            if os.path.exists(name): raise FileExistsError(f"File {name} already exists.")
            mode = 'x'+mode
        fnameinfo[key]=name,mode
    return fnameinfo

class FakeFile():
    """Good enough to fool pandas!"""
    def write(self, *args, **kwargs): return
    def __iter__(self): return None

@contextlib.contextmanager
def optional_write(file, mode, *args, openfn=None, **kwargs):
    # Note: this is suboptimal in that whatever write operations
    # are still performed ; this is negligible compared to the computation, at the moment.
    """If file is None, yields a dummy file object."""
    if file is None: return#if file is None: yield FakeFile()
    else:
        if openfn is None:
            openfn = gzip.open if file.endswith('.gz') else open
        with openfn(file,mode,*args,**kwargs) as open_file:
            with printdoing(f'Writing file "{file}"'):
                yield open_file


##### A few lines for printing convenience ####
def print_error_summary(error_metrics):
    es = error_metrics
    es = es[es.index.get_level_values("Weighting") == "Unweighted"]
    es.index = es.index.droplevel("Weighting")
    es = es.unstack("Subsystem")
    print("Summary of fit MAE errors (Unweighted):")
    print(es["mae"].to_string())
    print("Summary of fit R^2 values (Unweighted):")
    print(es["rsq"].to_string())

@contextlib.contextmanager
def printdoing(msg, sep='',flush=True,end='', **kwargs):
    """Let the user know that the code is performing MSG. Also makes code more self-documenting."""
    start_time= datetime.datetime.now()
    print(msg, '...',sep=sep,flush=flush,end=end,**kwargs); yield;
    print(f"Done! ({datetime.datetime.now()-start_time})")

def main():
    args = parse_cmdline()
    vprint = print if args.verbose else lambda *args,**kwargs:None

    cp = configparser.ConfigParser(inline_comment_prefixes='#')
    excp = configparser.ConfigParser(inline_comment_prefixes='#')
    excp._interpolation = configparser.ExtendedInterpolation()
    cp.read(args.infile)
    excp.read(args.infile)
    if args.keyword_replacements:
        for kwg, kwn, kwv in args.keyword_replacements:
            if kwg not in cp: raise ValueError(f"{kwg} is not a valid keyword group")
            vprint(f"Substituting {kwg}:{kwn}={kwv}")
            cp[kwg][kwn] = kwv

    base_path, _ = os.path.split(args.infile)

    fnameinfo = get_filenames(
                        base_path = base_path if args.relative else None,
                        allow_exists = args.overwrite,
                        **cp["OUTFILE"]
    )

    # Check that lammps can open, and whether it has exception handling enabled.
    deploy.check_lammps(args.lammps_noexceptions)

    bispec_options = bispecopt.read_bispec_options(cp["BISPECTRUM"],cp["MODEL"],cp["REFERENCE"])
    vprint("Bispectrum options:") #Sorted.
    for k,v in sorted(bispec_options.items(), key=lambda kv:kv[0]):
        if k == "bnames": continue # Informing the user about the individual bispectrum names is not helpful.
        vprint(f"{k:>16} : {v}")

    # Set fallback values if not found in input file
    bispec_options["BOLTZT"] = cp.get("BISPECTRUM","BOLTZT",fallback='10000')
    bispec_options["compute_testerrs"] = strtobool(cp.get("MODEL","compute_testerrs",fallback=0))
    bispec_options["units"] = cp.get("REFERENCE","units",fallback='metal').lower()
    bispec_options["atom_style"] = cp.get("REFERENCE","atom_style",fallback='atomic').lower()

    lmp_pairdecl = []
    lmp_pairdecl.append("pair_style "+excp.get("REFERENCE","pair_style",fallback='zero 10.0'))
    for name, value in cp.items("REFERENCE"):
        if not name.find("pair_coeff") : lmp_pairdecl.append("pair_coeff "+value)
    if "pair_coeff" in lmp_pairdecl : lmp_pairdecl.append("pair_coeff * * ")

    bispec_options["pair_func"] = lmp_pairdecl
    bispec_options["verbosity"] = args.verbose
    #print(bispec_options)
    #Get group info
    group_file = cp.get("PATH","groupFile",fallback='grouplist.in')
    group_file = os.path.join(base_path, group_file)
    group_table = scrape.read_groups(group_file)
    vprint("Group table:")
    vprint(group_table)

    with printdoing("Scraping Configurations",end='\n'):
        json_directory = cp["PATH"]["jsonPath"]
        json_directory = os.path.join(base_path, json_directory)
        configs,style_info = scrape.read_configs(json_directory, group_table,bispec_options)

    with printdoing("Computing bispectrum data",end='\n'):
        configs = deploy.compute_bispec_datasets(configs,
                                                        bispec_options,
                                                        n_procs=args.jobs,
                                                        mpi=args.mpi,
                                                        log=args.lammpslog)
        configs = serialize.pack(configs)

    # If performing fit, do this before saving the computed data.
    # If the fit fails, we still serialize the computed data before dealing with thee error.
    try:
        if args.perform_fit:
            with printdoing("Assembling linear system"):
                offset = not bispec_options["bzeroflag"]
                subsystems = (True,True,True) if bispec_options["compute_dbvb"] else (True,False,False)
                A, b, w = linearfit.make_Abw(configs=configs, offset=offset, return_subsystems=False,subsystems=subsystems)

            with printdoing("Performing fit"):
                solver = linearfit.get_solver_fn(**cp["MODEL"])
                fit_coeffs, solver_info = linearfit.solve_linear_snap(A,b,w, solver=solver, offset=offset)

            with printdoing("Measuring errors"):
                error_metrics = linearfit.group_errors(fit_coeffs,configs,bispec_options,subsystems=subsystems)
                configs.update(linearfit.get_residuals(fit_coeffs,configs,subsystems=subsystems))
            print("Units for reported errors:")
            if bispec_options["units"]=="real":
                print("Energy(kcal/mol), Force(kcal/mol-Angstrom), Pressure(atm)")
            if bispec_options["units"]=="metal":
                print("Energy(eV/atom), Force(eV/Angstrom), Pressure(bar)")

            if args.verbose:
                print_error_summary(error_metrics)

            with optional_write(*fnameinfo["metrics"]) as file:
                error_metrics.to_csv(file)
            with optional_write(*fnameinfo["snapparam"]) as file:
                file.write(serialize.to_param_string(**bispec_options))
            with optional_write(*fnameinfo["snapcoeff"]) as file:
                file.write(serialize.to_coeff_string(fit_coeffs, bispec_options))

    except Exception as e:
        print("Fitting interrupted! Attempting to save computed data.",file=sys.stderr)
        raise e
    finally:
        save_dict = {"configs": configs,"bispec_otions": bispec_options,"styles": style_info}
#        with optional_write(*fnameinfo["config"]) as pfile:
#            pickle.dump(save_dict, pfile, protocol=pickle.DEFAULT_PROTOCOL)

    return

if __name__ == "__main__":
    main()
