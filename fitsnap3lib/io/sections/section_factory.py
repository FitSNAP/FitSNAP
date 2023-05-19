from fitsnap3lib.parallel_tools import ParallelTools
from fitsnap3lib.io.sections.sections import Section
from fitsnap3lib.io.error import ExitFunc
from fitsnap3lib.io.sections.calculator_sections.bispectrum import Bispectrum
from fitsnap3lib.io.sections.calculator_sections.calculator import Calculator
from fitsnap3lib.io.sections.calculator_sections.ace import Ace
from fitsnap3lib.io.sections.calculator_sections.basic_calculator import Basic
from fitsnap3lib.io.sections.calculator_sections.custom import Custom
from fitsnap3lib.io.sections.eshift import Eshift
from fitsnap3lib.io.sections.trainshift import Trainshift
from fitsnap3lib.io.sections.extras import Extras
from fitsnap3lib.io.sections.groups import Groups
from fitsnap3lib.io.sections.memory import Memory
from fitsnap3lib.io.sections.outfile import Outfile
from fitsnap3lib.io.sections.path import Path
from fitsnap3lib.io.sections.reference import Reference
from fitsnap3lib.io.sections.scraper import Scraper
from fitsnap3lib.io.sections.solver_sections.solver import Solver
from fitsnap3lib.io.sections.solver_sections.ard import Ard
from fitsnap3lib.io.sections.solver_sections.lasso import Lasso
from fitsnap3lib.io.sections.solver_sections.ridge import Ridge
from fitsnap3lib.io.sections.solver_sections.jax import JAX
from fitsnap3lib.io.sections.solver_sections.pytorch import PYTORCH
from fitsnap3lib.io.sections.solver_sections.network import NETWORK
from fitsnap3lib.io.sections.template import Default


#pt = ParallelTools()


def new_section(section, config, pt, infile, args):
    """Section Factory"""
    instance = search(section)
    try:
        instance.__init__(section, config, pt, infile, args)
    except ExitFunc:
        pass
    return instance


def search(section):
    instance = None
    for cls in Section.__subclasses__():
        if cls.__name__.lower() == section.lower():
            instance = Section.__new__(cls)

    if instance is None:
        raise IndexError("{} was not found in fitsnap sections".format(section))
    else:
        return instance
