from fitsnap3.io.sections.sections import Section
from fitsnap3.io.sections.template import Default
from fitsnap3.io.sections.bispectrum import Bispectrum
from fitsnap3.io.sections.model import Model
from fitsnap3.io.sections.path import Path
from fitsnap3.io.sections.outfile import Outfile
from fitsnap3.io.sections.reference import Reference
from fitsnap3.io.sections.eshift import Eshift
from fitsnap3.io.sections.memory import Memory
from fitsnap3.io.sections.scraper import Scraper


def new_section(section, config, args):
    """Section Factory"""
    instance = search(section)
    instance.__init__(section, config, args)
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
