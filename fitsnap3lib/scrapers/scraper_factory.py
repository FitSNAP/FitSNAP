from fitsnap3lib.parallel_tools import ParallelTools
from fitsnap3lib.scrapers.scrape import Scraper
from fitsnap3lib.scrapers.json_scraper import Json
from fitsnap3lib.scrapers.xyz_scraper import XYZ
from fitsnap3lib.scrapers.vasp_scraper import Vasp
from fitsnap3lib.scrapers.pacemaker_scraper import Pacemaker

# only import HDF5 scraper if python module h5py is available
import importlib.util
if importlib.util.find_spec("h5py") is not None:
    from fitsnap3lib.scrapers.hdf5_scraper import HDF5
else:
    HDF5 = None

# only import LMDB scraper if required modules are available
if (importlib.util.find_spec("lmdb") is not None and 
    importlib.util.find_spec("ase") is not None):
    from fitsnap3lib.scrapers.fairchem_scraper import FAIRChem
else:
    FAIRChem = None

#pt = ParallelTools()


def scraper(scraper_name, pt, config):
    """Section Factory"""
    instance = search(scraper_name)
    instance.__init__(scraper_name, pt, config)
    return instance


def search(scraper_name):
    instance = None
    for cls in Scraper.__subclasses__():
        if cls.__name__.lower() == scraper_name.lower():
            instance = Scraper.__new__(cls)

    if instance is None:
        raise IndexError("{} was not found in fitsnap scrapers".format(scraper_name))
    else:
        return instance
