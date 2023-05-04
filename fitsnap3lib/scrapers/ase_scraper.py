from fitsnap3lib.scrapers.scrape import Scraper
import numpy as np
from ase import Atoms,Atom
from ase.io import read,write
from ase.io import extxyz


class ASE(Scraper):

    def __init__(self, name, pt, config):
        super().__init__(name, pt, config)
        self.conversions = copy(self.default_conversions)
        self.all_data = []
        self.style_info = {}