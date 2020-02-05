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

from input_output import parse_config
from parallel_tools import pt
import scrape


class FitSnap:
    def __init__(self, cmdline_args):
        self.cmdline_args = cmdline_args
        self.config = parse_config(self.cmdline_args)
        self.scraper = None

    def scrape_configs(self):
        self.scraper = scrape.JsonScraper()

    def run_lammps(self):
        return None

    @pt.sub_rank_zero
    def perform_fit(self):
        return None
        # for key in self.config:
        #     pt.single_print(key)
        #     for sub_key in self.config[key]:
        #         pt.single_print(sub_key, self.config[key][sub_key])

