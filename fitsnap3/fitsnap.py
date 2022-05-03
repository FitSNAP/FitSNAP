
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
#     Charles Sievers (Sandia National Labs)
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

from .parallel_tools import pt
from .scrapers.scraper_factory import scraper
from .calculators.calculator_factory import calculator
from .solvers.solver_factory import solver
from .io.output import output
from .io.input import config


class FitSnap:
    def __init__(self):
        self.scraper = scraper(config.sections["SCRAPER"].scraper)
        self.data = []
        self.calculator = calculator(config.sections["CALCULATOR"].calculator)
        self.solver = solver(config.sections["SOLVER"].solver)
        self.fit = None
        self.multinode = 0
        if config.sections["EXTRAS"].only_test:
            self.fit = output.read_fit()

    @pt.single_timeit
    def scrape_configs(self):
        self.scraper.scrape_groups()
        self.scraper.divvy_up_configs()
        self.data = self.scraper.scrape_configs()
        del self.scraper

    @pt.single_timeit
    def process_configs(self):
        self.calculator.create_a()
        for i, configuration in enumerate(self.data):
            self.calculator.process_configs(configuration, i)
        del self.data
        self.calculator.collect_distributed_lists()
        self.calculator.extras()

    @pt.single_timeit
    def perform_fit(self):
        if not config.args.perform_fit:
            return
        elif self.fit is None:
            self.solver.perform_fit()
        else:
            self.solver.fit = self.fit
        self.solver.fit_gather()
        self.solver.error_analysis()

    @pt.single_timeit
    def write_output(self):
        if not config.args.perform_fit:
            return
        output.output(self.solver.fit, self.solver.errors)
