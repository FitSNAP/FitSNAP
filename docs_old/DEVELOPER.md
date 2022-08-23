### Data & configuration extraction.

`__main__.py` shows the flow of the FitSNAP program, which we see is

    initialize_fitsnap_run()
    snap = FitSnap()
    snap.scrape_configs()
    snap.process_configs()
    pt.all_barrier()
    snap.perform_fit()
    snap.write_output()

After creating the FitSNAP object, the first step is scraping the configs.
Then we process the configs with `snap.process_configs()`.
This calls the function in `fitsnap.py`:

    def process_configs(self):
        self.calculator.create_a()
        for i, configuration in enumerate(self.data):
            self.calculator.process_configs(configuration, i)
        del self.data
        self.calculator.collect_distributed_lists()
        self.calculator.extras()

The `Calculator` class in `calculators/calculator.py` has a `create_a` method which allocates the 
size of the `a` and `b` matrices, containing data such as descriptors and target energies/forces.
`calculators/calculator.py` also has a `process_configs` method which is overwritten by the 
derived class, e.g. `LammpsSnap` in `lammps_snap.py`.
The `calculator.process_configs` method therefore gets directed to the method in the derived 
class, which depends on the particular calculator being used.

### Modifying the output dataframe

The `error_analysis` function in `solvers/solver.py` builds a dataframe containing arrays from 
`pt.shared_arrays`. If you want to add your own column to the dataframe, it must first be 
declared/allocated as a `pt.shared_array` in `calculators/calculator.py`, with the 
`pt.add_2_fitsnap` function. When extracting LAMMPS data in a particular calculator subclass, 
there are loops over energy `bik` rows, force rows, and stress rows. These are located in 
`lammps_snap.py` and `lammps_pace.py`, in the `_collect_lammps()` function. There it is seen that 
data is added to the `pt.fitsnap_dict['Column_Name'][indices]` array, where `'Column_Name'` is the 
name of the new column declared earlier, and `'indices'` are the rows of the array.

### Adding new keywords in the input file

First you need to realize what section the keyword you're adding is in. For example in the input
file you will see a `CALCULATOR` section. If you want to add a keyword to this section, go to 
`fitsnap3lib/io/sections/calculator_sections/calculator.py`, and use the existing keyword examples
to add a new keyword. Likewise for other sections such as `SOLVER`, we edit 
`fitsnap3lib/io/sections/solver_sections/solver.py`. If you want to access this keyword later in
the FitSNAP code somewhere, it is done with `config.sections['SOLVER'].new_keyword` for example.  
