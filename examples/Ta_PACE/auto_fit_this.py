import numpy as np

def change_descriptor_hyperparams(config):
	"""
	Modify descriptor hyperparams associated with a certain config object.

	Attributes
	----------

	config: FitSNAP Config object
		Holds input settings/hyperparameters to tweak

	Returns
	-------

	config: modified Config object
	"""

	# twojmax, wj, and radelem are lists of chars

	config.sections['ACE'].rcutfac = ['4.67637']
	config.sections['ACE'].lmbda = ['3.059235105']
	config.sections['ACE'].rcinner = ['0.0']
	config.sections['ACE'].drcinner = ['0.01']

	config.sections['ACE'].lmin = [1]
	config.sections['ACE'].ranks = [1, 2, 3, 4, 5, 6]
	config.sections['ACE'].lmax = [1, 2, 2, 2, 1, 1]
	config.sections['ACE'].nmax =[22, 2, 2, 2, 1, 1]
	


	# after changing twojmax, need to generate_b_list to adjust all other variables

	config.sections['ACE']._generate_b_list()

	return config

def change_weights(config, data):
	"""
	Change fitting weights associated with each configuration of atoms.

	Attributes
	----------

	config: FitSNAP Config object
		Holds input setting data

	data: FitSNAP data object
		Holds configuration data, positions, forces, weights, for each configuration of atoms

	Returns
	-------

	config: modified Config object

	data: modified data object
	"""

	# loop through all group weights in the group_table and change the value

	for key in config.sections['GROUPS'].group_table:
		for subkey in config.sections['GROUPS'].group_table[key]:
			if ("weight" in subkey):
				# change the weight
				config.sections['GROUPS'].group_table[key][subkey] = np.random.rand(1)[0]

	# loop through all configurations and set a new weight based on the group table

	for i, configuration in enumerate(data):
		group_name = configuration['Group']
		new_weight = config.sections['GROUPS'].group_table[group_name]
		for key in config.sections['GROUPS'].group_table[group_name]:
			if ("weight" in key):
				# set new weight 
				configuration[key] = config.sections['GROUPS'].group_table[group_name][key]

	return(config, data)


from fitsnap3lib.tools.dataframe_tools import DataframeTools
from fitsnap3lib.parallel_tools import ParallelTools
from fitsnap3lib.io.input import Config
config = Config(arguments_lst = ["Ta.in", "--overwrite"])
from fitsnap3lib.fitsnap import FitSnap

ngenerations = 2
for g in range(0,ngenerations):

	# instantiate library objects

	pt = ParallelTools()
	#config = Config(arguments_lst = ["Ta.in", "--overwrite"])
	snap = FitSnap()

	# scrape configs

	snap.scraper.scrape_groups()
	snap.scraper.divvy_up_configs()
	snap.data = snap.scraper.scrape_configs()

	# change the bispectrum hyperparams

	config = change_descriptor_hyperparams(config)

	# change weight hyperparams

	(config, snap.data) = change_weights(config, snap.data)
	
	# process configs with new hyperparams
	# set indices to zero for populating new data array

	snap.calculator.shared_index=0
	snap.calculator.distributed_index=0 
	snap.process_configs()
	 
	# perform a fit and gather dataframe with snap.solver.error_analysis()

	snap.solver.perform_fit()
	snap.solver.fit_gather()
	# need to empty errors before doing error analysis
	snap.solver.errors = []
	snap.solver.error_analysis()

	# now we have the dataframe, calculate errors with it

	df_tool = DataframeTools(snap.solver.df)
	mae_energy = df_tool.calc_error("Energy", "Training")
	mae_force = df_tool.calc_error("Force", "Training")

	print(f"---------- Generation {g} Force MAE: {mae_force} Energy MAE: {mae_energy}")
