import os, json, time, math
import numpy as np
from copy import deepcopy
from fitsnap3lib.fitsnap import FitSnap
from mpi4py import MPI
from psutil import virtual_memory


class CostObject:
    def __init__(self):
        self.conts = None
        self.cost = 999999999.9999
        self.unweighted = np.array([])
        self.weights = np.array([])

    def cost_contributions(self,cost_conts,weights):
        cost_conts = np.array(cost_conts) 
        weights = np.array(weights)
        wc_conts = weights*cost_conts
        self.unweighted = cost_conts
        self.weights = weights
        self.conts = wc_conts

    def add_contribution(self,costi,weighti=1.0):
        try:
            if self.conts == None:
                cond = True
            else:
                cond = False
        except ValueError:
            cond = False
        if cond:
            cost_conts = np.array([costi])
            weights = np.array([weighti])
            wc_conts = weights*cost_conts
            self.conts = wc_conts
            self.weights = np.append(self.weights, weighti)
            self.unweighted = np.append(self.unweighted, costi)
        else:
            self.conts = np.append(self.conts,costi*weighti)
            self.weights = np.append(self.weights,weighti)
            self.unweighted = np.append(self.unweighted,costi)

    def evaluate_cost(self):
        costi = np.sum(self.conts)
        self.cost = costi
        return costi


class HyperparameterStruct:
    """
    Manages hyperparameters for genetic algorithm and simulated annealing.
    
    Args: 
        ne: number of energy rows
        nf: number of force rows (same as ne for FitSNAP genetic_algorithm)
        ns: number of stress rows (same as ne for FitSNAP genetic_algorithm)
        eranges: allowed span of energy weights
        ffactors: allowed ratios of forces (relative to energies)
        sfactors: allowed ratios of stresses (relative to energies)
    """
    def __init__(self,
            ne,
            nf,
            ns,
            eranges,
            ffactors,
            sfactors,):
        self.ne = ne
        self.nf = nf
        self.ns = ns
        self.nh = ne+nf+ns
        self.erangesin = eranges
        self.ffactorsin = ffactors
        self.sfactorsin = sfactors
        self.set_eranges()
        self.set_ffactors()
        self.set_sfactors()

    def set_eranges(self):
        if len(self.erangesin) != self.ne and len(self.erangesin) == 1:
            self.eranges = self.erangesin * self.ne
        elif len(self.erangesin) == self.ne:
            self.eranges = self.erangesin
        else:
            raise ValueError('incorrect number of values for energy group weight ranges, ether specify range for each group or sepecify one range to be applied to all groups')


    def set_ffactors(self):
        if len(self.ffactorsin) != self.nf and len(self.ffactorsin) == 1:
            self.ffactors = self.ffactorsin * self.nf
        elif len(self.ffactorsin) == self.nf:
            self.ffactors = self.ffactorsin
        else:
            raise ValueError('incorrect number of values for force group weight ratios, ether specify range for each group or sepecify one range to be applied to all groups')
    
    
    def set_sfactors(self):
        if len(self.sfactorsin) != self.ns and len(self.sfactorsin) == 1:
            self.sfactors = self.sfactorsin * self.ns
        elif len(self.sfactorsin) == self.ns:
            self.sfactors = self.sfactorsin
        else:
            raise ValueError('incorrect number of values for stress group weight ratios, ether specify range for each group or sepecify one range to be applied to all groups')


    def random_params(self,inputseed=None):
        if inputseed != None:
            np.random.seed(inputseed)
        self.eweights = np.random.rand(self.ne) * np.array([np.random.choice(self.eranges[ihe]) for ihe in range(self.ne)])
        f_factors =  np.random.rand(self.nf) * np.array([np.random.choice(self.ffactors[ihf]) for ihf in range(self.nf)])
        self.fweights = self.eweights * f_factors
        s_factors =  np.random.rand(self.ns) * np.array([np.random.choice(self.sfactors[ihs]) for ihs in range(self.ns)])
        self.fweights = self.eweights * s_factors
        return np.concatenate((self.eweights,self.fweights, self.sweights))

    def lhs_params(self, num_samples, inputseed=None):
        """
        Sets up parameters for this class to sample weights using a Latin hypercube scheme.

        Args:
            num_samples: how many Latin Hypercube points to return
            inputseed: value to initialize RNG for np.random functions

        Returns:
            lhsamples: a Numpy array of shape (num_samples, self.ne*3). Each sample created is itself a concatenated list of [eweight, fweight, vweight] values (may need reshaping)

        """
        if inputseed != None:
            np.random.seed(inputseed)
        variable_ranges_dicti = {}
        variable_types_dict = {}
        for i in range(self.ne):
            # print (i,'ew,fr', [np.log10(min(self.eranges[0])),np.log10(max(self.eranges[0]))], [np.log10(min(self.ffactors[0])),np.log10(max(self.ffactors[0]))])
            # set energy range data
            variable_ranges_dicti['ew%d'%i] = [float(np.log10(min(self.eranges[0]))),float(np.log10(max(self.eranges[0])))]
            variable_types_dict['ew%d'%i] = 'logfloat'

            # set force ratio data
            variable_ranges_dicti['fr%d'%i] = [float(np.log10(min(self.ffactors[0]))),float(np.log10(max(self.ffactors[0])))]
            variable_types_dict['fr%d'%i] = 'logfloat'

            # set stress ratio data
            if self.sfactors[0] != [0,0]:
                variable_ranges_dicti['sr%d'%i] = [float(np.log10(min(self.sfactors[0]))),float(np.log10(max(self.sfactors[0])))] 
                variable_types_dict['sr%d'%i] = 'logfloat'
            else:
                # set to 0 if not fitting stresses
                variable_ranges_dicti['sr%d'%i] = self.sfactors[0]
                variable_types_dict['sr%d'%i] = float

        # print ("HH varrange dict: ",variable_ranges_dicti)
        lhsamples = latin_hypercube_sample(variable_ranges_dicti, variable_types_dict, num_samples)
        return lhsamples


# tournament selection
def tournament_selection(population, scores, k=3, inputseed=None):
    if inputseed != None:
        np.random.seed(inputseed)
    selection_ix = np.random.randint(len(population))
    for ix in np.random.randint(0, len(population), k-1):
        # check if better (e.g. perform a tournament)
        if scores[ix] < scores[selection_ix]:
            selection_ix = ix
    return population[selection_ix]


class Selector:
    def __init__(self,selection_style = 'tournament'):
        self.selection_style = selection_style
        self.set_selector()

    def set_selector(self):
        if self.selection_style == 'tournament':
            self.selection = tournament_selection


def crossover(p1, p2, ne, w_combo_delta=np.array([]), ef_rat_delta=np.array([]), es_rat_delta=np.array([]), num_wcols=3, inputseed=None):
    """
    Genetic operator function that takes two existing population members ('creatures') and recombines them to create two new ones for the next generation. Or in the immortal words of the Goff... "when 2 parent creatures love eachother very much, they make/adopt 2 children creatures"
    
    Args:
        p1: creature 1
        p2: creature 2
        ne: number of elements
        w_combo_delta: sets certain group energy weights to zero
        ef_rat_delta: sets certain group force weights to zero
        es_rat_delta: sets certain group stress weights to zero
        num_wcols: number of weight columns (2 or 3)
        inputseed: value to initialize RNG for np.random functions

    Returns:
        [c1, c2]:  a list containing two new candidate population members
    """

    if inputseed != None:
        np.random.seed(inputseed)
    c1, c2 = p1.copy(), p2.copy()
    c1e,c1f,c1s = tuple(c1.reshape((num_wcols ,ne)))
    c2e,c2f,c2s = tuple(c2.reshape((num_wcols ,ne)))
    p1e,p1f,p1s = tuple(p1.reshape((num_wcols ,ne)))
    p2e,p2f,p2s = tuple(p2.reshape((num_wcols ,ne)))

    # select crossover point that corresponds to a certain group
    # NOTE meg changed var name 'pt' to 'cpt' to avoid confusion with parallel tools "pt" later
    ##LOGAN QUESTION: Can you confirm these indices are correct?
    cpt = np.random.randint(1, ne-2)
    # only perform crossover between like hyperparameters (energy and energy then force and force, etc.)
    if np.shape(w_combo_delta)[0] != 0:
        c1e = np.append(p1e[:cpt], p2e[cpt:])*w_combo_delta
        c1f = np.append(p1f[:cpt], p2f[cpt:])*ef_rat_delta
        c1s = np.append(p1s[:cpt], p2s[cpt:])*es_rat_delta
        c2e = np.append(p2e[:cpt], p1e[cpt:])*w_combo_delta
        c2f = np.append(p2f[:cpt], p1f[cpt:])*ef_rat_delta
        c2s = np.append(p2s[:cpt], p1s[cpt:])*es_rat_delta
    else:
        c1e = np.append(p1e[:cpt] , p2e[cpt:])
        c1f = np.append(p1f[:cpt] , p2f[cpt:])
        c1s = np.append(p1s[:cpt] , p2s[cpt:])
        c2e = np.append(p2e[:cpt] , p1e[cpt:])
        c2f = np.append(p2f[:cpt] , p1f[cpt:])
        c2s = np.append(p2s[:cpt] , p1s[cpt:])
    c1 = np.concatenate((c1e,c1f,c1s))
    c2 = np.concatenate((c2e,c2f,c2s))
    return [c1, c2]


def update_weights(fs, test_w_combo, test_ef_rat, test_es_rat, gtks, size_b, grouptype, initial_weights=False):
    """
    Updates creature weights 
    
    Args:
        fs: instance of FitSNAP class
        test_w_combo: set of energy weights to test
        test_ef_rat: set of force weights to test
        test_es_rat:  set of stress weights to test
        gtks: group table keys from FitSNAP config
        size_b: length of b matrix
        grouptype: list of group labels from fs.pt.fitsnap_dict
        test_virial_w: if stress fitting not turned on, sets stress (virial) weights to near-zero value ? TODO double check
        initial_weights: use the initial weights (? and score?) to seed new candidates

    Returns:
        (none, updates FitSNAP instance in-place)
    """
    if initial_weights:
        if len(test_es_rat) == 1:
            tstwdct = {gkey:{'eweight':initial_weights[gkey][0]*test_w_combo[ig], 
                'fweight':initial_weights[gkey][1]*test_w_combo[ig]*test_ef_rat[ig], 
                'vweight':initial_weights[gkey][2]*test_w_combo[ig]*test_es_rat[0]} for ig,gkey in enumerate(gtks)}
        elif len(test_es_rat) == len(test_ef_rat):
            tstwdct = {gkey:{'eweight':initial_weights[gkey][0]*test_w_combo[ig], 
                'fweight':initial_weights[gkey][1]*test_w_combo[ig]*test_ef_rat[ig],
                'vweight':initial_weights[gkey][2]*test_w_combo[ig]*test_es_rat[ig]} for ig,gkey in enumerate(gtks)}
        else:
            raise IndexError("not enough virial indices per energy and force indices")
    else:
        if len(test_es_rat) == 1:
            tstwdct = {gkey:{'eweight':test_w_combo[ig], 'fweight':test_w_combo[ig]*test_ef_rat[ig], 'vweight':test_w_combo[ig]*test_es_rat[0]} for ig,gkey in enumerate(gtks)}
        elif len(test_es_rat) == len(test_ef_rat):
            tstwdct = {gkey:{'eweight':test_w_combo[ig], 'fweight':test_w_combo[ig]*test_ef_rat[ig], 'vweight':test_w_combo[ig]*test_es_rat[ig]} for ig,gkey in enumerate(gtks)}
        else:
            raise IndexError("not enough virial indices per energy and force indices")

    new_w = np.zeros(size_b)
    for index_b in range(size_b):
        gkey = grouptype[index_b]
        if fs.pt.fitsnap_dict['Row_Type'][index_b] == 'Energy':
            new_w[index_b] = tstwdct[gkey]['eweight']
        elif fs.pt.fitsnap_dict['Row_Type'][index_b] == 'Force':
            new_w[index_b] = tstwdct[gkey]['fweight']
        elif fs.pt.fitsnap_dict['Row_Type'][index_b] == 'Stress':
            new_w[index_b] = tstwdct[gkey]['vweight']
    
    return new_w


def ediff_cost(fs, fit, g1, g2, target, grouptype, rowtype):
    """
    NOTE: this function will eventually be used in calculating objective functions on properties, but is currently not implemented!

    Provides 'cost' for energy differences for A matrix entries:
        (beta * a1 - beta * a2) - target,
    where target is some target energy difference between structures from group 1 and group 2. 
    Important: note that a1 and a2 A matrix entries are identified by indexing. 

    Args: 
        fs: instance of FitSNAP class 
        fit: a fit to training performed by FitSNAP. note this does not need to be the same as one contained in the fs object
        g1: structure group 1
        g2: structure group 2
        target: energy threshhold
        grouptype: list of groups contained in fitsnap_dictionary
        rowtype: whether input is an energy, force, or stress calculation

    Return: 
        np.abs(diff - target): 
        
    Objective function dev. note: You MUST put only one structure per group to use in this objective function.
    """

    indexg1 = grouptype.index(g1)
    indexg2 = grouptype.index(g2)
    assert rowtype[indexg1] == 'Energy',"not accessing energy row for group %s" % g1
    assert rowtype[indexg2] == 'Energy',"not accessing energy row for group %s" % g2
    a1 = fs.pt.shared_arrays['a'].array[indexg1].copy()
    a2 = fs.pt.shared_arrays['a'].array[indexg2].copy()

    e1 = np.sum(a1*fit)
    e2 = np.sum(a2*fit)
    diff = e1-e2

    #fs.pt.single_print(e1,e2,diff,target)

    return np.abs(diff - target)

# NOTE from James: fit_and_cost will likely need to be modified to print current fit if
# other objective functions are to be added. 
def fit_and_cost(fs, fitobjects, costweights):
    """
    Perform FitSNAP fit and evaluate errors (cost or score).

    Args:
        fs: instance of FitSNAP class
        costweights: tuple with user-specified weights on energies, forces, stresses, default is (1.,1.,1.) (no special weighting)

    Returns:
        cost: a value representing the fit's performance (currently uses RMSE on training only)

    Objective function note: fit_and_cost will likely need to be modified to print current fit if other objective functions are to be added. See commented-out examples below.
    """
    etot_weight, ftot_weight, stot_weight = tuple(costweights)
    a, b, w, fs_dict = tuple(fitobjects)

    #clear old fit and solve test fit
    fs.solver.fit = None
    
    # Perform new fit and create fs.solver.fit object
    fs.solver.perform_fit(a, b, w, fs_dict)
    fittst = fs.solver.fit
    
    # Analyze errors and create fs2.solver.errors object
    fs.solver.error_analysis(a, b, w, fs_dict)
    errstst = fs.solver.errors

    # collect rmse errors for score
    # TODO: eventually refactor for different cost calculation methods, for now keep RMSE
    rmse_tst = errstst.iloc[:,2].to_numpy()
    rmse_eattst, rmse_fattst, rmse_sattst = rmse_tst[0:3]

    # calculate score 
    CO = CostObject()
    CO.add_contribution(rmse_eattst,etot_weight)
    CO.add_contribution(rmse_fattst,ftot_weight)
    CO.add_contribution(rmse_sattst,stot_weight)

    # NOTE from James: commented examples on how to use energy differences in the objective function
    # a SINGLE structure is added to two new fitsnap groups, the corresponding energy
    # difference between group 1 and group 2 is given as the target (in eV) (NOT eV/atom)
    #obj_ads = ediff_cost(fittst,g1='H_ads_W_1',g2='H_above_W_1',target=-777.91314380000-(-775.10721929000),grouptype=grouptype)
    #CO.add_contribution(obj_ads,etot_weight * 1)
    #obj_ads = ediff_cost(fittst,g1='H_ads_W_2',g2='H_above_W_2',target=-781.72430764000-(-781.66087408000),grouptype=grouptype)
    #CO.add_contribution(obj_ads,etot_weight * 1)

    cost = CO.evaluate_cost()
    del CO
    return cost


def seed_maker(fs, mc, mmax = 1e9,use_saved_seeds=True,seeds_file='seeds.npy'):
    """
    Manages the creation and use of seeds for Numpy's random number generators.
    In the genetic_algorithm function, setting 'use_saved_seeds' to True guarantees that fits with the same FitSNAP input (training, settings, etc.), and evolution parameters are guaranteed to result in the same best scoring fits.

    Args:
        fs: instance of FitSNAP class
        mc: max count of seeds
        mmax: largest value accepted by random number generator
        use_saved_seeds: toggle reading of seeds from a file or not
        seeds_file: name of file to read seeds from, if use_saved_seeds toggled on 

    Returns:
        seeds: a Numpy array of integer seeds

    """

    if use_saved_seeds:
        try:
            seeds = np.load(seeds_file)
            fs.pt.single_print(f'Loaded seeds from {seeds_file}')
            if np.shape(seeds)[0] < mc:
                fs.pt.single_print('potentially not enough seeds for this run, appending more')
                seeds = np.append(seeds , np.random.randint(0,mmax, mc- (np.shape(seeds)[0])  ))
                np.save(seeds_file,seeds)
            else:
                seeds = seeds
        except FileNotFoundError:
            seeds = np.random.randint(0,mmax,mc)
            np.save(seeds_file,seeds)
    else:
        seeds = np.random.randint(0,mmax,mc)
        np.save(seeds_file,seeds)
    return seeds


def mutation(current_w_combo, current_ef_rat, current_es_rat, my_w_ranges, my_ef_ratios, my_es_ratios, ng, w_combo_delta=np.array([]), ef_rat_delta=np.array([]), es_rat_delta=np.array([]), apply_random=True, full_mutation=False):
    """
    Genetic operator function that injects random values into weights (exploration).
    
    Args: 
        current_w_combo: values of current creature's weight combinations
        current_ef_rat: values of current creature's force ratios
        current_es_rat: values of current creature's stress ratios
        my_w_ranges: allowed ranges of (energy) weights
        my_ef_ratios: allowed ratios of force weights relative to energy weights
        my_es_ratios: allowed ratios of stress weights relative to energy weights
        ng: number of groups
        w_combo_delta: sets given groups' energy weights to 0
        ef_rat_delta: sets given groups' force weights to 0
        efsrat_delta: sets given groups' stress weights to 0
        apply-random: 
        full_mutation: 

    Returns:
        test_w_combo: new set of energy weights to test
        test_ef_rat: new set of force weights to test
        test_es_rat:  new set of stress weights to test
    """

    if type(current_w_combo) == tuple:
        current_w_combo = np.array(current_w_combo)
    if type(current_ef_rat) == tuple:
        current_ef_rat = np.array(current_ef_rat)
    if type(current_es_rat) == tuple:
        current_es_rat = np.array(current_es_rat)
    if full_mutation:
        if apply_random:
            test_w_combo = np.random.rand()*np.random.choice(my_w_ranges,ng)
            test_ef_rat = np.random.rand()*np.random.choice(my_ef_ratios,ng)
            test_es_rat = np.random.rand()*np.random.choice(my_es_ratios,ng)
        else:
            test_w_combo = np.random.choice(my_w_ranges,ng)
            test_ef_rat = np.random.choice(my_ef_ratios,ng)
            test_es_rat = np.random.choice(my_es_ratios,ng)
    else:
        test_w_combo = current_w_combo.copy()
        test_ef_rat = current_ef_rat.copy()
        test_es_rat = current_es_rat.copy()
        test_w_ind = np.random.choice(range(ng))
        if apply_random:
            plusvsprd =  1 #TODO implement addition/product steps after constraining min/max weights
            if plusvsprd:
                test_w_combo[test_w_ind] = np.random.rand() * np.random.choice(my_w_ranges)
                test_ef_rat[test_w_ind] = np.random.rand() * np.random.choice(my_ef_ratios)
                test_es_rat[test_w_ind] = np.random.rand() * np.random.choice(my_es_ratios)
            else:
                test_w_combo[test_w_ind] *= np.random.rand() * np.random.choice(my_w_ranges)
                test_ef_rat[test_w_ind] *= np.random.rand() * np.random.choice(my_ef_ratios)
                test_es_rat[test_w_ind] *= np.random.rand() * np.random.choice(my_es_ratios)
        else:
            test_w_combo[test_w_ind] = np.random.choice(my_w_ranges)
            test_ef_rat[test_w_ind] = np.random.choice(my_ef_ratios)
            test_es_rat[test_w_ind] = np.random.choice(my_es_ratios)
    if np.shape(w_combo_delta)[0] != 0:
        return test_w_combo * w_combo_delta, test_ef_rat*ef_rat_delta, test_es_rat*es_rat_delta
    else:
        return test_w_combo,test_ef_rat,test_es_rat


def print_final(fs, gtks, ew_frcrat_final, best_gen, best_score, write_to_json=False):
    ew_final, frcrat_final, srcrat_final = ew_frcrat_final

    calc_stress = fs.config.sections["CALCULATOR"].stress
    print_stress = True
    wcols = [v for v in fs.config.sections["GROUPS"].group_sections if "weight" in v]
    num_wcols = len(wcols)
    if num_wcols == 2:
        print_stress = False
        
    # fitsnap TODO: when using the JSON scraper, training_size and_testing size are converted from floats into integers, which is inconsistent and should be updated
    loc_gt = fs.config.sections["GROUPS"].group_table

    collect_lines = []
    fs.pt.single_print(f'\n---> Best group weights (from generation {best_gen}, score {best_score}):')
    for idi, dat in enumerate(gtks):
        en_weight = ew_final[idi]
        frc_weight = ew_final[idi]*frcrat_final[idi]
        if print_stress:
            src_weight = ew_final[idi]*srcrat_final[idi]

        ntrain = loc_gt[dat]['training_size']
        ntest = loc_gt[dat]['testing_size']
        ntot = ntrain + ntest
        train_sz = round(ntrain/ntot,2)
        test_sz = round(ntest/ntot,2)

        # fs.pt.single_print('%s       =  %1.2f      %1.2f      %.16E      %.16E      1.E-12' % (dat, train_sz,test_sz,en_weight,frc_weight))
        group_line = f'{dat}       =  {train_sz}      {test_sz}      {en_weight}      {frc_weight}'
        if print_stress:
            group_line += f"      {src_weight}"
        fs.pt.single_print(group_line)
        collect_lines.append([dat, group_line.replace(f'{dat}       =  ','')])
    fs.pt.single_print("")

    # MEG NOTE: this write_to_json works fine but is a bit of a mess
    # TODO rework with better configparser settings
    if write_to_json:
        settings = get_fs_input_dict(fs)

        # remove stress parameters and update smartweights from config object
        # TODO make sure to manage this if stress/smartweights management changes
        settings["GROUPS"]["group_sections"] = " ".join([gs for gs in settings["GROUPS"]["group_sections"].split() if gs != "vweight"])
        settings["GROUPS"]["group_types"] = " ".join([gt for gt in settings["GROUPS"]["group_types"].split()][:-1])
        settings["GROUPS"]["smartweights"] = str(fs.config.sections["GROUPS"].smartweights)

        # automatically create an outfile name from the potential name
        # TODO eventually allow user to override
        potential = settings["OUTFILE"]["potential"]
        outfile = f"{potential}.fs-input.json"

        # update group weights string in settings
        for group, line in collect_lines:
            settings["GROUPS"][group] = line

        # currently a bug where some lower/uppercase versions of groups are doubled
        del_invalid = []
        lower_gtks = [g.lower() for g in gtks]
        for sgroup in settings["GROUPS"].keys():
            input_path = fs.config.sections["PATH"].datapath
            group_path = f"{input_path}/{sgroup}"
            if not os.path.exists(group_path) and sgroup in lower_gtks:
                del_invalid.append(sgroup)
        [settings["GROUPS"].pop(g) for g in del_invalid] 

        # write to json
        fs.pt.single_print("Writing to JSON: ", outfile)
        with open(outfile, 'w') as f:
            json.dump(settings, f, indent=4)


def get_fs_input_dict(fs):
    """
    Loads or generates current FitSNAP settings

    Args:
        fs: instance of FitSNAP class

    Returns:
        settings: Python dictionary with all FitSNAP settings in current instance
    """
    infile_name = fs.config.infile
    settings = fs.config.indict

    # if a dictionary wasn't used to import settings, create one from the input file
    if settings == None:
        # read in config again
        import configparser
        c = configparser.ConfigParser()

        # next line is SUPER IMPORTANT! without this, the ConfigParser puts everything into lower case, leading to unmatched variables and bad pathing for group scrapers
        c.optionxform = str 
        c.read(infile_name)
        settings = {s:dict(c.items(s)) for s in c.sections()}
    return settings

#LOGAN NOTE: HAVE NOT UPDATED THIS FUNCTION - DON'T THINK IT NEEDS IT?
# MEG NOTE: i think you're correct but let's have James take a look
def latin_hypercube_sample(variable_ranges_dict, variable_types_dict, num_samples, seed=12345):
    # TODO is this doubled from lhparams, should be removed?
    # TODO if not, should this be varied or taken from lhparams?
    np.random.seed(seed)
    variable_ranges = [tuple(v) for v in list(variable_ranges_dict.values())]
    variable_types = [typ for typ in list(variable_types_dict.values())]
    num_variables = len(variable_ranges)
    varseeds = np.random.randint(0,10000,num_variables)
    samples_per_variable = num_samples // num_variables
    #samples_per_variable = 1
    # print (samples_per_variable,num_samples)
    # Generate the initial Latin Hypercube
    lhs_matrix = np.zeros((num_samples, num_variables))
    #(24, 2) (26, 24) nsamp 26
    # print ('lh info',np.shape(variable_ranges),np.shape(lhs_matrix),'nsamp',num_samples)
    for i in range(num_variables):
        np.random.seed(varseeds[i])
        vtyp = variable_types[i]
        # print (i,'vtyp',vtyp,'varranges',variable_ranges[i])#,(min(variable_ranges),max(variable_ranges) ))
        if vtyp == float:
            lhs_matrix[:, i] = np.random.uniform(min(variable_ranges[i]), max(variable_ranges[i]), num_samples)
        elif vtyp == int:
            vrange = list(range(min(variable_ranges[i]),max(variable_ranges[i]) +1,1))
            lhs_matrix[:, i] = np.random.choice(vrange,size=num_samples)
        elif vtyp == 'logfloat':
            #vrange = list(range(min(variable_ranges[i]),max(variable_ranges[i]) +1,1))
            coldata = np.array([10**kk for kk in np.random.uniform(min(variable_ranges[i]), max(variable_ranges[i]), num_samples)])
            # print ('col info',i,coldata)
            lhs_matrix[:, i] = coldata #np.array([10**kk for kk in np.random.choice(vrange,size=num_samples)])
        elif vtyp == str:
            lhs_matrix[:, i] = np.random.choice(variable_ranges[i],size=num_samples)
        else:
            raise TypeError("variable type not implemented")

    # Shuffle the samples within each variable column
    for i in range(num_variables):
        np.random.shuffle(lhs_matrix[:, i])
    # Randomly select one sample from each column to form the final Latin Hypercube
    lhs_samples = np.zeros((num_samples, num_variables))
    for i in range(num_variables):
        lhs_samples[0 :num_samples , i] = lhs_matrix[0 :num_samples , i]

    return lhs_samples


def assign_ranks(population0, ncores):
    """ 
    Sort GA populations into indexed lists used in MPI communication. Works the same in serial mode.

    Args:
        population0: an initial list of creatures with weights
        ncores: number of cores used to run GA

    Returns: 
        population: a list of lists containing the number of creatures to calculate per core
        pop_indices: helper list with the same structure as population that tracks the serial indices of each creature
    """
    population = [[] for _ in range(ncores)]
    pop_indices = [[] for _ in range(ncores)] 
    for i, creature in enumerate(population0):
        # assign creatures alternating cores 
        rank_i = i % ncores 
        population[rank_i].append(creature) 
        pop_indices[rank_i].append(i)
    
    return population, pop_indices


def prep_fitsnap_input(fs, smartweights_override=False):
    """ 
    Manages known bugs and code weak spots by elegantly crashing or warning users of potentially weird behavior. Ideally this will eventually be factored out completely.

    Args:
        fs: instance of FitSNAP class
        smartweights_override: allows user to keep the 'smartweights' option turned on
    """
    # code is currently optimized without smartweights
    # turn off smartweights and warn user
    smartweights = fs.config.sections["GROUPS"].smartweights
    if smartweights == 1 and not smartweights_override:
      fs.pt.single_print(f"\nWARNING: Smartweights toggled on, but is not recommended for current version of genetic_algorithm.")
      fs.pt.single_print(f"WARNING: Setting smartweights to 0.")
      fs.pt.single_print(f"WARNING: Use the argument 'smartweights_override=True' to keep smartweights on.\n")
      fs.config.sections["GROUPS"].smartweights = 0

    # if the number of groups is three or less, crossover operations will fail
    # in goffptimize_output, crossover function: cpt = np.random.randint(1, ne-2)
    # for now, elegantly crash if user has 3 or fewer groups 
    num_groups = len(fs.config.sections["GROUPS"].group_table.keys())
    if num_groups <= 3:
        fs.pt.single_print("\n")
        fs.pt.single_print("\n!ERROR: Need 4 or more groups to use genetic algorithm (see comment)!")
        fs.single_print("!ERROR: I am elegantly crashing now so that you can contact the FitSNAP team to have them solve this for you!!")
        fs.pt.single_print("\n")
        exit()

    # because fitting to stresses is optional, there are some allowable inconsistencies in the way FitSNAP settings are written
    # this code takes care of a few of them, other inconsistencies are managed elsewhere in the code
    calc_stress = fs.config.sections["CALCULATOR"].stress
    has_vweights = True if "vweight" in fs.config.sections["GROUPS"].group_sections else False
    if not calc_stress and has_vweights:
        fs.pt.single_print("\n")
        fs.pt.single_print(f"!WARNING: Your FitSNAP input script indicates you don't want to fit to stresses ([CALCULATOR] stress = 0), but your [GROUPS] have a 'vweights' column!")
        fs.pt.single_print(f"!WARNING: Since you don't want to fit to stresses, we're gonna populate all vweights columns with 0!")
        fs.pt.single_print(f"!WARNING: We're just warning you here because the output might be confusing.")
        fs.pt.single_print(f"!WARNING: <---- consider yourself warned!")
        fs.pt.single_print("\n")
    if calc_stress and not has_vweights:
        fs.pt.single_print("\n")
        fs.pt.single_print(f"!ERROR: Your FitSNAP input script indicates you want to fit to stresses ([CALCULATOR] stress = 1), but your [GROUPS] section is MISSING the 'vweights' column!")
        fs.pt.single_print(f"!ERROR: To fix this, add the word 'vweights' to the end of the [GROUPS] group_section variable, 'float' to the end of the [GROUPS] group_types variable, and some number (doesn't matter what) to the end of each of your group's weights.")
        fs.pt.single_print(f"!ERROR: Try again after adding that stuff! Now exiting.")
        fs.pt.single_print("\n")
        exit()


#-----------------------------------------------------------------------
# begin the primary optimzation functions
#-----------------------------------------------------------------------
# @fs.pt.rank_zero
def genetic_algorithm(fs, population_size=50, ngenerations=100, my_w_ranges=[1.e-4,1.e-3,1.e-2,1.e-1,1,1.e1,1.e2,1.e3,1.e4], my_ef_ratios=[0.001,0.01,0.1,1,10,100,1000], etot_weight=1.0, ftot_weight=1.0, stot_weight=1.0, r_cross=0.9, r_mut=0.1, conv_thr = 1.E-10, conv_check = 2., force_delta_keywords=[], stress_delta_keywords=[], write_to_json=False, my_es_ratios=[], use_initial_weights_flag=False, parallel_population=True ):
    """
    Function to perform optimization of FitSNAP group weights.

    Args:
        fs: FitSnap instance being handled by genetic algorithm
        population_size: number of candidates ("creatures") generated within one generation and tested for fitness. in this code, fitness is how well group weights perform in a FitSnap fit (no puns intended)
        ngenerations: maximum number of allowed iterations of populations. this ends the genetic algorithm calculations if the convergence threshold (conv_thr, see below) is not reached beforehand
        my_w_ranges, my_ef_ratios, my_es_ratios: allowed scaling factors for energy, force, and stress weights
        etot_weight, ftot_weight, stot_weight: weights for energy and force rmse in the optimizer cost function
        r_cross and r_mut: cross over (parenting) and mutation hyperparameters
        conv_thr: convergence threshold for full function (value of RMSE E + RMSE F at which simulation is terminated" 
        conv_check: fraction of ngenerations to start checking for convergence (convergence checks wont be performed very early)
        force_delta_keywords, stress_delta_keywords: 
        write_to_json: whether to write the final best fit to a new FitSNAP settings dictionary to a JSON file
        use_initial_weights_flag: whether to bias fitting with initial_weights 
        parallel_population: whether to use MPI*/split communicator when ncores > 1

    Returns:
        (none, prints best group weights to stdout and/or a JSON file)

    *When using MPI, a quick note on number of cores P: 
    - For now, it's best to use an even number of cores P.
    - If your population_size setting is smaller than P, it will be increased to P (to avoid running empty cores per generation).
    - If you're using MPI but don't want to run the GA in parallel, set the optional genetic_algorithm argument `parallel_population = False.`
    """
    #---------------------------------------------------------------------------
    # Begin in-function optimization hyperparameters

    time1 = time.time()
    rank = fs.pt.get_rank()
    ncores = fs.pt.get_size()

    # get all group names  
    gtks = fs.config.sections["GROUPS"].group_table.keys()
    gtks = list(gtks)
    fs.pt.single_print('Groups:', gtks)

    # check if fitting to stresses turned on
    # if not, then set all stress weights to zero by populating stress_delta_keywords with all group names
    # a warning about this behavior is included in this module's "prep_fitnsap_input" function
    calc_stress = fs.config.sections["CALCULATOR"].stress
    if not calc_stress:
        stress_delta_keywords = gtks
    
    # all calculations must include 'vweights' column for internal calculations
    # MEG NOTE: we could refactor all internal GA inputs into arrays with 2 or 3 cols to get around this, but that's a bigger overhaul and not really important for now
    wcols = [v for v in fs.config.sections["GROUPS"].group_sections if "weight" in v]
    num_wcols = len(wcols)
    if num_wcols == 2:
        for key in gtks:
            fs.config.sections["GROUPS"].group_table[key]['vweight'] = 0.0

    if parallel_population and ncores > 1:
        # split population list into a list of lists
        # index = rank of MPI core
        npop_per_core = 1 # always 1 for split comm
        nremain = population_size % ncores
        npop_per_rank = (population_size//ncores) + 1 if nremain != 0 else (population_size//ncores)
        nfilled_grid = ncores*npop_per_rank
        pop_diff = nfilled_grid - population_size

        # if MPI grid uneven, population cost comparison breaks down
        # to avoid wasting compute power, increase population size to fill an even-sized MPI grid
        while nfilled_grid % 2 == 1:
            npop_per_rank += 1

            nfilled_grid = ncores*npop_per_rank
            pop_diff = nfilled_grid - population_size

        if pop_diff != 0:
            fs.pt.single_print("\n")
            fs.pt.single_print(f"MPI: Population size ({population_size}) is currently smaller than even-sized MPI grid allows ({ncores}x{npop_per_rank}={nfilled_grid}).")
            fs.pt.single_print(f"MPI: Updating population size by {pop_diff} to fill even-sized MPI grid.")
            population_size += pop_diff
            fs.pt.single_print(f"MPI: New population_size: {population_size}")
            fs.pt.single_print("\n")
     
    # population can't have odd numbers currently
    if population_size % 2 == 1:
        fs.pt.single_print("\n")
        fs.pt.single_print(f"NOTE: Cannot use odd numbers for population size (input: {population_size}), adding one.")
        population_size += 1
        fs.pt.single_print(f"NOTE: New population_size: {population_size}")
        fs.pt.single_print("\n")

    # start getting weights 
    initial_weights={}
    if use_initial_weights_flag:
        fs.pt.single_print("NOTE: Using weights from FitSNAP input to populate first generation!")
        fs.pt.single_print("\n")
        for key in gtks:
            initial_weights[key] = [fs.config.sections["GROUPS"].group_table[key]['eweight'], fs.config.sections["GROUPS"].group_table[key]['fweight'], \
                                    fs.config.sections["GROUPS"].group_table[key]['vweight']]

    size_b = np.shape(fs.pt.fitsnap_dict['Row_Type'])[0]
    grouptype = fs.pt.fitsnap_dict['Groups'].copy()
    rowtype = fs.pt.fitsnap_dict['Row_Type'].copy()

    countmaxtot = int(population_size*(ngenerations+2))

    #number of hyperparameters:
    # num of energy group weights
    ne = len(gtks)
    # num of force group weights (in general these will always be the same as ne)
    nf = ne
    ns = ne

    # total
    if not my_es_ratios:
        nh = ne + nf
    else:
        nh = ne + nf + ns

    # update ranges and ratios
    eranges = [my_w_ranges]
    ffactors = [my_ef_ratios]
    if calc_stress:
        sfactors = [my_es_ratios]  ##LOGAN NOTE: EMPTY LIST IF NOT USING
    else:
        sfactors = [[0.,0.]]

    # selection method (only tournament is currently implemented)
    # TODO implement other methods?
    selection_method = 'tournament'

    # modify convergence check for new conv_flag
    check_gen = int((ngenerations*conv_check))

    # End optimization hyperparameters
    #---------------------------------------------------------------------------

    # set up generation 0
    best_gen = 0
    best_score = 1e10
    conv_flag = False

    # TODO confirm that seed_maker is working as expected
    input1 = get_fs_input_dict(fs)
    seedsi = seed_maker(fs, countmaxtot)
    first_seeds = seedsi[:population_size+1]

    # Latin hypercube population for first generation
    hp = HyperparameterStruct(ne,nf,ns,eranges,ffactors,sfactors)

    # population = [hp.random_params(inputseed=first_seeds[ip]) for ip in range(population_size)] # NOTE orig

    # Random initial population for first generation:
    #population = [hp.random_params(inputseed=first_seeds[ip]) for ip in range(population_size)]
    population_lhs = hp.lhs_params(num_samples=population_size,inputseed=first_seeds[0])

    # Reorder here so that reshaping can be done without extra parameter
    # for N groups, the initial Latin hypercube shape: [group1_eweight, group1_fweight, group1_vweight,... groupN_vweight]
    # for N groups, the output shape: [group1_eweight, group2_eweight, group3_eweight,...,  group(N-1)_vweight, groupN_vweight]
    population0 = []
    for creature0 in population_lhs:
        creature = creature0.reshape((num_wcols, ne), order='F').flatten()
        population0.append(creature)

    # set up initial MPI settings (even in serial)
    population, pop_indices = assign_ranks(population0, ncores)

    # set up GA variables
    generation = 0
    best_gen = 0
    best_gens = []
    best_scores = []
    best = None
    best_weights = []
    all_lowest_scores = [best_score] # for mpi testing, statistics
    all_lowest_weights = [tuple(population0[0])] # for mpi testing, statistics
    all_lowest_creature_idxs = [-1]
    sim_seeds = seedsi[population_size:]
    np.random.seed(sim_seeds[generation])
    w_combo_delta = np.ones(len(gtks))

    # delta function to zero out force weights on structures without forces  
    # now implemented with user-specified keywords
    if force_delta_keywords != []:
        not_in_fdkws = lambda gti: all([True if fdkw not in gti else False for fdkw in force_delta_keywords])
        ef_rat_delta = np.array([1.0 if not_in_fdkws(gti) else 0.0 for gti in gtks])
    else:
        ef_rat_delta = np.array([1.0]*len(gtks))

    if stress_delta_keywords != [] or len(stress_delta_keywords) == len(gtks):
        not_in_fdkws = lambda gti: all([True if fdkw not in gti else False for fdkw in stress_delta_keywords])
        es_rat_delta = np.array([1.0 if not_in_fdkws(gti) else 0.0 for gti in gtks])
    else:
        es_rat_delta = np.array([1.0]*len(gtks))
    
    # prepare fitting objects from original fs descriptor calculation
    a = fs.pt.shared_arrays['a'].array
    b = fs.pt.shared_arrays['b'].array
    # w0 = fs.pt.shared_arrays['w'].array 
    size_b = b.shape[0]
    fs_dict = fs.pt.fitsnap_dict
    grouptype = fs_dict["Groups"]
    
    # begin evolution
    while generation <= ngenerations and best_score > conv_thr and not conv_flag:
        # toggle variable used to assign FitSNAP communicator (comm) type based on MPI state
        # if serial, par_fs is assigned to the current FitSNAP instance
        # if MPI, a new split comm is created and assigned
        par_fs = None
        pop_list = None

        # these split variables stay -1 if in serial
        # in MPI.COMM_WORLD.Split()
        rank_split, size_split = -1, -1 
        if parallel_population and ncores > 1:
            # MPI lingo for split and grouped communicators, here for future refactoring
            # color: 'name' of comm group
            # key: 'rank' within comm group
            color, key = rank, rank 
            comm_split = MPI.COMM_WORLD.Split(color, key)
            rank_split = comm_split.Get_rank()
            size_split = comm_split.Get_size()

            # create split comm FitSNAP instance outside of creature loop below
            par_fs = FitSnap(input1, comm=comm_split, arglist=["--overwrite"])
        else:
            par_fs = fs

        # take the previously-allocated sublist assigned to this rank
        pop_list = population[rank]

        # prep scores array for MPI
        scores_nrow, scores_ncol = np.array(population).shape[:2]

        # set up data collection (MPI-friendly)
        collect_scores = np.full((scores_nrow, scores_ncol), 1e8)
        per_rank_scores = np.full(len(pop_list), 1e8)

        # loop through creatures in current generation          
        gen_start = time.time()
        for i, creature in enumerate(pop_list):           
            # get creature values from generated population
            creature_ew, creature_ffac, creature_sfac = tuple(creature.reshape((num_wcols,ne)).tolist())  
            creature_ew = tuple(creature_ew)
            creature_ffac = tuple(creature_ffac)
            creature_sfac = tuple(creature_sfac)
            
            # make sure original fs_dict is copied to all members of par_fs comm
            par_fs.pt.fitsnap_dict = deepcopy(fs_dict)

            new_w = update_weights(par_fs, creature_ew, creature_ffac, creature_sfac, gtks, size_b, grouptype,initial_weights=initial_weights)

            costi = fit_and_cost(par_fs, [a, b, new_w, fs_dict], [etot_weight,ftot_weight,stot_weight])

            per_rank_scores[i] = costi

        if parallel_population and ncores > 1:
            par_fs.pt.all_barrier()
            fs.pt._comm.Allgather([per_rank_scores, MPI.FLOAT],[collect_scores, MPI.FLOAT])

            # clean up split comm FitSNAP instance to prep for next generation
            del par_fs
        else:
            collect_scores = per_rank_scores
            
        # unpack in same order as MPI grid assignment
        flat_scores = collect_scores.flatten().tolist()
        flat_pop_indices = [item for items in pop_indices for item in items]
        scores = [flat_scores[flat_pop_indices.index(i)] for i in range(population_size)]

        # NOTE from James: to add another contribution to the cost function, you need to evaluate it in the loop
        # and add it to the fit_and_cost function
        # if this involves a lammps simulation, you will have to print potentials at the different steps
        # to run the lammps/pylammps simulation. To do so, the fitsnap output file name prefix should
        # be updated per step, then fs.write_output() should be called per step. This will likely increase
        # the optimization time.

        # Anything printed with fs.pt.single_print will be included in output file.

        # Print generation and best fit.
        lowest_score_in_gen = 1e10
        lowest_weight_in_gen = []
        lowest_creature_idx = -1
        for i in range(population_size):
            if scores[i] < lowest_score_in_gen:
                lowest_score_in_gen = scores[i]  
                lowest_weight_in_gen = population0[i]
                lowest_creature_idx = i        
            if scores[i] < best_score:
                best, best_score, best_gen = tuple(population0[i]), scores[i], generation
        
        best_weights.append(best)
        best_gens.append(best_gen)
        best_scores.append(best_score)
        
        # track generations for testing mpi, statistics
        all_lowest_weights.append(lowest_weight_in_gen)       
        all_lowest_scores.append(lowest_score_in_gen)
        all_lowest_creature_idxs.append(lowest_creature_idx) 

        # check for convergence
        try:
            # original flag
            # conv_flag = np.round(np.var(best_scores[int(ngenerations/conv_check)-int(ngenerations/10):]),14) == 0

            # new flag, currently testing
            if len(best_scores) >= check_gen:
                conv_flag = np.round(np.var(best_scores[-(check_gen*math.floor(len(best_scores)/check_gen)):]),14) == 0
            else:
                conv_flag = False

        except IndexError:
            conv_flag = False
        printbest = tuple([tuple(ijk) for ijk in np.array(best).reshape((num_wcols,ne)).tolist()])
        
        fs.pt.single_print(f"------------ GENERATION {generation} ------------")
        fs.pt.single_print(f'Lowest score:', lowest_score_in_gen)
        print_final(fs, gtks, printbest, best_gens[-1], best_scores[-1])

        # Choose/rank candidates for next generation
        slct = Selector(selection_style = selection_method)
        selected = [slct.selection(population0, scores) for creature_idx in range(population_size)]
        del slct

        # new generation
        children = list()
        for ii in range(0, population_size, 2):
            # get selected parents in pairs
            p1, p2 = selected[ii], selected[ii+1]
            # crossover and mutation
            rndcross, rndmut = tuple(np.random.rand(2).tolist())
            if rndcross <= r_cross:
                cs = crossover(p1, p2, len(gtks), w_combo_delta, ef_rat_delta, es_rat_delta)
            else:
                cs = [p1,p2]
            for c in cs:
                # mutation
                if rndmut <= r_mut:
                    current_creature_ew, current_creature_ffac, current_creature_sfac = tuple(c.reshape((num_wcols, ne)))
                    current_creature_ew = tuple(current_creature_ew)
                    current_creature_ffac = tuple(current_creature_ffac)
                    current_creature_sfac = tuple(current_creature_sfac)

                    mutated_creature_ew, mutated_creature_ffac, mutated_creature_sfac = mutation(current_creature_ew,current_creature_ffac,current_creature_sfac,\
                    my_w_ranges,my_ef_ratios,my_es_ratios,ng=len(gtks),\
                    w_combo_delta=w_combo_delta,ef_rat_delta=ef_rat_delta, es_rat_delta=es_rat_delta,\
                    apply_random=True,
                    full_mutation=False)

                    c = np.concatenate((mutated_creature_ew,mutated_creature_ffac,mutated_creature_sfac))

                # store for next generation
                children.append(c)

        np.random.seed(sim_seeds[generation])
        population, pop_indices = assign_ranks(children, ncores)
        generation += 1
        gen_end = time.time()
        elapsed = round(gen_end - gen_start, 2)
        fs.pt.single_print(f'Total time to compute generation (population_size {population_size}): {elapsed} s')
        fs.pt.single_print('\n')

    # evolution completed, final steps
    # TODO add message describing whether ngenerations reached or conv_flag = True
    best_ew, best_ffac, best_sfac = tuple(np.array(best_weights[-1]).reshape((num_wcols ,ne)).tolist())
    best_ew = tuple(best_ew)
    best_ffac = tuple(best_ffac)
    best_sfac = tuple(best_sfac)

    ##LOGAN NOTE | TODO: should confirm this is working as expected (always multiplying factor by initial weight and not a previous generation product by initial weight)
    if rank == 0:
        best_w = update_weights(fs, best_ew, best_ffac, best_sfac, gtks, size_b, grouptype, initial_weights=initial_weights)
        costi = fit_and_cost(fs, [a, b, best_w, fs_dict], [etot_weight,ftot_weight,stot_weight])

        fs.pt.single_print('\n------------ Final results ------------')

        # Print out final best fit
        print_final(fs, gtks, tuple([best_ew,best_ffac,best_sfac]), best_gens[-1],best_scores[-1], write_to_json=write_to_json)

        fs.pt.single_print('Writing final output')
        fs.write_output()

        # output final
        time2 = time.time()
        elapsed = round(time2 - time1, 2)
        fs.pt.single_print(f'Total optimization time: {elapsed} s')

# Currently not implemented 
# def sim_anneal(fs):  
#     ##LOGAN NOTE: I have not yet updated this function
#     ##MEG NOTE: same here
#     #---------------------------------------------------------------------------
#     # Begin optimization hyperparameters
#     time1 = time.time()

#     # get groups and weights 
#     gtks = fs.config.sections["GROUPS"].group_table.keys()
#     gtks = list(gtks)
#     fs.pt.single_print('Groups:', gtks)
#     fs.pt.single_print('\n')

#     # check if fitting to stresses turned on
#     # if not, then set all stress weights to zero by populating stress_delta_keywords with all group names
#     # a warning about this behavior is included in this module's "prep_fitnsap_input" function
#     calc_stress = fs.config.sections["CALCULATOR"].stress
#     if calc_stress:
#         fs.pt.single_print("Stress fitting not yet implemented for simulated anneal!")
#         fs.pt.all_barrier()
#         return 0
#     #for future implementation
#     #if not calc_stress:
#         #stress_delta_keywords = gtks

    
#     size_b = np.shape(fs.pt.fitsnap_dict['Row_Type'])[0]
#     grouptype = fs.pt.fitsnap_dict['Groups'].copy()
#     rowtype = fs.pt.fitsnap_dict['Row_Type'].copy()

#     etot_weight = 1.0
#     ftot_weight = 1.5
#     rmse_tot = 500

#     # sampling magnitudes per hyperparameter
#     my_w_ranges = [1.e-3,1.e-2,1.e-1,1.e0,1.e1,1.e2,1.e3]
#     my_ef_ratios = [0.1,1,10]

#     # Artificial temperatures
#     betas = [1.e0,1.e1,1.e2,1.e3,1.e4]
#     # Max number of steps per artificial temperature
#     count_per_beta = [400,400,600,1000,1000]
#     # threshhold for convergence of cost function
#     thresh = 0.005

#     seedpad = 50
#     #build seeds (uses saved seeds by default)
#     countmaxtot = int(np.sum(count_per_beta))
#     seedsi = seed_maker(fs, countmaxtot + seedpad)

#     # End optimization hyperparameters
#     #---------------------------------------------------------------------------
    
#     tot_count = 0
#     #threshold for cost function before accepting model
#     current_w_combo = [1.e0]*len(gtks)
#     current_ef_rat = [10]*len(gtks)
#     tot_count = 0
#     apply_random = True # flag to select a single random hyperparam to step rather than stepping all hyperparams
#     naccept = 0
#     np.random.seed(seedsi[tot_count])
#     # loop over fictitious temperatures
#     for ibeta,beta in enumerate(betas):
#         count = 0
#         naccepti = 0
#         maxcount = count_per_beta[ibeta]
#         # propose trial weights while counts are below maximums and 
#         # objective function is above threshhold
#         while count <= maxcount and rmse_tot >= thresh:

#             if tot_count <= 5: # allow for large steps early in simulation
#                 test_w_combo, test_ef_rat = mutation(current_w_combo,current_ef_rat,my_w_ranges,my_ef_ratios,ng=len(gtks),apply_random=True,full_mutation=True)
#             else:
#                 test_w_combo, test_ef_rat = mutation(current_w_combo,current_ef_rat,my_w_ranges,my_ef_ratios,ng=len(gtks),apply_random=True,full_mutation=False)

#             test_w = update_weights(fs, test_w_combo, test_ef_rat, gtks, size_b, grouptype)

#             rmse_tottst = fit_and_cost(fs, [fs.pt.shared_arrays['a'], fs.pt.shared_arrays['b'], test_w, fs.pt.fitsnap_dict],[etot_weight,ftot_weight])

#             delta_Q = rmse_tottst - rmse_tot
#             boltz = np.exp(-beta*delta_Q)
#             rndi = np.random.rand()
#             logical = rndi <= boltz
#             if logical:
#                 naccept += 1
#                 naccepti +=1
#                 rmse_tot = rmse_tottst
#                 current_w_combo = test_w_combo
#                 current_ef_rat = test_ef_rat

#             meta = (tuple(list(current_w_combo)),) + (tuple(list(current_ef_rat)),)
#             count += 1
#             fs.pt.single_print('beta',beta,'count',count,' accept ratio for current beta %f' % (naccepti/count) ,meta,boltz,rmse_tottst,rmse_tot)
#             tot_count += 1
#             np.random.seed(seedsi[tot_count])

#     # write output for optimized potential
#     print_final(meta)
#     time2 = time.time()
#     fs.pt.single_print('Total optimization time,', time2 - time1, 'total number of fits', tot_count)
#     fs.write_output()