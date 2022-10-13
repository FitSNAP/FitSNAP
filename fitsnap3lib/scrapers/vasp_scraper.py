from fitsnap3lib.scrapers.scrape import Scraper, convert
from fitsnap3lib.io.input import Config
from json import loads
from fitsnap3lib.parallel_tools import ParallelTools
from fitsnap3lib.io.output import output
from copy import copy
import os, random, glob, json ## TODO clean up once done
import numpy as np


config = Config()
pt = ParallelTools()

class Vasp(Scraper):

    def __init__(self, name):
        super().__init__(name)
        pt.single_print("Initializing VASP scraper...")
        self.log_data = []
        self.all_data = []
        self.bad_configs = {}
        self.unmatched_groups = {}
        self.outcars_per_group = {}
        self.bc_bool = False
        self.infile = config.args.infile
        self.group_table = config.sections["GROUPS"].group_table
        self.vasp2json = config.sections["GROUPS"].vasp2json

        ## Before scraping, esnure that user has correct input
        ## NOTE: i think Logan recently fixed this, check before putting in again
        # self.check_train_test_sizes()
        
        ## If vasp2json enabled:
        if self.vasp2json:
            pass

    def scrape_groups(self):
        ## skipped some parts of Charlie's XYZ code here

        ### Locate all OUTCARs in datapath
        ## TODO rework pathing/glob with os.path.join() to make system agnostic
        glob_asterisks = '/**/*'
        outcars_base = config.sections['PATH'].datapath + glob_asterisks
        ## TODO make this search user-specify-able
        all_outcars = [f for f in glob.glob(outcars_base,recursive=True) if f.endswith('OUTCAR')]

        ## Grab test|train split
        self.group_dict = {k: config.sections['GROUPS'].group_types[i] for i, k in enumerate(config.sections['GROUPS'].group_sections)}
        for group in self.group_table:
            training_size = None
            if 'size' in self.group_table[group]:
                training_size = self.group_table[group]['size']
                self.bc_bool = True
            if 'training_size' in self.group_table[group]:
                if training_size is not None:
                    raise ValueError('Do not set both size and training size')
                training_size = self.group_table[group]['training_size']
                #size_type = group_dict['training_size']
            if 'testing_size' in self.group_table[group]:
                testing_size = self.group_table[group]['testing_size']
                testing_size_type = self.group_dict['testing_size']
            else:
                testing_size = 0
            if training_size is None:
                raise ValueError('Please set training size for {}'.format(group))
            
            ## Grab OUTCARS for this training group
            ## TODO emulate XYZ pop() to ensure items are processed once only
            group_outcars = [f for f in all_outcars if group in f]

            file_base = os.path.join(config.sections['PATH'].datapath, group)
            self.files[file_base] = group_outcars
            self.configs[group] = []  ##TODO ? need this? copied from XYZ

            if False:
                pass
            else:
                try:
                    for outcar in self.files[file_base]:
                        ## Open file
                        with open(outcar, 'r') as fp:
                            lines = fp.readlines()
                        nlines = len(lines)

                        ## Use ion loop text to partition ionic steps
                        ion_loop_text = 'aborting loop because EDIFF is reached'
                        start_idx_loops = [i for i, line in enumerate(lines) if ion_loop_text in line]
                        end_idx_loops = [i for i in start_idx_loops[1:]] + [nlines]

                        ## Grab potcar and element info
                        header_lines = lines[:start_idx_loops[0]]
                        potcar_elements, ions_per_type = self.parse_outcar_header(header_lines)

                        ## Each config in a single OUTCAR is assigned the same
                        ## parent data (i.e. filename, potcar and ion data)
                        ## but separated for each iteration (idx loops on 'lines')
                        ## TODO as in scrape_outcar, will need to check each config for completeness!
                        outcar_tuples = [(outcar, potcar_elements, ions_per_type,
                                          lines[start_idx_loops[i]:end_idx_loops[i]])
                                         for i in range(0, len(start_idx_loops))]
                        self.configs[group].extend(outcar_tuples)
                except IndexError:
                    self.configs[file_base].pop(-1)

              ## TODO fix random sampling!
            if config.sections["GROUPS"].random_sampling:
                random.shuffle(self.configs[group], pt.get_seed)
            nconfigs = len(self.configs[group])

            ## Assign configurations to train/test groups
            ## check_train_test_sizes() confirms that training_size > 0 and
            ## that training_size + testing_size = 1.0
            ## TODO make sure this doesn't conflict with Logan's fix
            if training_size == 1:
                training_configs = nconfigs
                testing_configs = 0
            else:
                training_configs = max(1, int(round(training_size * nconfigs)))
                if training_configs == nconfigs:
                    ## If training_size is not exactly 1.0, add at least 1 testing config
                    training_configs -= 1
                    testing_configs = 1
                else:
                    testing_configs = nconfigs - training_configs

            if nconfigs - testing_configs - training_configs < 0:
                raise ValueError("training configs: {} + testing configs: {} is greater than files in folder: {}".format(
                    training_configs, testing_configs, nconfigs))

            output.screen(f"{group}: Detected {nconfigs}, fitting on {training_configs}, testing on {testing_configs}")

            ## Populate tests dictionary
            if self.tests is None:
                self.tests = {}
            self.tests[group] = []

            ## Removed next two lines since we gracefully crash if train/test not OK
            # for i in range(nconfigs - training_configs - testing_configs):
            #     self.configs[group].pop()
            for i in range(testing_configs):
                self.tests[group].append(self.configs[group].pop())

            ## TODO propagate change of variable from "_size" to "_configs" or something similar
            self.group_table[group]['training_size'] = training_configs
            self.group_table[group]['testing_size'] = testing_configs
            # self.files[folder] = natsorted(self.files[folder])

    def scrape_configs(self):
        """Generate and send (mutable) data to send to fitsnap"""
        # DATA: outcar_tuples = [(outcar, potcar_elements, ions_per_type,
                          # lines[start_idx_loops[i]:end_idx_loops[i]])
                         # for i in range(0, len(start_idx_loops))

        ## TODO implement scraper on "lines"
        all_data = []
        for config in self.configs:
            filename, potcar_elements, ions_per_type, lines = config[0]
            group = config[1]
            num_lines = len(lines)
            data = self.parse_outcar_config(lines,potcar_elements, ions_per_type)
            all_data.append(data)
        return all_data

    def parse_outcar_config(self,lines,list_atom_types,ions_per_type):
        ## TODO clean up syntax to match FitSNAP3
        ## TODO clean up variable names to match input, increase clarity
        ## LIST SECTION_MARKERS AND RELATED FUNCTIONS ARE HARD-CODED!!
        ## DO NOT CHANGE UNLESS YOU KNOW WHAT YOU'RE DOING!!

        # with open('/home/megmcca/test_outcar_config.txt', 'w') as f:
        #     for row in lines:
        #         f.write(row)

        section_markers = [
            'FORCE on cell',
            'direct lattice vectors',
            'TOTAL-FORCE (eV/Angst)',
            'FREE ENERGIE OF THE ION-ELECTRON SYSTEM (eV)',
        ]

        idx_stress_vects = 0 # 4
        idx_lattice_vects = 1 # 5
        idx_force_vects = 2 # 6
        idx_energie = 3 # 7

        ## Index lines of file containing JSON data
        section_idxs = []
        list_atom_types, atom_coords, atom_forces, stress_component, all_lattice, total_energie  = \
            [], None, None, None, None, None

        natoms = sum(ions_per_type)

        ## Search entire file to create indices for each section
        ## TODO: current weakness is that if one section marker missing, we'll get one-off errors, need to make sure that can't happen (exit gracefully or set OUTCAR aside)
        for i, line in enumerate(lines):
            ## TODO refactor - probably a smarter/faster way to do the "line_test" part...
            line_test = [True if sm in line else False for sm in section_markers]
            if any(line_test):
                test_idx = [n for n, b in enumerate(line_test) if b][0]
                section_idxs.append(i)

        ## Create data dict for this config, with global information already included
        data = {}
        data['AtomTypes'] = list_atom_types  ## orig in poscar, done
        data['NumAtoms'] = natoms  ## orig in poscar, done

        ## Lattice vectors in real space
        ## Note: index to initial lattice vector output (POSCAR) in OUTCAR has already been removed.
        ## Actual vector starts one line after that, and has 3 lines
        lidx_last_lattice0 = section_idxs[idx_lattice_vects] + 1
        lidx_last_lattice1 = lidx_last_lattice0 + 3
        lines_last_lattice = lines[lidx_last_lattice0:lidx_last_lattice1]
        all_lattice = self.get_direct_lattice(lines_last_lattice)

        ## Stresses
        lidx_stresses = section_idxs[idx_stress_vects] + 14
        line_stresses = lines[lidx_stresses]
        stress_component = self.get_stresses(line_stresses)

        ## Atom coordinates and forces
        lidx_forces0 = section_idxs[idx_force_vects] + 2
        lidx_forces1 = lidx_forces0 + natoms
        lines_forces = lines[lidx_forces0:lidx_forces1]
        atom_coords, atom_forces = self.get_forces(lines_forces)

        ## Energie :-)
        ## We are getting the value without entropy
        lidx_energie = section_idxs[idx_energie] + 4
        line_energie = lines[lidx_energie]
        total_energie = self.get_energie(line_energie)

        # Here is where all the data is put together since the energy value is the last
        # one listed in each configuration.  After this, all these values will be overwritten
        # once the next configuration appears in the sequence when parsing
        data['Positions'] = atom_coords
        data['Forces'] = atom_forces
        data['Stress'] = stress_component
        data['Lattice'] = all_lattice
        data['Energy'] = total_energie

        return data

    def parse_outcar_header(self, header):
        ## These searches replace the POSCAR and POTCAR, and also check IBRION for AIMD runs
        lines_vrhfin, lines_ions_per_type = [], []
        potcar_elements, ions_per_type = [], []
        # line_ibrion, is_aimd = "", False

        for line in header:
            if "VRHFIN" in line:
                lines_vrhfin.append(line)
            elif "ions per type" in line:
                lines_ions_per_type.append(line)
            # elif "IBRION" in line:
            #     line_ibrion = line

        for line in lines_vrhfin:
            str0 = line.strip().replace("VRHFIN =", "")
            str1 = str0[:str0.find(":")]
            potcar_elements.append(str1)

        for line in lines_ions_per_type:
            str0 = line.replace("ions per type = ","").strip()
            ions_per_type = [int(s) for s in str0.split()]

        return potcar_elements, ions_per_type

    ## Ideas/plans for future versions
    def scrape_incar(self):
        ## Might be able to add this info to FitSNAP JSONs easily, will need to check compatibility
        pass

    def generate_lammps_test(self):
        ## Maybe create option where we can take scraped OUTCARs and make LAMMPS-compatible *dat files
        pass

    def vasp_namer(self):
        ## Tryiing to think of a clever naming scheme so that users can trace back where they got the file
        return

    def get_vrhfin(self, lines):
        ## Scrapes vrhfin lines to get elements
        ## These lines appear only once per element in OUTCARs
        ## Format: VRHFIN =W: 5p6s5d
        elem_list = []
        for line in lines:
            str0 = line.strip().replace("VRHFIN =", "")
            str1 = str0[:str0.find(":")]
            elem_list.append(str1)
        return elem_list

    def get_ions_per_type(self, lines):
        ions_per_type = []
        for line in lines:
            str0 = line.replace("ions per type = ","").strip()
            ions_per_type = [int(s) for s in str0.split()]
        return ions_per_type

    def get_ibrion(self, line):
        ## There should be only one of these lines (from INCAR print)
        ## IBRION value should always be first number < 10 to appear after "="
        line1 = line.split()
        idx_equals = line1.index("=")
        probably_ibrion = line1[idx_equals+1]
        if probably_ibrion.isdigit():
            if probably_ibrion == "0":
                is_aimd = True ## https://www.vasp.at/wiki/index.php/IBRION
            else:
                is_aimd = False
        else:
            print("!!WARNING: incomplete coding with scrape_ibrion, assuming not AIMD for now.")
            is_aimd = False
        return is_aimd

    def get_direct_lattice(self, lines):
        lattice_coords = []
        for i in range(0, 3):
            lattice_coords.append([float(v) for v in lines[i].split()[:3]])
        return lattice_coords

    def get_stresses(self, line):
        ## TODO check that we can assume symmetric stress tensors
        ## TODO where do we set the cell type (Bravais)
        columns = line.split()
        stress_xx, stress_yy, stress_zz = [float(c) for c in columns[2:5]]
        stress_xy, stress_yz, stress_zx = [float(c) for c in columns[5:8]]
        stresses = [[stress_xx, stress_xy, stress_zx],
                    [stress_xy, stress_yy, stress_yz],
                    [stress_zx, stress_yz, stress_zz]]
        return stresses

    def get_forces(self, lines):
        coords, forces = [], []
        for line in lines:
            x, y, z, fx, fy, fz = [float(v) for v in line.split()[:6]]
            coords.append([x, y, z])
            forces.append([fx, fy, fz])
        return coords, forces

    def get_energie(self, line):
        str0 = line[:line.rfind("energy(sigma->")].strip()
        str1 = "".join([c for c in str0 if c.isdigit() or c == "-" or c == "."])
        energie = float(str1)
        return energie

## ------------------------------------------------ JSON methods
    def check_json_labels(self):
        if not self.json_label and self.only_label:
            pt.single_print("!WARNING: \t'only_label' set to True, but empty value for 'json_label'!")
            pt.single_print("\t\t\tReverting to default naming scheme (vs?_subdirectory-name_*.json).")
            self.json_label = "vs"
            self.only_label = True

    def generate_FitSNAP_JSONs(self):
        new_converted, bad_outcar_not_converted, already_converted = 0, 0, 0
        json_path = config.sections['PATH'].datapath
        if not os.path.exists(json_path):
            os.mkdir(json_path)
        for group, outcars in self.outcars_per_group.items():
            ## Check group and group path, create if it doesn't exist
            json_group_path = json_path + "/" + group
            if not os.path.exists(json_group_path):
                os.mkdir(json_group_path)

            ## Begin OUTCAR processing
            for i, outcar in enumerate(outcars):
                pt.single_print(f"Reading: {outcar}")
                ## Get OUTCAR directory name for labeling (e.g. default naming scheme, sorting, etc.)
                outcar_path_stem = outcar.replace("/OUTCAR", "")[outcar.replace("/OUTCAR", "").rfind("/") + 1:]

                ## Create stem for JSON file naming
                if self.only_label:
                    json_file_stem = f"{json_group_path}/{self.json_label}{i}"
                else:
                    json_file_stem = f"{json_group_path}/{self.json_label}{i}_{outcar_path_stem}"
                pt.single_print(f"\tNew JSON group path and file name(s): {json_file_stem}_*.json ")

                ## Find existing JSON files
                json_files = glob(json_file_stem + "*.json")

                ## Begin converting OUTCARs to FitSNAP JSON format
                ## Credit for next sections goes to Mary Alice Cusentino's VASP2JSON script!
                if not json_files or self.overwrite:
                    ## Reading/scraping of outcar
                    data_outcar_configs, num_configs = self.scrape_outcar(outcar)

                    ## Check that all expected data in configs from OUTCAR is present
                    for n, data in enumerate(data_outcar_configs):
                        m = n + 1
                        if any([True if val is None else False for val in data.values()]):
                            pt.single_print(
                                f"!!WARNING: OUTCAR file is missing data: {outcar} \n"
                                f"!!WARNING: Continuing without writing JSON...\n")
                            self.bad_configs[group] = self.bad_configs[group] + [outcar]
                            bad_outcar_not_converted += 1
                            status = "could_not_convert"
                        else:
                            self.write_json(outcar, data, json_file_stem, m)
                            new_converted += 1
                            status = "new_converted"
                else:
                    already_converted += 1
                    status = "already_converted"
                log_info = [group, outcar, outcar_path_stem, json_file_stem, num_configs, status]
                self.log_data.append(log_info)

        pt.single_print(f"Completed writing JSON files. Summary: \n"
                        f"\t\t{new_converted} new JSON files created \n"
                        f"\t\t{already_converted} OUTCARs already converted \n"
                        f"\t\t{bad_outcar_not_converted} OUTCARs could not be converted \n"
                        f"\t\tSee {self.log_file} for more details.\n")


    ## TODO decouple data from writing of JSONs
    def write_json(self, outcar_name, outcar_data, json_filestem, file_num):
        ## Credit for next section goes to Mary Alice Cusentino's VASP2JSON script!
        json_file = f"{json_filestem}_{file_num}.json"
        self.comment_line += outcar_name

        allDataHeader = {}
        allDataHeader['Data'] = [outcar_data]
        allDataHeader['EnergyStyle'] = "electronvolt"
        allDataHeader['StressStyle'] = "kB"
        allDataHeader['AtomTypeStyle'] = "chemicalsymbol"
        allDataHeader['PositionsStyle'] = "angstrom"
        allDataHeader['ForcesStyle'] = "electronvoltperangstrom"
        allDataHeader['LatticeStyle'] = "angstrom"

        myDataset = {}

        myDataset['Dataset'] = allDataHeader

        ## Write rando comment line at start or scrape_configs crashes
        ## TODO ... (do we really need the comment line? It breaks the JSON format)
        with open(json_file, "w") as f:
            f.write(self.comment_line + "\n")

        ## Write actual JSON object
        with open(json_file, "a+") as f:
            json.dump(myDataset, f, indent=2, sort_keys=True)
        return

    # def find_vasp_files(self):
    #     ## OUTCAR finder/checker
    #     ## TODO make more flexible check for OUTCAR (maybe user can flag that?)
    #     ## Currently, this piece of code will only accept files strictly named OUTCAR
    #     ## and expects only one unique OUTCAR to exist per folder.
    #     ## It will ignore any other OUTCAR-type file (for example, "OUTCAR1").
    #     print("MEG find_vasp_files")
    #     print("self.group_table.items()", self.group_table.items())
    #     for group, path_dict in self.group_table.items():
    #         print(group, path_dict)
    #         # path = path_dict["paths_to_outcars"]
    #         self.outcars_per_group[group] = []
    #         all_files = glob(path + "/**/*", recursive=True)
    #         outcars = [f for f in all_files if f.endswith("OUTCAR")]
    #         self.outcars_per_group[group].extend(outcars)

    # def scrape_outcar0(self, outcar):
    #     ## LIST SECTION_MARKERS AND RELATED FUNCTIONS ARE HARD-CODED!!
    #     ## DO NOT CHANGE UNLESS YOU KNOW WHAT YOU'RE DOING!!
    #     section_markers = [
    #         'VRHFIN',  ## replace POSCAR
    #         'ions per type',  ## replace POSCAR
    #         'IBRION',  ## AIMD check
    #         'aborting loop because EDIFF is reached',
    #         'FORCE on cell',
    #         'direct lattice vectors',
    #         'TOTAL-FORCE (eV/Angst)',
    #         'FREE ENERGIE OF THE ION-ELECTRON SYSTEM (eV)',
    #     ]
    #     idx_vrhfin = 0
    #     idx_ions_per_type = 1
    #     idx_ibrion = 2
    #     idx_ion_steps = 3
    #     idx_stress_vects = 4
    #     idx_lattice_vects = 5
    #     idx_force_vects = 6
    #     idx_energie = 7

    #     ## Index lines of file containing JSON data
    #     section_idxs = [[] for _ in range(0, len(section_markers))]
    #     list_atom_types, atom_coords, atom_forces, stress_component, all_lattice, total_energie, natoms = \
    #         [], None, None, None, None, None, None

    #     ## Read in file and gather section marker indices
    #     with open(outcar, "r") as f:
    #         lines = f.readlines()

    #     ## Search entire file to create indices for each section
    #     for i, line in enumerate(lines):
    #         ## TODO refactor - probably a smarter/faster way to do the "line_test" part...
    #         line_test = [True if sm in line else False for sm in section_markers]
    #         if any(line_test):
    #             test_idx = [n for n, b in enumerate(line_test) if b][0]
    #             section_idxs[test_idx].append(i)

    #     ## Ignore initial lattice section (which is the static POSCAR)
    #     section_idxs[idx_lattice_vects] = section_idxs[idx_lattice_vects][1:]

    #     ## Find number of configurations in OUTCAR (e.g. for relaxations, AIMD)
    #     num_ion_steps_in_outcar = len(section_idxs[idx_ion_steps])
    #     pt.single_print(f"\t{num_ion_steps_in_outcar} ionic steps found in OUTCAR. Processing...")
    #     smallest_section = num_ion_steps_in_outcar
    #     for n, section in enumerate(section_idxs[3:]):
    #         # m = n + 3  ## shift enumeration to match section_idxs
    #         ion_steps_found_in_section = len(section)
    #         if ion_steps_found_in_section < smallest_section:
    #             smallest_section = ion_steps_found_in_section

    #     ## If run was cut short, collect only complete sets of data by fixing lengths of section_idxs
    #     if smallest_section != num_ion_steps_in_outcar:
    #         pt.single_print(f"\t!WARNING: current OUTCAR may be incomplete. "
    #                         f"{smallest_section} complete ionic steps found instead of {num_ion_steps_in_outcar}.\n"
    #                         f"\t!WARNING: Processing {smallest_section} ionic steps...")
    #         for n, section in enumerate(section_idxs[3:]):
    #             m = n + 3  ## shift enumeration to match section_idxs
    #             section_idxs[m] = section[:smallest_section]
    #     num_configs = smallest_section

    #     ## Scrapes to get element list, natoms, and list of atom types
    #     lines_vrhfin = [lines[i] for i in section_idxs[idx_vrhfin]]
    #     elements = self.get_vrhfin(lines_vrhfin)

    #     lines_ions_per_type = [lines[i] for i in section_idxs[idx_ions_per_type]]
    #     ions_per_type = self.get_ions_per_type(lines_ions_per_type)

    #     ## Check if IBRION > -1 (AIMD run), this changes how following configs are treated
    #     line_ibrion = lines[section_idxs[idx_ibrion][0]]
    #     is_aimd = self.get_ibrion(line_ibrion)

    #     list_atom_types0 = [(f"{elements[i]} " * ions_per_type[i]).strip().split() for i in range(0, len(elements))]
    #     list_atom_types = sum(list_atom_types0, [])
    #     natoms = len(list_atom_types)

    #     all_outcar_configs = []
    #     for n, step_idx in enumerate(section_idxs[idx_ion_steps]):
    #         ## Create data dict for this config, with global information already included
    #         data = {}
    #         data['AtomTypes'] = list_atom_types  ## orig in poscar, done
    #         data['NumAtoms'] = natoms  ## orig in poscar, done

    #         ## Lattice vectors in real space
    #         ## Note: index to initial lattice vector output (POSCAR) in OUTCAR has already been removed.
    #         ## Actual vector starts one line after that, and has 3 lines
    #         lidx_last_lattice0 = section_idxs[idx_lattice_vects][n] + 1
    #         lidx_last_lattice1 = lidx_last_lattice0 + 3
    #         lines_last_lattice = lines[lidx_last_lattice0:lidx_last_lattice1]
    #         all_lattice = self.get_direct_lattice(lines_last_lattice)

    #         ## Stresses
    #         lidx_stresses = section_idxs[idx_stress_vects][n] + 14
    #         line_stresses = lines[lidx_stresses]
    #         stress_component = self.get_stresses(line_stresses)

    #         ## Atom coordinates and forces
    #         lidx_forces0 = section_idxs[idx_force_vects][n] + 2
    #         lidx_forces1 = lidx_forces0 + natoms
    #         lines_forces = lines[lidx_forces0:lidx_forces1]
    #         atom_coords, atom_forces = self.get_forces(lines_forces)

    #         ## Energie :-)
    #         ## We are getting the value without entropy
    #         lidx_energie = section_idxs[idx_energie][n] + 4
    #         line_energie = lines[lidx_energie]
    #         total_energie = self.get_energie(line_energie)

    #         # Here is where all the data is put together since the energy value is the last
    #         # one listed in each configuration.  After this, all these values will be overwritten
    #         # once the next configuration appears in the sequence when parsing

    #         data['Positions'] = atom_coords
    #         data['Forces'] = atom_forces
    #         data['Stress'] = stress_component
    #         data['Lattice'] = all_lattice
    #         data['Energy'] = total_energie

    #         all_outcar_configs.append(data)
    #     return all_outcar_configs, num_configs
