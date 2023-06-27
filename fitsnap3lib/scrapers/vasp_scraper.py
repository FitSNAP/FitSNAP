from fitsnap3lib.scrapers.scrape import Scraper, convert
#from fitsnap3lib.io.input import Config
#from fitsnap3lib.parallel_tools import ParallelTools
#from fitsnap3lib.io.output import output
from copy import copy
from glob import glob
from random import shuffle
from datetime import datetime
import numpy as np
import os, json


#config = Config()
#pt = ParallelTools()

class Vasp(Scraper):

    def __init__(self, name): 
        super().__init__(name)
        pt.single_print("Initializing VASP scraper")
        self.all_data = []
        self.configs = {}
        self.bad_configs = {}
        self.all_config_dicts = []
        self.bc_bool = False
        self.infile = config.args.infile
        self.vasp_path = config.sections['PATH'].datapath
        self.use_TOTEN = config.sections["GROUPS"].vasp_use_TOTEN
        self.group_table = config.sections["GROUPS"].group_table
        self.jsonpath = config.sections['GROUPS'].vasp_json_pathname
        self.vasp_ignore_incomplete = config.sections["GROUPS"].vasp_ignore_incomplete
        self.vasp_ignore_jsons = config.sections["GROUPS"].vasp_ignore_jsons
        self.unconverged_label = config.sections["GROUPS"].vasp_unconverged_label

        if 'TRAINSHIFT' in config.sections.keys():
            self.trainshift = config.sections['TRAINSHIFT'].trainshift
            output.screen("!WARNING: 'TRAINSHIFT' is in input file!\n!WARNING: This section is used to shift (and clearly document) per-atom energies between VASP datasets.\n!WARNING: . \n!WARNING: Only use this section if you know what you're doing!")
        else:
            self.trainshift = {}


    def scrape_groups(self):
        # Locate all OUTCARs in datapath
        glob_asterisks = '/**/*'
        outcars_base = os.path.join(self.vasp_path, *glob_asterisks.split('/'))

        # TODO make this search user-specify-able (e.g., OUTCARs have labels/prefixes etc.)
        all_outcars = [f for f in glob(outcars_base,recursive=True) if f.endswith('OUTCAR')]

        # Grab test|train split
        self.group_dict = {k: config.sections['GROUPS'].group_types[i] for i, k in enumerate(config.sections['GROUPS'].group_sections)}

        for group in self.group_table:
            # First, check that all group folders exist
            group_vasp_path = f'{self.vasp_path}/{group}'
            if not os.path.exists(group_vasp_path):
                raise Exception('!!ERROR: group folder not detected!!' 
                    '\n!!Please check that all groups in the input file have an associated group folder' 
                    f'\n!!\tInput file: {self.infile}'
                    f'\n!!\tMissing group folder: {group_vasp_path}'
                    '\n')

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
            
            # Grab OUTCARS for this training group
            # Test filepath to be sure that unique group name is being matched
            group_outcars = [f for f in all_outcars if group_vasp_path + '/' in f]
            if len(group_outcars) == 0:
                raise Exception('!!ERROR: no OUTCARs found in group!!' 
                    '\n!!Please check that all groups in the input file have at least one file named "OUTCAR"' 
                    '\n!!in at least one subdirectory of the group folder' 
                    f'\n!!\tMissing group data root: {group_vasp_path}'
                    '\n')

            file_base = os.path.join(config.sections['PATH'].datapath, group)
            self.files[file_base] = group_outcars
            self.configs[group] = []  

            for outcar in self.files[file_base]:
                # Open file
                with open(outcar, 'r') as fp:
                    lines = fp.readlines()
                nlines = len(lines)

                # Use ion loop text to partition ionic steps
                ion_loop_text = 'aborting loop'
                start_idx_loops = [i for i, line in enumerate(lines) if ion_loop_text in line]
                converged_list = [False if 'unconverged' in lines[i] else True for i in start_idx_loops]
                end_idx_loops = [i for i in start_idx_loops[1:]] + [nlines]

                # Grab potcar and element info
                header_lines = lines[:start_idx_loops[0]]
                potcar_list, potcar_elements, ions_per_type, is_duplicated = self.parse_outcar_header(header_lines)

                # Each config in a single OUTCAR is assigned the same
                # parent data (i.e. filename, potcar and ion data)
                # but separated for each iteration (idx loops on 'lines')
                # Tuple data: outcar file name str, config number int, starting line number (for debug)  int, 
                # potcar list, potcar elements list, number ions per element list, configuration lines list 
                unique_configs = [(outcar, i, start_idx_loops[i], potcar_list, potcar_elements, ions_per_type, converged_list[i],
                                    lines[start_idx_loops[i]:end_idx_loops[i]])
                                    for i in range(0, len(start_idx_loops))]
                
                # Avoid adding degenerate structures (for different energies) to training set
                # Take only final one for JSON
                # See 'parse_outcar_header' method, the IBRION and NSW check, for more details
                if is_duplicated:
                    unique_configs = unique_configs[-1:]

                # Parse and process OUTCAR data per configuration
                for uc in unique_configs:
                    config_dict = self.generate_config_dict(group, uc)
                    if config_dict != -1: 
                        self.configs[group].append(config_dict)
                del lines

            # If random_sampling toggled on, shuffle training and testing data
            if config.sections["GROUPS"].random_sampling:
                shuffle(self.configs[group], pt.get_seed)
            nconfigs = len(self.configs[group])

            # Assign configurations to train/test groups
            if training_size == 1:
                training_configs = nconfigs
                testing_configs = 0
            else:
                training_configs = max(1, int(round(training_size * nconfigs)))
                if training_configs == nconfigs:
                    # If training_size is not exactly 1.0, add at least 1 testing config
                    training_configs -= 1
                    testing_configs = 1
                else:
                    testing_configs = nconfigs - training_configs

            if nconfigs - testing_configs - training_configs < 0:
                raise ValueError("training configs: {} + testing configs: {} is greater than files in folder: {}".format(
                    training_configs, testing_configs, nconfigs))

            output.screen(f"{group}: Detected {nconfigs}, fitting on {training_configs}, testing on {testing_configs}")

            # Populate tests dictionary
            if self.tests is None:
                self.tests = {}
            self.tests[group] = []

            for i in range(testing_configs):
                self.tests[group].append(self.configs[group].pop())

            self.group_table[group]['training_size'] = training_configs
            self.group_table[group]['testing_size'] = testing_configs


    def scrape_configs(self):
        """Generate and send (mutable) data to send to fitsnap"""
        self.conversions = copy(self.default_conversions)
        for i, data0 in enumerate(self.configs):
            data = data0[0]
            
            assert len(data) == 1, "More than one object (dataset) is in this file"

            self.data = data['Dataset']

            assert len(self.data['Data']) == 1, "More than one configuration in this dataset"

            assert all(k not in self.data for k in self.data["Data"][0].keys()), \
                "Duplicate keys in dataset and data"
            
            # Move self.data up one level
            self.data.update(self.data.pop('Data')[0])  

            for key in self.data:
                if "Style" in key:
                    if key.replace("Style", "") in self.conversions:
                        temp = config.sections["SCRAPER"].properties[key.replace("Style", "")]
                        temp[1] = self.data[key]
                        self.conversions[key.replace("Style", "")] = convert(temp)

            for key in config.sections["SCRAPER"].properties:
                if key in self.data:
                    self.data[key] = np.asarray(self.data[key])

            natoms = np.shape(self.data["Positions"])[0]
            pt.shared_arrays["number_of_atoms"].sliced_array[i] = natoms
            self.data["QMLattice"] = (self.data["Lattice"] * self.conversions["Lattice"]).T

            # Populate with LAMMPS-normalized lattice
            del self.data["Lattice"]  

            # TODO Check whether "Label" container useful to keep around
            if "Label" in self.data:
                del self.data["Label"] 

            # TODO test this with VASP files (taken from JSON scraper)
            # Insert electronegativities, which are per-atom scalars
            # if (self.config.sections["CALCULATOR"].per_atom_scalar):
            #    if not isinstance(self.data["Chis"], float):
            #        self.data["Chis"] = self.data["Chis"]

            if not isinstance(self.data["Energy"], float):
                self.data["Energy"] = float(self.data["Energy"])

            # Currently, ESHIFT should be in units of your training data (note there is no conversion)
            if hasattr(config.sections["ESHIFT"], 'eshift'):
                for atom in self.data["AtomTypes"]:
                    self.data["Energy"] += config.sections["ESHIFT"].eshift[atom]

            self.data["test_bool"] = self.test_bool[i]

            self.data["Energy"] *= self.conversions["Energy"]

            self._rotate_coords()
            self._translate_coords()

            self._weighting(natoms)

            self.all_data.append(self.data)

        return self.all_data

    def generate_config_dict(self, group, outcar_config):
        """If no JSON has been created, create dictionary for each configuration contained in a single OUTCAR"""
        """Otherwise, read existing JSON (unless input file variable 'vasp_overwrite_jsons' is toggled to True)"""
        # TODO future: create CSV of converted files and check that (instead of loading full OUTCAR again)
        config_dict = {}
        is_bad_config = False
        has_json = None
        outcar_filename, config_num, start_idx, potcar_list, potcar_elements, ions_per_type, converged, lines = outcar_config
        file_num = config_num + 1

        # JSON read/write setup
        json_path = f'{self.jsonpath}/{group}'
        json_filestem = outcar_filename.replace('/','_').replace('_OUTCAR','') #.replace(f'_{group}','')
        if converged:
            json_filename = f"{json_path}/{json_filestem}_{file_num}.json"
        else:
            if self.unconverged_label != '\'\'':
                json_filename = f"{json_path}/{json_filestem}_{file_num}_{self.unconverged_label}.json"
            else:
                json_filename = f"{json_path}/{json_filestem}_{file_num}.json"

        # Check if JSON was already created from this OUTCAR
        if not os.path.exists(json_path):
            os.makedirs(json_path)
            has_json = False
        else:
            has_json = os.path.exists(json_filename)

        if has_json and not self.vasp_ignore_jsons:
            with open(json_filename, 'r') as f:
                config_dict = json.loads(f.read(), parse_constant=True)
            return config_dict
        else:
            config_data = self.parse_outcar_config(lines, potcar_list, potcar_elements, ions_per_type)
            if type(config_data) == tuple:
                crash_type, crash_line = config_data
                is_bad_config = True
                if not self.vasp_ignore_incomplete:
                    raise Exception('!!ERROR: Incomplete OUTCAR configuration found!!' 
                        '\n!!Not all atom coordinates/forces were written to a configuration' 
                        '\n!!Please check the OUTCAR for incomplete steps and adjust, '
                        '\n!!or toggle variable "vasp_ignore_incomplete" to True'
                        '\n!!(not recommended as you may miss future incomplete steps)' 
                        f'\n!!\tOUTCAR location: {outcar_filename}' 
                        f'\n!!\tConfiguration number: {config_num}' 
                        f'\n!!\tLine number of error: {start_idx}' 
                        f'\n!!\tExpected {crash_type}, {crash_line} '
                        '\n')
                else:
                    output.screen('!!WARNING: Incomplete OUTCAR configuration found!!'
                        '\n!!Not all atom coordinates/coordinates were written to a configuration'
                        '\n!!Variable "vasp_ignore_incomplete" is toggled to True'
                        '\n!!Note that this may result in missing training set data (e.g., missing final converged structures)'
                        f'\n!!\tOUTCAR location: {outcar_filename}' 
                        f'\n!!\tConfiguration number: {config_num}'
                        f'\n!!\tLine number of warning: {start_idx}'
                        f'\n!!\tExpected {crash_type}, {crash_line} '
                        '\n')

            config_header = {}
            config_header['Group'] = group
            config_header['File'] = json_filename
            config_header['use_TOTEN'] = self.use_TOTEN
            config_header['EnergyStyle'] = "electronvolt"
            config_header['StressStyle'] = "kB"
            config_header['AtomTypeStyle'] = "chemicalsymbol"
            config_header['PositionsStyle'] = "angstrom"
            config_header['ForcesStyle'] = "electronvoltperangstrom"
            config_header['LatticeStyle'] = "angstrom"
            config_header['Data'] = [config_data]

            config_dict['Dataset'] = config_header

            if not is_bad_config:
                self.write_json(json_filename, outcar_filename, config_dict)
                return config_dict
            else:
                return -1

    def parse_outcar_config(self,lines,potcar_list,potcar_elements,ions_per_type):
        # Many thanks to Mary Alice Cusentino and Logan Williams for parts of the following code
        # Based on the VASP2JSON.py script in the 'tools' directory
        # LIST SECTION_MARKERS AND RELATED FUNCTIONS ARE HARD-CODED!!
        # DO NOT CHANGE UNLESS YOU KNOW WHAT YOU'RE DOING!!
        section_markers = [
            'FORCE on cell',
            'direct lattice vectors',
            'TOTAL-FORCE (eV/Angst)',
            'FREE ENERGIE OF THE ION-ELECTRON SYSTEM (eV)',
        ]

        section_names = [
            'stresses',
            'lattice',
            'coords & forces',
            'energie',
        ]

        idx_stress_vects = 0
        idx_lattice_vects = 1 
        idx_force_vects = 2 
        idx_energie = 3

        # Index lines of file containing JSON data
        section_idxs = [None,None,None,None]
        atom_coords, atom_forces, stress_component, all_lattice, total_energie  = None, None, None, None, None

        list_atom_types = []
        for i, elem in enumerate(potcar_elements):
            num_repeats = ions_per_type[i]
            elem_list = [elem.strip()]*num_repeats
            list_atom_types.extend(elem_list)
        natoms = sum(ions_per_type)

        # Search entire config/ionic step to create indices for each section
        for i, line in enumerate(lines):
            line_test = [True if sm in line else False for sm in section_markers]
            if any(line_test):
                test_idx = [n for n, b in enumerate(line_test) if b][0]
                section_idxs[test_idx] = i
        
        # If this config has any sections missing, it is incomplete, return crash
        missing_sections = [True if i == None else False for i in section_idxs]
        if any(missing_sections):
            crash_type = '4 sections'
            missing_sections_str = 'missing sections: '
            missing_sections_str += ', '.join([section_names[i] for i, b in enumerate(missing_sections) if b])
            del lines
            return (crash_type, missing_sections_str)

        # Create data dict for this config, with global information already included
        data = {}
        data['AtomTypes'] = list_atom_types  # orig in poscar, done
        data['NumAtoms'] = natoms  # orig in poscar, done

        # Lattice vectors in real space
        # Note: index to initial lattice vector output (POSCAR) in OUTCAR has already been removed.
        # Actual vector starts one line after that, and has 3 lines
        lidx_last_lattice0 = section_idxs[idx_lattice_vects] + 1
        lidx_last_lattice1 = lidx_last_lattice0 + 3
        lines_last_lattice = lines[lidx_last_lattice0:lidx_last_lattice1]
        all_lattice = self.get_direct_lattice(lines_last_lattice)

        # Stresses
        lidx_stresses = section_idxs[idx_stress_vects] + 14
        line_stresses = lines[lidx_stresses]
        stress_component = self.get_stresses(line_stresses)

        # Atom coordinates and forces
        lidx_forces0 = section_idxs[idx_force_vects] + 2
        lidx_forces1 = lidx_forces0 + natoms
        lines_forces = lines[lidx_forces0:lidx_forces1]
        atom_coords, atom_forces = self.get_forces(lines_forces)
        if type(atom_coords) == str:
            crash_type = 'atom coords, atom forces'
            crash_atom_coord_line = 'found bad line: ' + atom_coords
            del lines
            return (crash_type, crash_atom_coord_line)

        # Energie WITH entropy (TOTEN)
        lidx_TOTEN = section_idxs[idx_energie] + 2
        line_TOTEN = lines[lidx_TOTEN]
        total_energie_with_entropy = self.get_energie_with_entropy(line_TOTEN)

        # Energie without entropy
        lidx_energie = section_idxs[idx_energie] + 4
        line_energie = lines[lidx_energie]
        total_energie_without_entropy = self.get_energie_without_entropy(line_energie)

        # Check if we should use the default without entropy, or use TOTEN 
        # (when vasp_use_TOTEN = True in [GROUPS])
        if self.use_TOTEN:
            total_energie = total_energie_with_entropy
        else:
            total_energie = total_energie_without_entropy

        # Special toggled shift in energie if converting training data
        if self.trainshift:
            # Shift energies
            shifted_energies = [self.trainshift[element] for element in potcar_elements]
            energy_shifts = [ions_per_type[i]*shifted_energies[i] for i in range(len(potcar_elements))]
            total_energie += sum(energy_shifts)

        # Here is where all the data is put together since the energy value is the last
        # one listed in each configuration.  After this, all these values will be overwritten
        # once the next configuration appears in the sequence when parsing
        data['Positions'] = atom_coords
        data['Forces'] = atom_forces
        data['Stress'] = stress_component
        data['Lattice'] = all_lattice
        data['Energy'] = total_energie
        data["computation_code"] = "VASP"
        data["pseudopotential_information"] = potcar_list

        # Clean up (othewrise we get memory errors)
        del lines

        return data

    def parse_outcar_header(self, header):
        # These searches replace the POSCAR and POTCAR, and can also check IBRION for AIMD runs (commented out now)
        lines_potcar, lines_vrhfin, lines_ions_per_type = [], [],[]
        potcar_list, potcar_elements, ions_per_type = [], [], []
        ibrion, nsw = None, None 
        is_duplicated = False
        # scf: self-consistent framework - ions don't move, electrons 'moved' until convergence criterion reached
        # if IBRION = -1 and NSW > 0, VASP sometimes prints duplicate structures, check for this

        for line in header:
            if "VRHFIN" in line:
                lines_vrhfin.append(line)
            elif "ions per type" in line:
                lines_ions_per_type.append(line)
            elif "POTCAR" in line:
                lines_potcar.append(line)
                # Look for the ordering of the atom types - grabbing POTCAR filenames first, then atom labels separately because VASP has terribly inconsistent formatting
                # OUTCARs have the following lines printed twice, and we don't need to append them the second time
                if line.split()[1:] not in potcar_list:  
                    potcar_list.append(line.split()[1:])  

            # TODO add check that warns user if POSCAR elements and POTCAR order are not the same (if possible)
            elif 'IBRION' in line:
                ibrion = self.get_ibrion(line)
            elif 'NSW' in line:
                nsw = self.get_nsw(line)

        if ibrion == -1 and nsw > 0:
            output.screen('!WARNING: degenerate energies on same structure!\n'
                            '!This can happen when IBRION = -1 and NSW > 0.\n'
                            f'!(your settings: IBRION = {ibrion}, NSW = {nsw})\n'
                            f'!Jumping to final (optimized) configuration.\n\n')
            is_duplicated = True

        for line in lines_vrhfin:
            str0 = line.strip().replace("VRHFIN =", "")
            str1 = str0[:str0.find(":")]
            potcar_elements.append(str1.strip())

        for line in lines_ions_per_type:
            str0 = line.replace("ions per type = ","").strip()
            ions_per_type = [int(s) for s in str0.split()]

        return potcar_list, potcar_elements, ions_per_type, is_duplicated

    def get_vrhfin(self, lines):
        # Scrapes vrhfin lines to get elements
        # These lines appear only once per element in OUTCARs
        # Format: VRHFIN =W: 5p6s5d
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
        # There should be only one of these lines (from INCAR print)
        # IBRION value should always be first number < 10 to appear after "="
        line1 = line.split()
        idx_equals = line1.index("=")
        probably_ibrion = line1[idx_equals+1]
        return int(probably_ibrion)
    
    def get_nsw(self, line):
        # There should be only one of these lines (from INCAR print)
        # Format:   >NSW    =    100    number of steps for IOM
        line1 = line.split()
        idx_equals = line1.index("=")
        probably_nsw = line1[idx_equals+1]
        return int(probably_nsw)

    def get_direct_lattice(self, lines):
        lattice_coords = []
        for i in range(0, 3):
            lattice_coords.append([float(v) for v in lines[i].split()[:3]])
        return lattice_coords

    def get_stresses(self, line):
        # TODO check that we can assume symmetric stress tensors
        # TODO where do we set the cell type (Bravais)
        columns = line.split()
        stress_xx, stress_yy, stress_zz = [float(c) for c in columns[2:5]]
        stress_xy, stress_yz, stress_zx = [float(c) for c in columns[5:8]]
        stresses = [[stress_xx, stress_xy, stress_zx],
                    [stress_xy, stress_yy, stress_yz],
                    [stress_zx, stress_yz, stress_zz]]
        return stresses

    def get_forces(self, lines):
        coords, forces = [], []
        try:
            [float(v) for v in lines[-1].split()[:6]]
        except:
            print('OUTCAR config did not run to completion! Discarding configuration')
            # Return faulty string for error message
            return lines[-1],-1 

        for line in lines:
            x, y, z, fx, fy, fz = [float(v) for v in line.split()[:6]]
            coords.append([x, y, z])
            forces.append([fx, fy, fz])
        return coords, forces

    def get_energie_without_entropy(self, line):
        str0 = line[:line.rfind("energy(sigma->")].strip()
        str1 = "".join([c for c in str0 if c.isdigit() or c == "-" or c == "."])
        energie = float(str1)
        return energie


    def get_energie_with_entropy(self, line):
        energie_with_entropy = float(line.split()[4])
        return energie_with_entropy
    
    def write_json(self, json_filename, outcar_filename, config_dict):
        dt = datetime.now().strftime('%B %d %Y %I:%M%p')

        # TODO future versions, include generation metadata in JSON file
        comment_line = f'# Generated on {dt} from: {os.getcwd()}/{outcar_filename}'

        # Write JSON object
        # Note: older versions of FitSNAP wrote JSONs with a leading comment line
        # If you need that, use append mode ('a+') instead of write mode below
        with open(json_filename, 'w') as f:
            json.dump(config_dict, f, indent=2, sort_keys=True)
        return


# ------------------------ Ideas/plans for future versions ------------------------ #
    def scrape_incar(self):
        # Might be able to add this info to FitSNAP JSONs easily, will need to check compatibility
        pass


    def only_read_JSONs_if_OUTCAR_already_converted(self):
        # Many OUTCARs (esp. for AIMD) are HUGE and take a long time to parse. Design a user-friendly and elegant way to confirm that OUTCAR has already been parsed and only read available JSON data (see above)
        pass

    def check_OUTCAR_filesizes(self):
        # Many OUTCARs (esp. for AIMD) are HUGE and take a long time to parse. In these cases, strongly recommend to user to toggle vasp2json on, and then use JSONs only, or have an elegant way to only read available JSON data (see above)
        pass

    def generate_lammps_test(self):
        # Maybe create option where we can take scraped OUTCARs and make LAMMPS-compatible *dat files right away
        pass

    def vasp_namer(self):
        # Trying to think of a clever naming scheme so that users can trace back where they got the file
        return