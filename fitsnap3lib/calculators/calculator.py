#from fitsnap3lib.parallel_tools import double_size, DistributedList, stubs #, ParallelTools
from fitsnap3lib.parallel_tools import DistributedList
#from fitsnap3lib.io.input import Config
#from fitsnap3lib.io.output import output
import numpy as np
import pandas as pd


#config = Config()
#pt = ParallelTools()


class Calculator:
    """ Class for allocating, calculating, and collating descriptors. """

    def __init__(self, name, pt, config):
        self.pt = pt #ParallelTools()
        self.config = config #Config()
        self.name = name
        self.number_of_atoms = None
        self.number_of_files_per_node = None
        self.shared_index = None
        self.distributed_index = 0

    def get_width(self):
        pass
    
    def create_dicts(self, nconfigs):
        """
        Create dictionaries for certain distributed lists.
        Each list should be of size `nconfigs` of a single proc.

        Args:
            nconfigs: int number of configs on this proc.
        """

        # Lists of length number of configs.
        self.pt.add_2_fitsnap("Groups", DistributedList(nconfigs))
        self.pt.add_2_fitsnap("Configs", DistributedList(nconfigs))
        self.pt.add_2_fitsnap("Testing", DistributedList(nconfigs))

    def allocate_per_config(self, data: list):
        """
        Allocate shared arrays for total number of atoms. This is only needed 
        when doing big A matrix fits (need number of atoms) or nonlinear 
        fits.

        Args:
            data: List of data dictionaries.
        """
        ncpn = self.pt.get_ncpn(len(data))

        self.pt.create_shared_array('number_of_atoms', ncpn, dtype='i')
        self.pt.slice_array('number_of_atoms')

        # number of dgrad rows serves similar purpose as number of atoms
        
        self.pt.create_shared_array('number_of_dgrad_rows', ncpn, dtype='i')
        self.pt.slice_array('number_of_dgrad_rows')

        # number of neighs serves similar purpose as number of atoms for custom calculator
        
        self.pt.create_shared_array('number_of_neighs_scrape', ncpn, dtype='i')
        self.pt.slice_array('number_of_neighs_scrape')

        # Loop through data and set sliced number of atoms.
        for i, configuration in enumerate(data):
            natoms = np.shape(configuration["Positions"])[0]
            self.pt.shared_arrays["number_of_atoms"].sliced_array[i] = natoms

    def create_a(self):
        """
        Allocate shared arrays for calculator.
        """

        pt = self.pt

        # TODO : Any extra config pulls should be done before this

        pt.sub_barrier()
        # total number of atoms in all configs, summed
        self.number_of_atoms = pt.shared_arrays["number_of_atoms"].array.sum()
        # total number of configs on all procs in a node
        self.number_of_files_per_node = len(pt.shared_arrays["number_of_atoms"].array)
        # self.nconfigs is the number of configs on this proc, assigned in lammps_base
        
        

        # create data matrices for nonlinear pytorch solver

        if (self.config.sections["SOLVER"].solver == "PYTORCH"):
            a_len = 0
            b_len = 0 # number reference energies for all configs
            c_len = 0 # number of reference forces for all configs
            dgrad_len = 0
            if self.config.sections["CALCULATOR"].energy:
                energy_rows = self.number_of_files_per_node
                if self.config.sections["CALCULATOR"].per_atom_energy:
                    energy_rows = self.number_of_atoms # total number of atoms in all configs
                a_len += energy_rows
                b_len += self.number_of_files_per_node # total number of configs

            if self.config.sections["CALCULATOR"].force:
                c_len += 3*self.number_of_atoms
                dgrad_len += pt.shared_arrays["number_of_dgrad_rows"].array.sum()

            if self.config.sections["CALCULATOR"].per_atom_scalar:

                # in this case we fitting NNs only to per-atom scalars, not to energies/forces

                a_len += self.number_of_atoms # total number of atoms in all configs


#            # stress fitting not supported yet.
#            if config.sections["CALCULATOR"].stress:
#                a_len += self.number_of_files_per_node * 6
#                b_len += self.number_of_files_per_node * 6

            a_width = self.get_width()
            assert isinstance(a_width, int)

            # TODO: Pick a method to get RAM accurately (pt.get_ram() seems to get RAM wrong on Blake)

            a_size = ( (a_len * a_width) + (dgrad_len * a_width) ) * pt.double_size
            if self.config.args.verbose:
                pt.single_print(">>> Matrix of descriptors and descriptor derivatives takes up ", 
                              "{:.4f}".format(100 * a_size / self.config.sections["MEMORY"].memory),
                              "% of the total memory:", 
                              "{:.4f}".format(self.config.sections["MEMORY"].memory*1e-9), "GB")
            if a_size / pt.get_ram() > 0.5 and not self.config.sections["MEMORY"].override:
                raise MemoryError("The descriptor matrix is larger than 50% of your RAM. \n Aborting...!")
            elif a_size / pt.get_ram() > 0.5 and self.config.sections["MEMORY"].override:
                pt.single_print("Warning: > 50% RAM. I hope you know what you are doing!")

            pt.create_shared_array('a', a_len, a_width)
            pt.create_shared_array('b', b_len)
            pt.create_shared_array('c', c_len)
            pt.create_shared_array('w', b_len, 2)
            pt.create_shared_array('t', a_len, 1)
            if self.config.sections["CALCULATOR"].per_atom_scalar:
                # create per-atom scalar arrays
                pt.create_shared_array('pas', a_len, 1)

            #if self.config.sections["CALCULATOR"].force:
            pt.create_shared_array('dgrad', dgrad_len, a_width)
            pt.create_shared_array('dbdrindx', dgrad_len, 3)

            # make an index for which the 'a' array starts on a particular proc
            pt.new_slice_a()
            self.shared_index = pt.fitsnap_dict["sub_a_indices"][0] 
            # make an index for which the 'b' array starts on a particular proc
            pt.new_slice_b()
            self.shared_index_b = pt.fitsnap_dict["sub_b_indices"][0] 
            # make an index for which the 'c' array starts on a particular proc
            pt.new_slice_c()
            self.shared_index_c = pt.fitsnap_dict["sub_c_indices"][0] 
            #pt.new_slice_t() # atom types

            # make an index for which the 'dgrad' array starts on a particular proc
            pt.new_slice_dgrad()
            self.shared_index_dgrad = pt.fitsnap_dict["sub_dgrad_indices"][0]

            # create fitsnap dicts - distributed lists of size nconfig per proc
            # these later get gathered on the root proc in calculator.gather_distributed_lists

            pt.add_2_fitsnap("Groups", DistributedList(self.nconfigs))
            pt.add_2_fitsnap("Configs", DistributedList(self.nconfigs))
            pt.add_2_fitsnap("NumAtoms", DistributedList(self.nconfigs))
            pt.add_2_fitsnap("NumDgradRows", DistributedList(self.nconfigs))
            pt.add_2_fitsnap("Testing", DistributedList(self.nconfigs))

        # get data arrays for network solvers

        elif (self.config.sections["SOLVER"].solver == "NETWORK"):

            a_len = 0 # per-atom quantities (types, numneighs) for all configs
            b_len = 0 # number reference energies for all configs
            c_len = 0 # number of reference forces for all configs
            c_width = 0 # 3 if fitting to forces
            neighlist_len = 0 # number of neighbors for all configs

            a_len += self.number_of_atoms # total number of atoms in all configs
            neighlist_len += pt.shared_arrays["number_of_neighs_scrape"].array.sum()

            if self.config.sections["CALCULATOR"].energy:
                b_len += self.number_of_files_per_node # total number of configs

            if self.config.sections["CALCULATOR"].force:
                c_len += 3*self.number_of_atoms

#            # stress fitting not supported yet.
#            if config.sections["CALCULATOR"].stress:
#                a_len += self.number_of_files_per_node * 6
#                b_len += self.number_of_files_per_node * 6

            a_width = 2 # types and numneighs
            neighlist_width = self.get_width()
            assert isinstance(neighlist_width, int)

            # TODO: Pick a method to get RAM accurately (pt.get_ram() seems to get RAM wrong on Blake)
            a_size = (neighlist_len * neighlist_width + 2 * c_len * c_width) * pt.double_size
            if self.config.args.verbose:
                pt.single_print(">>> Matrix of data takes up ", "{:.4f}".format(100 * a_size / self.config.sections["MEMORY"].memory),
                              "% of the total memory:", "{:.4f}".format(self.config.sections["MEMORY"].memory*1e-9), "GB")
            if a_size / pt.get_ram() > 0.5 and not self.config.sections["MEMORY"].override:
                raise MemoryError("The data memory larger than 50% of your RAM. \n Aborting...!")
            elif a_size / pt.get_ram() > 0.5 and self.config.sections["MEMORY"].override:
                pt.single_print("Warning: > 50 % RAM. I hope you know what you are doing!")

            # create shared arrays
            a_width = 5
            neighlist_width = 2 # i j 
            xneigh_width = 3 # xj yj zj, with PBC corrections
            pt.create_shared_array('a', a_len, a_width)
            pt.create_shared_array('neighlist', neighlist_len, neighlist_width)
            pt.create_shared_array('xneigh', neighlist_len, xneigh_width)
            pt.create_shared_array('transform_x', neighlist_len, xneigh_width)
            pt.create_shared_array('b', b_len)
            pt.create_shared_array('x', c_len)
            pt.create_shared_array('w', b_len, 2)
            pt.create_shared_array('t', a_len, 1)
            pt.create_shared_array('positions', a_len, 3)
            
            # also need descriptors for network standardization
            # for pairwise networks, there are num_neigh*num_descriptors total descriptors to store
            # TODO: if statement here to catch possibilities for custom networks, e.g. nonpairwise descriptors, etc.

            pt.create_shared_array('descriptors', neighlist_len, self.config.sections['CUSTOM'].num_descriptors)

            if self.config.sections["CALCULATOR"].force:
                pt.create_shared_array('c', c_len)

            # make an index for which the 'a' array starts on a particular proc
            pt.new_slice_a()
            self.shared_index = pt.fitsnap_dict["sub_a_indices"][0] 
            # make an index for which the 'b' array starts on a particular proc
            pt.new_slice_b()
            self.shared_index_b = pt.fitsnap_dict["sub_b_indices"][0] 
            # make an index for which the 'c' array starts on a particular proc
            pt.new_slice_c()
            self.shared_index_c = pt.fitsnap_dict["sub_c_indices"][0] 
            #pt.new_slice_t() # atom types
            # make an index for which the 'neighlist' array starts on a particular proc
            pt.new_slice_neighlist()
            self.shared_index_neighlist = pt.fitsnap_dict["sub_neighlist_indices"][0] 

            # create fitsnap dicts - distributed lists of size nconfig per proc
            # these later get gathered on the root proc in calculator.gather_distributed_lists

            pt.add_2_fitsnap("Groups", DistributedList(self.nconfigs))
            pt.add_2_fitsnap("Configs", DistributedList(self.nconfigs))
            pt.add_2_fitsnap("NumAtoms", DistributedList(self.nconfigs))
            pt.add_2_fitsnap("NumNeighs", DistributedList(self.nconfigs))
            pt.add_2_fitsnap("Testing", DistributedList(self.nconfigs))

        # get data arrays for linear solvers

        else:

            a_len = 0
            if self.config.sections["CALCULATOR"].energy:
                energy_rows = self.number_of_files_per_node
                if self.config.sections["CALCULATOR"].per_atom_energy:
                    energy_rows = self.number_of_atoms
                a_len += energy_rows
            if self.config.sections["CALCULATOR"].force:
                a_len += 3 * self.number_of_atoms
            if self.config.sections["CALCULATOR"].stress:
                a_len += self.number_of_files_per_node * 6

            a_width = self.get_width()
            assert isinstance(a_width, int)
            
            if (self.config.sections["SOLVER"].solver.lower() == "slate"):
            
                # TODO: dont bother checking memory on multinode for now
                # if it blows up just get more nodes
                            
                # preallocate more rows for augmented QR
                # do hard work upfront to avoid pain later
                    
                # spread regularization rows across nodes/ranks instead of all
                # at bottom of global matrix to balance tiles for SLATE
                    
                # waste a little memory to be more robust
                # optimal is same amount of configs per node
                    
                max_a_len = pt._comm.allreduce(a_len, op=pt.MPI.MAX)
                aw_len = int(np.ceil((max_a_len*pt._number_of_nodes + a_width)/pt._number_of_nodes))
                # Store info about SLATE augmentation
                pt.add_2_fitsnap("is_slate", True)
                pt.create_shared_array('aw', aw_len, a_width, order='F')
                pt.create_shared_array('bw', aw_len, order='F')
               
              
            else:

              # TODO: Pick a method to get RAM accurately (pt.get_ram() seems to get RAM wrong on Blake)
              a_size = a_len * a_width * pt.double_size
              if self.config.args.verbose:
                  pt.single_print(">>> Matrix of descriptors takes up ", "{:.4f}".format(100 * a_size / self.config.sections["MEMORY"].memory),
                              "% of the total memory:", "{:.4f}".format(self.config.sections["MEMORY"].memory*1e-9), "GB") #, "on rank", "{:d}".format(pt._rank))
              if a_size / pt.get_ram() > 0.5 and not self.config.sections["MEMORY"].override:
                  raise MemoryError("The descriptor matrix is larger than 50% of your RAM. \n Aborting...!")
              elif a_size / pt.get_ram() > 0.5 and self.config.sections["MEMORY"].override:
                  pt.single_print("Warning: > 50 % RAM. I hope you know what you are doing!")
            
            pt.create_shared_array('a', a_len, a_width)
            pt.create_shared_array('b', a_len)
            pt.create_shared_array('w', a_len)
            pt.new_slice_a()
            self.shared_index = pt.fitsnap_dict["sub_a_indices"][0]
            # pt.slice_array('a')

            pt.add_2_fitsnap("Groups", DistributedList(pt.fitsnap_dict["sub_a_size"]))
            pt.add_2_fitsnap("Configs", DistributedList(pt.fitsnap_dict["sub_a_size"]))
            pt.add_2_fitsnap("Row_Type", DistributedList(pt.fitsnap_dict["sub_a_size"]))
            pt.add_2_fitsnap("Atom_I", DistributedList(pt.fitsnap_dict["sub_a_size"]))
            pt.add_2_fitsnap("Testing", DistributedList(pt.fitsnap_dict["sub_a_size"]))
            pt.add_2_fitsnap("Atom_Type", DistributedList(pt.fitsnap_dict["sub_a_size"]))

    def process_configs(self, data, i):
        pass

    def preprocess_configs(self, data, i):
        pass

    def preprocess_allocate(self, nconfigs):
        pass

    #@staticmethod # NOTE: Does this need to be a static method?
    def collect_distributed_lists(self, allgather: bool=False):
        """
        Gathers all the distributed lists on each proc to the root proc.
        For each distributed list (fitsnap dicts) this will create a concatenated list on the root proc.
        We use this function in fitsnap.py after processing configs.

        Args:
            allgather: Whether to gather lists on all nodes or just the head node.
        """   
        for key in self.pt.fitsnap_dict.keys():
            if isinstance(self.pt.fitsnap_dict[key], DistributedList):
                self.pt.gather_fitsnap(key)
                if self.pt.fitsnap_dict[key] is not None and self.pt.stubs != 1:
                    self.pt.fitsnap_dict[key] = [item for sublist in self.pt.fitsnap_dict[key] for item in sublist]
                elif self.pt.fitsnap_dict[key] is not None:
                    self.pt.fitsnap_dict[key] = self.pt.fitsnap_dict[key].get_list()

    #@pt.rank_zero
    def extras(self):
        @self.pt.rank_zero
        def extras():
            if self.config.sections["EXTRAS"].dump_a:
                np.save(self.config.sections['EXTRAS'].descriptor_file, self.pt.shared_arrays['a'].array)
            if self.config.sections["EXTRAS"].dump_b:
                np.save(self.config.sections['EXTRAS'].truth_file, self.pt.shared_arrays['b'].array)
            if self.config.sections["EXTRAS"].dump_w:
                np.save(self.config.sections['EXTRAS'].weights_file, self.pt.shared_arrays['w'].array)
            if self.config.sections["EXTRAS"].dump_dataframe:
                df = pd.DataFrame(self.pt.shared_arrays['a'].array)
                df['truths'] = self.pt.shared_arrays['b'].array.tolist()
                df['weights'] = self.pt.shared_arrays['w'].array.tolist()
                for key in self.pt.fitsnap_dict.keys():
                    if isinstance(self.pt.fitsnap_dict[key], list) and len(self.pt.fitsnap_dict[key]) == len(df.index):
                        df[key] = self.pt.fitsnap_dict[key]
                df.to_pickle(self.config.sections['EXTRAS'].dataframe_file)
                del df
        if "EXTRAS" in self.config.sections:
            extras()

        # if not config.sections["SOLVER"].detailed_errors:
        #     print(
        #         ">>>Enable [SOLVER], detailed_errors = 1 to characterize the training/testing split of your output *.npy matricies")
