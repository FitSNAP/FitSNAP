from fitsnap3lib.solvers.slate_common import SlateCommon
import numpy as np

class SlateRIDGE(SlateCommon):

    def perform_fit_ridge(self):
        
        pt = self.pt
        a = pt.shared_arrays['a'].array
        b = pt.shared_arrays['b'].array
        w = pt.shared_arrays['w'].array
        
        # Note: a, b, w remain unchanged - only aw, bw get modified by SLATE
        aw = pt.shared_arrays['aw'].array
        bw = pt.shared_arrays['bw'].array
        
        # Debug output - print all in one statement to avoid tangled output
        # *** DO NOT REMOVE !!! ***
        if self.config.debug:
            np.set_printoptions(precision=4, suppress=True, floatmode='fixed', linewidth=np.inf)
            np.set_printoptions(formatter={'float': '{:.4f}'.format})
            pt.sub_print(f"*** ------------------------\n"
                         f"pt.fitsnap_dict['Testing']\n{pt.fitsnap_dict['Testing']}\n"
                         #f"a\n{a}\n"
                         #f"b {b}\n"
                         f"--------------------------------\n")
        
        pt.sub_barrier()
        
        # -------- LOCAL SLICE OF SHARED ARRAY AND REGULARIZATION ROWS --------

        a_start_idx, a_end_idx = pt.fitsnap_dict["sub_a_indices"]
        aw_start_idx, aw_end_idx = pt.fitsnap_dict["sub_aw_indices"]
        reg_row_idx = pt.fitsnap_dict["reg_row_idx"]
        reg_col_idx = pt.fitsnap_dict["reg_col_idx"]
        reg_num_rows = pt.fitsnap_dict["reg_num_rows"]
        #pt.all_print(f"pt.fitsnap_dict {pt.fitsnap_dict}")
        if self.config.debug:
            pt.all_print(f"*** aw_start_idx {aw_start_idx} aw_end_idx {aw_end_idx} reg_row_idx {reg_row_idx} reg_col_idx {reg_col_idx} reg_num_rows {reg_num_rows}")
        
        # -------- WEIGHTS --------
  
        # Apply weights to my local slice
        local_slice = slice(a_start_idx, a_end_idx+1)
        w_local_slice = slice(aw_start_idx, (aw_end_idx-reg_num_rows+1))
        aw[w_local_slice] = w[local_slice, np.newaxis] * a[local_slice]
        bw[w_local_slice] = w[local_slice] * b[local_slice]

        # -------- TRAINING/TESTING SPLIT --------
        
        if 'Testing' in pt.fitsnap_dict and pt.fitsnap_dict['Testing'] is not None:
            testing_mask = pt.fitsnap_dict['Testing'][local_slice]
            for i in range(a_end_idx-a_start_idx+1):
                if testing_mask[i]:
                    if self.config.debug:
                        pt.all_print(f"*** removing i {i} aw_start_idx+i {aw_start_idx+i}")
                    aw[aw_start_idx+i,:] = 0.0
                    bw[aw_start_idx+i] = 0.0

        # -------- REGULARIZATION ROWS --------

        sqrt_alpha = np.sqrt(self.alpha)
        n = a.shape[1]
    
        for i in range(reg_num_rows):
            if reg_col_idx+i < n: # avoid out of bounds padding from multiple nodes
                aw[reg_row_idx+i, reg_col_idx+i] = sqrt_alpha
            bw[reg_row_idx+i] = 0.0

        # -------- SLATE AUGMENTED QR --------
        pt.sub_barrier() # make sure all sub ranks done filling local tiles
        m = aw.shape[0] * self.pt._number_of_nodes # global matrix total rows
        lld = aw.shape[0]  # local leading dimension column-major shared array
        
        np.set_printoptions(precision=3, suppress=True, floatmode='fixed', linewidth=np.inf)
        if self.config.debug:
            pt.sub_print(f"*** SENDING TO SLATE ------------------------\n"
                         f"aw\n{aw}\n"
                         f"bw {bw}\n"
                         f"--------------------------------\n")
                     
        # Determine debug flag from EXTRAS section
        debug_flag = 0
        if self.config.debug:
            debug_flag = 1
            
        slate_augmented_qr_cython(aw, bw, m, lld, debug_flag)
        
        # Broadcast solution from Node 0 to all nodes via head ranks
        if pt._sub_rank == 0:  # This rank is head of its node
            pt._head_group_comm.Bcast(bw[:n], root=0)

        self.fit = bw[:n]
                
        # *** DO NOT REMOVE !!! ***
        if self.config.debug:
            pt.all_print(f"*** self.fit ------------------------\n"
                f"{self.fit}\n-------------------------------------------------\n")
            
