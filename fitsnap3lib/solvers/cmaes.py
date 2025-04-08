from fitsnap3lib.solvers.solver import Solver
from fitsnap3lib.calculators.lammps_reaxff import LammpsReaxff
import time, cma, itertools, functools
import numpy as np
import pandas as pd
from pprint import pprint

# ------------------------------------------------------------------------------------------------

def _loss_function_values(x):
    return reaxff_calculator.process_configs_with_values(x)

# ------------------------------------------------------------------------------------------------

class CMAES(Solver):

    # --------------------------------------------------------------------------------------------

    def __init__(self, name, pt, config):

        super().__init__(name, pt, config, linear=False)
        self.popsize = self.config.sections['SOLVER'].popsize
        self.sigma = self.config.sections['SOLVER'].sigma
        self.reaxff_io = self.config.sections['REAXFF']

        calculator = config.sections["CALCULATOR"]
        self._hsic_header=['iteration', *self.reaxff_io.parameter_names]
        if calculator.energy: self._hsic_header.append('energy')
        if calculator.force: self._hsic_header.append('force')
        if calculator.charge: self._hsic_header.append('charge')
        if calculator.dipole: self._hsic_header.append('dipole')
        if calculator.quadrupole: self._hsic_header.append('quadrupole')
        if calculator.bond_order: self._hsic_header.append('bond_order')
        self._hsic_header.append('total')
        self._hsic_data = []
        self._last_log_time = time.time()

    # --------------------------------------------------------------------------------------------

    def _loss_function(self, x):

        # print(f"*** rank {self.pt._rank} ok 2a")

        np.set_printoptions(precision=4, linewidth=2000)
        
        if self.pt.stubs==1:
            answer = _loss_function_values(x).tolist()
        else:
            futures = [self.executor.submit(_loss_function_values, x) for _ in self.range_workers]
            results = np.vstack([f.result() for f in futures])
            answer = np.sum(results, axis=0).tolist()
            #print(f"*** rank {self.pt._rank} results {results}")

        self._hsic_data.append([self._iteration, *x, *answer])
        #print(f"*** rank {self.pt._rank} iteration {self._iteration} answer {answer}")
        return answer[-1]

    # --------------------------------------------------------------------------------------------

    def perform_fit(self, fs):

        # "Avoid global variables that are declared outside class methods or attributes"
        # no way around this sorry
        global reaxff_calculator
        reaxff_calculator = fs.calculator
        x0 = self.reaxff_io.values
        #print( x0 )
        self.output = fs.output
        self.range_workers = range(1, self.pt.get_size())

        import warnings
        warnings.simplefilter("ignore", category=UserWarning)

        bounds = self.config.sections['REAXFF'].parameter_bounds
        cma_stds = np.array([self.sigma * (hi - lo) for lo, hi in bounds])

        options={
          'popsize': self.popsize, 'seed': 12345, #'maxiter': 5,
          'bounds': list(np.transpose(bounds)), 'CMA_stds': cma_stds
        }

        from concurrent.futures import ThreadPoolExecutor
        self.io_executor = ThreadPoolExecutor(max_workers=1)
        self._iteration = 1

        if self.pt._rank == 0:
            constraints_function = self.build_constraints_lambda()
            cfun = cma.ConstrainedFitnessAL(self._loss_function, constraints_function )

        if self.pt.stubs==0:
            # SAFER TO USE *MPICommExecutor* INSTEAD OF *MPIPoolExecutor*
            # [https://mpi4py.readthedocs.io/en/stable/mpi4py.futures.html#mpicommexecutor]
            from mpi4py import MPI
            from mpi4py.futures import MPICommExecutor

            with MPICommExecutor(MPI.COMM_WORLD, root=0) as self.executor:
                if self.executor is not None:
                    # parallel_objective=self._parallel_loss_function
                    _, es = cma.fmin2( cfun, x0, self.sigma, callback=self._log, options=options)
        else:
            _, es = cma.fmin2( None, x0, self.sigma,
                parallel_objective=self._parallel_loss_function, options=options)

        if self.pt._rank == 0:
            self.fit = self.reaxff_io.change_parameters_string(es.best.x)
            #self.errors = pd.DataFrame(self._hsic_data, columns=self._hsic_header)
            self._log_best(es)

        self.io_executor.shutdown(wait=True)

    # --------------------------------------------------------------------------------------------

    def _log(self, es):

        self._iteration = es.countiter + 1
        now = time.time()

        if es.countiter==1:
            self._log_best(es)

        if (now - self._last_log_time) >= 120:
            self._log_best(es)
            current_fit = self.reaxff_io.change_parameters_string(es.best.x)
            # offload i/o to a background thread and keep optimization loop going without stalling
            df = pd.DataFrame(self._hsic_data, columns=self._hsic_header)
            self.io_executor.submit(self.output.output, current_fit, df)
            self._hsic_data = []
            self._last_log_time = now

    # --------------------------------------------------------------------------------------------

    def _log_best(self, es):

        print("-----------------------------------------------------------------------")
        print(self.config.sections['CALCULATOR'].charge_fix)
        print("PARAMETER_NAME           INITIAL  LOWER_BOUND        NOW    UPPER_BOUND")
        for p, x0i, xbi in zip(self.reaxff_io.parameter_names, es.x0, es.best.x):
            p_bounds = self.reaxff_io.bounds[p.split('.')[-1]]
            print(f"{p:<22} {x0i: > 9.4f}  [  {p_bounds[0]: > 8.2f} {xbi: > 13.8f}  {p_bounds[1]: > 8.2f} ]")
        print(f"------------------------- {es.countiter:<7} {es.best.f:11.9g} -------------------------")

    # --------------------------------------------------------------------------------------------

    def error_analysis(self):

        #for i in self.range_all_data:
        #    d = reaxff_calculator.set_data_index(i)
        #    reaxff_calculator.process_data_for_parameter_values(0, self.x_best)


        #all_data = reaxff_calculator.pt.fitsnap_dict["Data"]
        #for subgroup in all_data:
        #  for c in subgroup['configs']:
        #      reaxff_calculator.process_configs(c, 99)
        #self.errors = all_data

        pass

    # --------------------------------------------------------------------------------------------

    def build_constraints_lambda(self):
        param_idx = {p: i for i, p in enumerate(self.config.sections["REAXFF"].parameter_names)}
        elements = reaxff_calculator.elements
        constraints = []
        constraints_desc = []

        # -------- 1. ATM r_s ≥ r_p ≥ r_pp --------
        # -------- 2. ATM r_vdw ≥ r_s --------

        for e in elements:

            key_r_s = f'ATM.{e}.r_s'
            key_r_p = f'ATM.{e}.r_p'
            key_r_pp = f'ATM.{e}.r_pp'
            key_r_vdw = f'ATM.{e}.r_vdw'

            if key_r_s in param_idx and key_r_p in param_idx:
                i1, i2 = param_idx[key_r_s], param_idx[key_r_p]
                constraints.append((lambda i1=i1, i2=i2: (lambda x: x[i1] - x[i2]))())
                constraints_desc.append(f"(1a) {key_r_s} ≥ {key_r_p}")

            if key_r_p in param_idx and key_r_pp in param_idx:
                i1, i2 = param_idx[key_r_p], param_idx[key_r_pp]
                constraints.append((lambda i1=i1, i2=i2: (lambda x: x[i1] - x[i2]))())
                constraints_desc.append(f"(1b) {key_r_p} ≥ {key_r_pp}")

            if key_r_p not in param_idx and key_r_s in param_idx and key_r_pp in param_idx:
                i1, i2 = param_idx[key_r_s], param_idx[key_r_pp]
                constraints.append((lambda i1=i1, i2=i2: (lambda x: x[i1] - x[i2]))())
                constraints_desc.append(f"(1c) {key_r_s} ≥ {key_r_pp}")

            if key_r_s in param_idx and key_r_vdw in param_idx:
                i1, i2 = param_idx[key_r_s], param_idx[key_r_vdw]
                constraints.append((lambda i1=i1, i2=i2: (lambda x: x[i2] - x[i1]))())
                constraints_desc.append(f"(2) {key_r_vdw} ≥ {key_r_s}")

        # -------- 3. ATM r_vdw > rcore2 --------
        # -------- 4. ATM ecore2 > epsilon --------

        for e in elements:
            key_rcore2 = f'ATM.{e}.rcore2'
            key_r_vdw = f'ATM.{e}.r_vdw'
            key_ecore2 = f'ATM.{e}.ecore2'
            key_epsilon = f'ATM.{e}.epsilon'

            if key_rcore2 in param_idx and key_r_vdw in param_idx:
                i1, i2 = param_idx[key_rcore2], param_idx[key_r_vdw]
                constraints.append((lambda i1=i1, i2=i2: lambda x: x[i2] - x[i1] - 0.01)())
                constraints_desc.append(f"(3) {key_r_vdw} > {key_rcore2}")

            if key_ecore2 in param_idx and key_epsilon in param_idx:
                i3, i4 = param_idx[key_ecore2], param_idx[key_epsilon]
                constraints.append((lambda i3=i3, i4=i4: lambda x: x[i3] - x[i4])())
                constraints_desc.append(f"(4) {key_ecore2} > {key_epsilon}")

        # -------- 5. ATM eta > 7.2*gamma (QEQ), eta > 8.13*gamma (ACKS2) --------

        factor = 8.13 if "acks2" in reaxff_calculator.charge_fix else 7.2

        for e in elements:

            key_gamma = f'ATM.{e}.gamma'
            key_eta   = f'ATM.{e}.eta'

            if key_gamma in param_idx and key_eta in param_idx:
                i_gamma = param_idx[key_gamma]
                i_eta   = param_idx[key_eta]
                constraints.append(lambda x, i1=i_gamma, i2=i_eta: x[i2] - factor * x[i1])
                constraints_desc.append(f"(5) {key_eta} ≥ {factor} * {key_gamma}")

        # -------- 6. ATM bcut_acks2 ≥ max(r_s) --------

        if "acks2" in reaxff_calculator.charge_fix:

            rs = [f'ATM.{e}.r_s' for e in elements if f'ATM.{e}.r_s' in param_idx]

            if rs:

                rs_indices = [param_idx[r] for r in rs]

                def max_rs_constraint(i_bcut, rs_indices=rs_indices):
                    return lambda x: x[i_bcut] - max(x[i] for i in rs_indices)

                for e in elements:
                    key_bcut = f'ATM.{e}.bcut_acks2'
                    if key_bcut in param_idx:
                        i_bcut = param_idx[key_bcut]
                        constraints.append(max_rs_constraint(i_bcut))
                        constraints_desc.append(f"(6) {key_bcut} ≥ max({', '.join(rs)})")

        # -------- 7. electronegativity hierarchy (FIXME EXTEND TO ALL ATOMS) --------
        # -------- example: chi(O) > chi(N) > chi(C) > chi(H) --------

        hierarchy = ['F', 'O', 'Cl', 'N', 'S', 'C', 'P', 'H', 'Mg', 'Ca', 'Na', 'K']
        present = [e for e in hierarchy if f'ATM.{e}.chi' in param_idx]

        for e1, e2 in zip(present[:-1], present[1:]):
            k1 = f'ATM.{e1}.chi'
            k2 = f'ATM.{e2}.chi'
            if k1 in param_idx and k2 in param_idx:
                i1, i2 = param_idx[k1], param_idx[k2]
                constraints.append(lambda x, i1=i1, i2=i2: x[i1] - x[i2] - 0.01)
                constraints_desc.append(f"(7) {k1} > {k2}")

        # -------- 8. BND De_s ≥ De_p ≥ De_pp --------

        bnd_De = {}  # BND.H.O: {"De_s": idx, "De_p": idx, "De_pp": idx}
        for pname in param_idx:
            if pname.startswith("BND.") and (".De_s" in pname or ".De_p" in pname or ".De_pp" in pname):
                base = ".".join(pname.split(".")[:3])
                if base not in bnd_De:
                    bnd_De[base] = {}
                if pname.endswith(".De_s"):
                    bnd_De[base]["De_s"] = param_idx[pname]
                elif pname.endswith(".De_p"):
                    bnd_De[base]["De_p"] = param_idx[pname]
                elif pname.endswith(".De_pp"):
                    bnd_De[base]["De_pp"] = param_idx[pname]

        for base, idxs in bnd_De.items():

            if "De_s" in idxs and "De_p" in idxs:
                i1, i2 = idxs["De_s"], idxs["De_p"]
                constraints.append((lambda i1=i1, i2=i2: lambda x: x[i1] - x[i2])())
                constraints_desc.append(f"(8a) {base}.De_s ≥ {base}.De_p")

            if "De_p" in idxs and "De_pp" in idxs:
                i1, i2 = idxs["De_p"], idxs["De_pp"]
                constraints.append((lambda i1=i1, i2=i2: lambda x: x[i1] - x[i2])())
                constraints_desc.append(f"(8b) {base}.De_p ≥ {base}.De_pp")

            if "De_p" not in idxs and "De_s" in idxs and "De_pp" in idxs:
                i1, i2 = idxs["De_s"], idxs["De_pp"]
                constraints.append((lambda i1=i1, i2=i2: lambda x: x[i1] - x[i2])())
                constraints_desc.append(f"(8c) {base}.De_s ≥ {base}.De_pp")

        # -------- 9. BND p_be2 ≥ p_bo2, p_bo4, p_bo6 --------

        for pname in param_idx:
            if pname.endswith(".p_be2"):
                base = ".".join(pname.split(".")[:3])
                key_be2 = f"{base}.p_be2"
                key_bo2 = f"{base}.p_bo2"
                key_bo4 = f"{base}.p_bo4"
                key_bo6 = f"{base}.p_bo6"

                if key_bo2 in param_idx:
                    i1 = param_idx[key_be2]
                    i2 = param_idx[key_bo2]
                    constraints.append((lambda i1=i1, i2=i2: lambda x: x[i1] - x[i2])())
                    constraints_desc.append(f"(9a) {key_be2} ≥ {key_bo2}")

                if key_bo4 in param_idx:
                    i1 = param_idx[key_be2]
                    i2 = param_idx[key_bo4]
                    constraints.append((lambda i1=i1, i2=i2: lambda x: x[i1] - x[i2])())
                    constraints_desc.append(f"(9b) {key_be2} ≥ {key_bo4}")

                if key_bo6 in param_idx:
                    i1 = param_idx[key_be2]
                    i2 = param_idx[key_bo6]
                    constraints.append((lambda i1=i1, i2=i2: lambda x: x[i1] - x[i2])())
                    constraints_desc.append(f"(9c) {key_be2} ≥ {key_bo6}")

        # -------- 10. BND p_bo6 ≥ p_bo4 ≥ p_bo2 --------

        for pname in param_idx:
            if pname.endswith(".p_bo2") or pname.endswith(".p_bo4") or pname.endswith(".p_bo6"):
                base = ".".join(pname.split(".")[:3])
                key_bo2 = f"{base}.p_bo2"
                key_bo4 = f"{base}.p_bo4"
                key_bo6 = f"{base}.p_bo6"

                if key_bo4 in param_idx and key_bo2 in param_idx:
                    i1 = param_idx[key_bo4]
                    i2 = param_idx[key_bo2]
                    constraints.append((lambda i1=i1, i2=i2: lambda x: x[i1] - x[i2])())
                    constraints_desc.append(f"(10a) {key_bo4} ≥ {key_bo2}")

                if key_bo6 in param_idx and key_bo4 in param_idx:
                    i1 = param_idx[key_bo6]
                    i2 = param_idx[key_bo4]
                    constraints.append((lambda i1=i1, i2=i2: lambda x: x[i1] - x[i2])())
                    constraints_desc.append(f"(10b) {key_bo6} ≥ {key_bo4}")

                if key_bo4 not in param_idx and key_bo6 in param_idx and key_bo2 in param_idx:
                    i1 = param_idx[key_bo6]
                    i2 = param_idx[key_bo2]
                    constraints.append((lambda i1=i1, i2=i2: lambda x: x[i1] - x[i2])())
                    constraints_desc.append(f"(10c) {key_bo6} ≥ {key_bo2}")

        # -------- 11. OFD r_pp ≥ r_p ≥ r_s --------

        ofd_keys = {}
        for pname in param_idx:
            if pname.startswith("OFD."):
                base = ".".join(pname.split('.')[:3])
                if base not in ofd_keys:
                    ofd_keys[base] = {}
                suffix = pname.split('.')[-1]
                if suffix in ("r_s", "r_p", "r_pp"):
                    ofd_keys[base][suffix] = param_idx[pname]

        for base, idxs in ofd_keys.items():

            if "r_pp" in idxs and "r_p" in idxs:
                constraints.append(lambda x, i1=idxs["r_pp"], i2=idxs["r_p"]: x[i1] - x[i2])
                constraints_desc.append(f"(11a) {base}.r_pp ≥ {base}.r_p")

            if "r_p" in idxs and "r_s" in idxs:
                constraints.append(lambda x, i1=idxs["r_p"], i2=idxs["r_s"]: x[i1] - x[i2])
                constraints_desc.append(f"(11b) {base}.r_p ≥ {base}.r_s")

            if "r_p" not in idxs and "r_pp" in idxs and "r_s" in idxs:
                constraints.append(lambda x, i1=idxs["r_pp"], i2=idxs["r_s"]: x[i1] - x[i2])
                constraints_desc.append(f"(11c) {base}.r_p ≥ {base}.r_s")

        # -------- 12. OFD.X.Y.alpha ≥ max(ATM.X.alpha, ATM.Y.alpha) --------

        for pname in param_idx:
            if pname.startswith("OFD.") and pname.endswith(".alpha"):
                parts = pname.split('.')
                if len(parts) >= 4:
                    atom1 = parts[1]
                    atom2 = parts[2]
                    ofd_idx = param_idx[pname]

                    atm1_key = f"ATM.{atom1}.alpha"
                    atm2_key = f"ATM.{atom2}.alpha"

                    if atm1_key in param_idx and atm2_key in param_idx:
                        atm1_idx = param_idx[atm1_key]
                        atm2_idx = param_idx[atm2_key]

                        constraints.append(
                            (lambda ofd_idx=ofd_idx, atm1_idx=atm1_idx, atm2_idx=atm2_idx:
                                lambda x: x[ofd_idx] - max(x[atm1_idx], x[atm2_idx])
                            )()
                        )
                        constraints_desc.append(f"(12) {pname} ≥ max({atm1_key}, {atm2_key})")

        # -------- 13. ANG p_pen1 ≥ |p_val1| --------
        # -------- 14. ANG p_pen1 ≥ |p_val2| --------

        for pname in param_idx:
            if pname.startswith("ANG.") and pname.endswith(".p_pen1"):
                base = ".".join(pname.split(".")[:-1])
                k_pen = pname
                k_val1 = f"{base}.p_val1"
                k_val2 = f"{base}.p_val2"
                if k_val1 in param_idx and k_val2 in param_idx:
                    i_pen = param_idx[k_pen]
                    i_val1 = param_idx[k_val1]
                    i_val2 = param_idx[k_val2]
                    constraints.append(lambda x, i_pen=i_pen, i_val1=i_val1, i_val2=i_val2: x[i_pen] - abs(x[i_val1]))
                    constraints_desc.append(f"(13) {k_pen} ≥ |{k_val1}|")
                    constraints.append(lambda x, i_pen=i_pen, i_val2=i_val2: x[i_pen] - abs(x[i_val2]))
                    constraints_desc.append(f"(14) {k_pen} ≥ |{k_val2}|")

        # -------- 15. TOR V1 ≥ |V2| --------
        # -------- 16. TOR V1 ≥ |V3| --------

        for pname in param_idx:
            if pname.startswith("TOR.") and pname.endswith(".V1"):
                key_v1 = pname
                base = ".".join(pname.split(".")[:-1])
                key_v2 = f"{base}.V2"
                key_v3 = f"{base}.V3"

                if key_v2 in param_idx:
                    i_v1 = param_idx[key_v1]
                    i_v2 = param_idx[key_v2]
                    constraints.append((lambda i_v1=i_v1, i_v2=i_v2: lambda x: x[i_v1] - abs(x[i_v2]))())
                    constraints_desc.append(f"(15) {key_v1} ≥ |{key_v2}|")

                if key_v3 in param_idx:
                    i_v1 = param_idx[key_v1]
                    i_v3 = param_idx[key_v3]
                    constraints.append((lambda i_v1=i_v1, i_v3=i_v3: lambda x: x[i_v1] - abs(x[i_v3]))())
                    constraints_desc.append(f"(16) {key_v1} ≥ |{key_v3}|")

        # -------- 17. HBD r0_hb > r_s (ensure H-bond length exceeds σ-bond radius) --------

        for pname in param_idx:
            if pname.startswith("HBD.") and pname.endswith(".r0_hb"):
                i_r0 = param_idx[pname]
                # parse donor atom from e.g., HBD.H.O.H.r0_hb
                parts = pname.split('.')
                if len(parts) >= 4:
                    donor = parts[1]
                    key_rs = f'ATM.{donor}.r_s'
                    if key_rs in param_idx:
                        i_rs = param_idx[key_rs]
                        constraints.append(lambda x, i_r0=i_r0, i_rs=i_rs: x[i_r0] - x[i_rs] - 0.01)
                        constraints_desc.append(f"(17) {pname} > {key_rs}")

        # -------- PRINT CONSTRAINTS --------

        if self.pt._rank==0:
            print("-----------------------------------------------------------------------------")
            print(f"PARAMETER CONSTRAINTS APPLIED: {len(constraints)} total")
            for d in constraints_desc: print(f"{d}")
            print("-----------------------------------------------------------------------------")

        return lambda x: [f(x) for f in constraints]
