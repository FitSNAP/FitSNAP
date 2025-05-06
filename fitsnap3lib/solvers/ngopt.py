from fitsnap3lib.solvers.solver import Solver
import time
import nevergrad as ng
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor

class NGOpt(Solver):

    # --------------------------------------------------------------------------------------------

    def __init__(self, name, pt, config):
        super().__init__(name, pt, config, linear=False)
        self.reaxff_io = self.config.sections['REAXFF']
        self.popsize = self.config.sections['SOLVER'].popsize or 8 * len(self.reaxff_io.parameters)
        self.sigma = self.config.sections['SOLVER'].sigma

        calc = config.sections["CALCULATOR"]
        self._hsic_header = ['iteration', *self.reaxff_io.parameter_names]
        if calc.energy: self._hsic_header.append('energy')
        if calc.force: self._hsic_header.append('force')
        if calc.charge: self._hsic_header.append('charge')
        if calc.dipole: self._hsic_header.append('dipole')
        if calc.quadrupole: self._hsic_header.append('quadrupole')
        if calc.esp: self._hsic_header.append('esp')
        if calc.bond_order: self._hsic_header.append('bond_order')
        self._hsic_header.append('total')
        self._hsic_data = []
        self._last_log_time = time.time()

    # --------------------------------------------------------------------------------------------

    def _loss_function(self, x):

        if self.pt.stubs == 1 or self.pt._size == 1:
            loss = self.reaxff_calculator.process_configs_with_values(x)
            self._hsic_data.append([self._iteration, *x, *loss])
            return loss[-1]

        x = self.pt._comm.bcast(x if self.pt._rank == 0 else None, root=0)
        if x is None: return False
        local = self.reaxff_calculator.process_configs_with_values(x)
        total = np.zeros_like(local) if self.pt._rank == 0 else None
        self.pt._comm.Reduce(local, total, root=0)
        #print(f"*** rank {self.pt._rank} local {local} total {total}")

        if self.pt._rank == 0:
            self._hsic_data.append([self._iteration, *x, *total.tolist()])
            return total[-1]
        return True

    # --------------------------------------------------------------------------------------------

    def _transform_water(self, x):
        """Forward transform: full physical array → transformed optimization space (13D)."""
        return np.array([
            x[0],                          # GEN.bond_softness
            x[1],                          # ATM.H.r_s
            x[2],                          # ATM.H.gamma
            x[3],                          # ATM.H.chi
            x[4] - 8.13 * x[2],            # delta_eta_H
            x[6],                          # ATM.O.gamma
            x[8] - 8.13 * x[6],            # delta_eta_O
            x[7] - x[3],                   # delta_chi_O
            x[5],                          # ATM.O.r_s
            x[9] - x[1],                   # delta_rhb_H
            x[11] - x[5],                  # delta_rhb_O
            x[10],                         # HBD.H.H.O.p_hb1
            x[12]                          # HBD.O.H.O.p_hb1
        ], dtype=np.float64)

    def _inverse_transform_water(self, y):
        """Inverse transform: transformed optimization vector → full physical parameter array (13D)."""
        x = np.array([
            y[0],                          # GEN.bond_softness
            y[1],                          # ATM.H.r_s
            y[2],                          # ATM.H.gamma
            y[3],                          # ATM.H.chi
            8.13 * y[2] + y[4],            # ATM.H.eta
            y[8],                          # ATM.O.r_s
            y[5],                          # ATM.O.gamma
            y[3] + y[7],                   # ATM.O.chi
            8.13 * y[5] + y[6],            # ATM.O.eta
            y[1] + y[9],                   # HBD.H.H.O.r0_hb
            y[10],                         # HBD.H.H.O.p_hb1
            y[8] + y[10],                  # HBD.O.H.O.r0_hb
            y[12]                          # HBD.O.H.O.p_hb1
        ], dtype=np.float64)

        np.set_printoptions(precision=4, suppress=True, linewidth=2000)

        if np.any(np.isnan(x)) or np.any(np.isinf(x)):
            print("*** _inverse_transform WARNING: NaN or inf detected in x")
        #print(f"*** y {y}")
        #print(f"*** x {x}")
        return x

    # --------------------------------------------------------------------------------------------

    def _transform(self, x):
        """Forward transform: full physical array → transformed optimization space (13D)."""
        return np.array([
            x[0],                          # GEN.bond_softness
            x[1],                          # ATM.N.r_s
            x[2],                          # ATM.N.gamma
            x[3],                          # ATM.N.chi
            x[4] - 8.13 * x[2],            # delta_eta_N
            x[6],                          # ATM.O.gamma
            x[8] - 8.13 * x[6],            # delta_eta_O
            x[7] - x[3],                   # delta_chi_O
            x[5],                          # ATM.O.r_s
        ], dtype=np.float64)

    def _inverse_transform(self, y):
        """Inverse transform: transformed optimization vector → full physical parameter array (13D)."""
        x = np.array([
            y[0],                          # GEN.bond_softness
            y[1],                          # ATM.N.r_s
            y[2],                          # ATM.N.gamma
            y[3],                          # ATM.N.chi
            8.13 * y[2] + y[4],            # ATM.N.eta
            y[8],                          # ATM.O.r_s
            y[5],                          # ATM.O.gamma
            y[3] + y[7],                   # ATM.O.chi
            8.13 * y[5] + y[6],            # ATM.O.eta
        ], dtype=np.float64)

        np.set_printoptions(precision=4, suppress=True, linewidth=2000)

        if np.any(np.isnan(x)) or np.any(np.isinf(x)):
            print("*** _inverse_transform WARNING: NaN or inf detected in x")
        #print(f"*** y {y}")
        #print(f"*** x {x}")
        return x

    # --------------------------------------------------------------------------------------------

    def perform_fit(self, fs):
        self.reaxff_calculator = fs.calculator
        if self.pt._size > 1 and self.pt._rank != 0:
            while self._loss_function(None): pass
            return

        self.output = fs.output
        self.initial_x = np.array(self.reaxff_io.values)
        bounds = self.config.sections['REAXFF'].parameter_bounds
        lb, ub = map(np.array, zip(*bounds))
        self._iteration = 1
        self.io_executor = ThreadPoolExecutor(max_workers=1)

        import logging
        logging.getLogger("nevergrad").setLevel(logging.DEBUG)

        lower = np.array([
            400.0,   # GEN.bond_softness
            0.3,     # ATM.N.r_s
            0.5,     # ATM.N.gamma
           -10.0,    # ATM.N.chi
            1e-4,    # delta_eta_N
            0.5,     # ATM.O.gamma
            1e-4,    # delta_eta_O
            1e-4,    # delta_chi_O
            0.3,     # ATM.O.r_s
        #   1e-4,    # delta_rhb_H
        #   1e-4,    # delta_rhb_O
        #  -10.0,    # HBD.H.H.O.p_hb1
        #  -10.0     # HBD.O.H.O.p_hb1
        ])

        upper = np.array([
            600.0,   # GEN.bond_softness
            2.0,     # ATM.N.r_s
            20.0,    # ATM.N.gamma
            10.0,    # ATM.N.chi
            10.0,    # delta_eta_N
            20.0,    # ATM.O.gamma
            10.0,    # delta_eta_O
            5.0,    # delta_chi_O
            2.0,     # ATM.O.r_s
        #    1.5,    # delta_rhb_H
        #    1.5,    # delta_rhb_O
        #    0.0,    # HBD.H.H.O.p_hb1
        #    0.0     # HBD.O.H.O.p_hb1
        ])

        param = ng.p.Array(init=self._transform(self.initial_x), mutable_sigma=False)
        sigma=(upper - lower) / 3.0
        #print(f"*** sigma {sigma}")
        param.set_mutation(sigma=sigma)
        param.set_bounds(lower, upper)

        from nevergrad.optimization.optimizerlib import ConfPortfolio, MetaCMA, TBPSA, PSO, MetaModel

        optimizer = ConfPortfolio(
            optimizers=[MetaCMA, TBPSA, PSO],
            warmup_ratio=1.0
        )(parametrization=param, budget=2**31-1, num_workers=1)
        optimizer.parametrization.random_state = np.random.RandomState(12345)
        self.hsic_logger = ng.callbacks.HSICLoggerCallback()
        #optimizer.register_callback("tell", self.hsic_logger)
        optimizer.register_callback("tell", self._log)
        #self._constraints = self._build_constraints()
        transformed_loss = lambda y: self._loss_function(self._inverse_transform(y))

        def transformed_loss_log(y):
            x = self._inverse_transform(y)
            loss = self._loss_function(x)
            #print(f"*** x {x} loss {loss}")
            return loss

        best = optimizer.minimize(transformed_loss_log, verbosity=0)
        best_x = self._inverse_transform(best._value)
        best_loss = best.loss
        self.fit = self.reaxff_io.change_parameters_string(best_x)
        self._log_best(self.initial_x, best_x, best_loss, optimizer.num_ask)
        self.io_executor.shutdown(wait=True)
        self.pt._comm.bcast(None, root=0)


    # --------------------------------------------------------------------------------------------

    def _log(self, optimizer, candidate, loss):

        self._iteration = optimizer.num_ask + 1
        now = time.time()

        def format_duration(seconds):
            d, r = divmod(int(seconds), 86400)
            h, m = divmod(r, 3600)
            return f"{d:02}-{h:02}:{m//60:02}"

        # Example
        # elapsed = 93784  # seconds
        # print(format_duration(elapsed))  # e.g. "01-02:03" for 1 day, 2 hours, 3 minutes

        if optimizer.num_tell==1:
            best = optimizer.recommend()
            self._log_best(self.initial_x, self._inverse_transform(best._value), best.loss, optimizer.num_ask)

        if (now - self._last_log_time) >= 1*60:
            print(self.hsic_logger.summary())
            best = optimizer.recommend()
            best_x = self._inverse_transform(best.value.copy())
            best_loss = best.loss
            hsic_data_copy = self._hsic_data
            self._hsic_data = []
            self._last_log_time = now

            def do_logging():
                self._log_best(self.initial_x, best_x, best_loss, optimizer.num_ask)
                current_fit = self.reaxff_io.change_parameters_string(best_x)
                df = pd.DataFrame(hsic_data_copy, columns=self._hsic_header)
                self.output.output(current_fit, df)

            self.io_executor.submit(do_logging)

    # --------------------------------------------------------------------------------------------

    def _log_best(self, x_initial, x_best, f_best, num_ask):
        #print(f"------------------------- {iteration:<7} {f_best:11.9g} -------------------------")

        print(f"------------------------- {num_ask:<7} {f_best} -------------------------")
        print(self.config.sections['CALCULATOR'].charge_fix)
        print("PARAMETER_NAME           INITIAL  LOWER_BOUND        NOW    UPPER_BOUND")
        for p, x0i, xbi in zip(self.reaxff_io.parameter_names, x_initial, x_best):
            b = self.reaxff_io.bounds[p.split('.')[-1]]
            print(f"{p:<22} {x0i: >9.4f}  [  {b[0]: >8.2f} {xbi: >13.8f}  {b[1]: >8.2f} ]")
        print("-----------------------------------------------------------------------")

    # --------------------------------------------------------------------------------------------

    def _build_constraints(self):
        param_idx = {p: i for i, p in enumerate(self.config.sections["REAXFF"].parameter_names)}
        elements = self.reaxff_calculator.elements
        constraints = []
        constraints_desc = []

        def greater_than(lhs_idx, rhs_idx, delta=0.01):
            return lambda x: x[rhs_idx] - x[lhs_idx] + delta

        def greater_than_or_equal(lhs_idx, rhs_idx):
            return lambda x: x[rhs_idx] - x[lhs_idx]

        def greater_than_or_equal_factor(lhs_idx, rhs_idx, factor):
            return lambda x: factor * x[rhs_idx] - x[lhs_idx]

        def greater_than_or_equal_abs(lhs_idx, rhs_idx):
            return lambda x: abs(x[rhs_idx]) - x[lhs_idx]

        def bcut_acks2_ge_max_rs(i_bcut, rs_indices):
            return lambda x: max(x[i] for i in rs_indices) - x[i_bcut]

        def ofd_alpha_ge_max_atm(ofd_idx, atm1_idx, atm2_idx):
            return lambda x: max(x[atm1_idx], x[atm2_idx]) - x[ofd_idx]

        # -------- 1. ATM r_s ≥ r_p ≥ r_pp --------
        # -------- 2. ATM r_vdw ≥ r_s --------

        for e in elements:
            key_r_s = f'ATM.{e}.r_s'
            key_r_p = f'ATM.{e}.r_p'
            key_r_pp = f'ATM.{e}.r_pp'
            key_r_vdw = f'ATM.{e}.r_vdw'

            if key_r_s in param_idx and key_r_p in param_idx:
                constraints.append(greater_than_or_equal(param_idx[key_r_s], param_idx[key_r_p]))
                constraints_desc.append(f"(1a) {key_r_s} ≥ {key_r_p}")

            if key_r_p in param_idx and key_r_pp in param_idx:
                constraints.append(greater_than_or_equal(param_idx[key_r_p], param_idx[key_r_pp]))
                constraints_desc.append(f"(1b) {key_r_p} ≥ {key_r_pp}")

            if key_r_p not in param_idx and key_r_s in param_idx and key_r_pp in param_idx:
                constraints.append(greater_than_or_equal(param_idx[key_r_s], param_idx[key_r_pp]))
                constraints_desc.append(f"(1c) {key_r_s} ≥ {key_r_pp}")

            if key_r_s in param_idx and key_r_vdw in param_idx:
                constraints.append(greater_than_or_equal(param_idx[key_r_vdw], param_idx[key_r_s]))
                constraints_desc.append(f"(2) {key_r_vdw} ≥ {key_r_s}")

        # -------- 3. ATM r_vdw > rcore2 --------
        # -------- 4. ATM ecore2 > epsilon --------

        for e in elements:
            key_rcore2 = f'ATM.{e}.rcore2'
            key_r_vdw = f'ATM.{e}.r_vdw'
            key_ecore2 = f'ATM.{e}.ecore2'
            key_epsilon = f'ATM.{e}.epsilon'

            if key_rcore2 in param_idx and key_r_vdw in param_idx:
                constraints.append(greater_than(param_idx[key_rcore2], param_idx[key_r_vdw]))
                constraints_desc.append(f"(3) {key_r_vdw} > {key_rcore2}")

            if key_ecore2 in param_idx and key_epsilon in param_idx:
                constraints.append(greater_than(param_idx[key_ecore2], param_idx[key_epsilon]))
                constraints_desc.append(f"(4) {key_ecore2} > {key_epsilon}")

        # -------- 5. ATM eta ≥ 7.2*gamma (QEQ), eta ≥ 8.13*gamma (ACKS2) --------

        factor = 8.13 if "acks2" in self.reaxff_calculator.charge_fix else 7.2

        for e in elements:
            key_gamma = f'ATM.{e}.gamma'
            key_eta = f'ATM.{e}.eta'

            if key_gamma in param_idx and key_eta in param_idx:
                constraints.append(greater_than_or_equal_factor(param_idx[key_eta], param_idx[key_gamma], factor))
                constraints_desc.append(f"(constraint type 5) [{param_idx[key_eta]}]{key_eta} ≥ {factor} * [{param_idx[key_gamma]}]{key_gamma}")

        # -------- 6. ATM bcut_acks2 ≥ max(r_s) --------

        if "acks2" in self.reaxff_calculator.charge_fix:
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
        # chi(F)>chi(O)>chi(Cl)>chi(N)>chi(S)>chi(C)>chi(P)>chi(H)>chi(Mg)>chi(Ca)>chi(Na)>chi(K)

        hierarchy = ['F', 'O', 'Cl', 'N', 'S', 'C', 'P', 'H', 'Mg', 'Ca', 'Na', 'K']
        present = [e for e in hierarchy if f'ATM.{e}.chi' in param_idx]

        for e1, e2 in zip(present[:-1], present[1:]):
            k1 = f'ATM.{e1}.chi'
            k2 = f'ATM.{e2}.chi'
            constraints.append(greater_than(param_idx[k1], param_idx[k2]))
            constraints_desc.append(f"(constraint type 7) [{param_idx[k1]}]{k1} > [{param_idx[k2]}]{k2}")

        # -------- 8. BND De_s ≥ De_p ≥ De_pp --------

        bnd_De = {}
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
                constraints.append(greater_than_or_equal(idxs["De_s"], idxs["De_p"]))
                constraints_desc.append(f"(8a) {base}.De_s ≥ {base}.De_p")
            if "De_p" in idxs and "De_pp" in idxs:
                constraints.append(greater_than_or_equal(idxs["De_p"], idxs["De_pp"]))
                constraints_desc.append(f"(8b) {base}.De_p ≥ {base}.De_pp")
            if "De_p" not in idxs and "De_s" in idxs and "De_pp" in idxs:
                constraints.append(greater_than_or_equal(idxs["De_s"], idxs["De_pp"]))
                constraints_desc.append(f"(8c) {base}.De_s ≥ {base}.De_pp")

        # -------- 9. BND p_be2 ≥ p_bo2, p_bo4, p_bo6 --------

        for pname in param_idx:
            if pname.endswith(".p_be2"):
                base = ".".join(pname.split(".")[:3])
                key_be2 = f"{base}.p_be2"
                for key_bo in [f"{base}.p_bo2", f"{base}.p_bo4", f"{base}.p_bo6"]:
                    if key_bo in param_idx:
                        constraints.append(greater_than_or_equal(param_idx[key_be2], param_idx[key_bo]))
                        constraints_desc.append(f"(9) {key_be2} ≥ {key_bo}")

        # -------- 10. BND p_bo6 ≥ p_bo4 ≥ p_bo2 --------

        for pname in param_idx:
            if any(pname.endswith(s) for s in [".p_bo2", ".p_bo4", ".p_bo6"]):
                base = ".".join(pname.split(".")[:3])
                key_bo2 = f"{base}.p_bo2"
                key_bo4 = f"{base}.p_bo4"
                key_bo6 = f"{base}.p_bo6"
                if key_bo4 in param_idx and key_bo2 in param_idx:
                    constraints.append(greater_than_or_equal(param_idx[key_bo4], param_idx[key_bo2]))
                    constraints_desc.append(f"(10a) {key_bo4} ≥ {key_bo2}")
                if key_bo6 in param_idx and key_bo4 in param_idx:
                    constraints.append(greater_than_or_equal(param_idx[key_bo6], param_idx[key_bo4]))
                    constraints_desc.append(f"(10b) {key_bo6} ≥ {key_bo4}")
                if key_bo4 not in param_idx and key_bo6 in param_idx and key_bo2 in param_idx:
                    constraints.append(greater_than_or_equal(param_idx[key_bo6], param_idx[key_bo2]))
                    constraints_desc.append(f"(10c) {key_bo6} ≥ {key_bo2}")

        # -------- 11. OFD r_pp ≥ r_p ≥ r_s --------

        ofd_keys = {}
        for pname in param_idx:
            if pname.startswith("OFD."):
                base = ".".join(pname.split('.')[:3])
                suffix = pname.split('.')[-1]
                if base not in ofd_keys:
                    ofd_keys[base] = {}
                if suffix in ("r_s", "r_p", "r_pp"):
                    ofd_keys[base][suffix] = param_idx[pname]

        for base, idxs in ofd_keys.items():
            if "r_pp" in idxs and "r_p" in idxs:
                constraints.append(greater_than_or_equal(idxs["r_pp"], idxs["r_p"]))
                constraints_desc.append(f"(11a) {base}.r_pp ≥ {base}.r_p")
            if "r_p" in idxs and "r_s" in idxs:
                constraints.append(greater_than_or_equal(idxs["r_p"], idxs["r_s"]))
                constraints_desc.append(f"(11b) {base}.r_p ≥ {base}.r_s")
            if "r_p" not in idxs and "r_pp" in idxs and "r_s" in idxs:
                constraints.append(greater_than_or_equal(idxs["r_pp"], idxs["r_s"]))
                constraints_desc.append(f"(11c) {base}.r_pp ≥ {base}.r_s")

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
                        constraints.append(ofd_alpha_ge_max_atm(ofd_idx, atm1_idx, atm2_idx))
                        constraints_desc.append(f"(12) {pname} ≥ max({atm1_key}, {atm2_key})")

        # -------- 13. ANG p_pen1 ≥ |p_val1| --------
        # -------- 14. ANG p_pen1 ≥ |p_val2| --------

        for pname in param_idx:
            if pname.startswith("ANG.") and pname.endswith(".p_pen1"):
                base = ".".join(pname.split(".")[:-1])
                k_pen = pname
                k_val1 = f"{base}.p_val1"
                k_val2 = f"{base}.p_val2"
                if k_val1 in param_idx:
                    constraints.append(greater_than_or_equal_abs(param_idx[k_pen], param_idx[k_val1]))
                    constraints_desc.append(f"(13) {k_pen} ≥ |{k_val1}|")
                if k_val2 in param_idx:
                    constraints.append(greater_than_or_equal_abs(param_idx[k_pen], param_idx[k_val2]))
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
                    constraints.append(greater_than_or_equal_abs(param_idx[key_v1], param_idx[key_v2]))
                    constraints_desc.append(f"(15) {key_v1} ≥ |{key_v2}|")
                if key_v3 in param_idx:
                    constraints.append(greater_than_or_equal_abs(param_idx[key_v1], param_idx[key_v3]))
                    constraints_desc.append(f"(16) {key_v1} ≥ |{key_v3}|")

        # -------- 17. HBD r0_hb > r_s (ensure H-bond length exceeds σ-bond radius) --------

        for pname in param_idx:
            if pname.startswith("HBD.") and pname.endswith(".r0_hb"):
                parts = pname.split('.')
                if len(parts) >= 4:
                    donor = parts[1]
                    key_rs = f'ATM.{donor}.r_s'
                    if key_rs in param_idx:
                        constraints.append(greater_than(param_idx[pname], param_idx[key_rs]))
                        constraints_desc.append(f"(constraint type 17) [{param_idx[pname]}]{pname} > [{param_idx[key_rs]}]{key_rs}")

        # -------- PRINT CONSTRAINTS --------
        if self.pt._rank == 0:
            print("-----------------------------------------------------------------------------")
            print(f"PARAMETER CONSTRAINTS APPLIED: {len(constraints)} total")
            for d in constraints_desc:
                print(d)
            print("-----------------------------------------------------------------------------")

        return constraints

    # --------------------------------------------------------------------------------------------

    def error_analysis(self):
        import os
        import numpy as np
        import pandas as pd

        def compute_metrics(y_ref, y_pred):
            error = y_pred - y_ref
            mae = np.mean(np.abs(error))
            rmse = np.sqrt(np.mean(error ** 2))
            ss_res = np.sum(error ** 2)
            ss_tot = np.sum((y_ref - np.mean(y_ref)) ** 2)
            rsq = 1 - ss_res / ss_tot if ss_tot != 0 else float('nan')
            return mae, rmse, rsq

        group_names = self.reaxff_calculator.group_names
        d = self.reaxff_calculator.fitsnap_dict
        output_dir = self.output.path

        rows = []
        for group in group_names:
            mask = d['group'] == group
            ncount = np.sum(mask)

            for prop in ['energy', 'force', 'charge', 'dipole', 'quadrupole']:
                ref_key = f'ref_{prop}'
                pred_key = f'pred_{prop}'
                if ref_key not in d or pred_key not in d:
                    continue
                y_ref = d[ref_key][mask]
                y_pred = d[pred_key][mask]

                if y_ref.shape != y_pred.shape or y_ref.size == 0:
                    continue

                mae, rmse, rsq = compute_metrics(y_ref, y_pred)
                rows.append(((group, prop), ncount, mae, rmse, rsq))

        if self.pt._rank == 0:
            df = pd.DataFrame(rows, columns=['(group, property)', 'ncount', 'mae', 'rmse', 'rsq'])
            df['(group, property)'] = df['(group, property)'].apply(lambda x: f"{x[0]}, {x[1]}")
            md_lines = ['| (group, property) | ncount | mae | rmse | rsq |',
                        '|--------------------|--------|------|------|-----|']
            for _, row in df.iterrows():
                md_lines.append(f"| {row['(group, property)']} | {row.ncount} | {row.mae:.6f} | {row.rmse:.6f} | {row.rsq:.6f} |")
            with open(os.path.join(output_dir, "metrics.md"), "w") as f:
                f.write("\n".join(md_lines))

    # --------------------------------------------------------------------------------------------

    def perform_fit_old(self, fs):
        self.reaxff_calculator = fs.calculator
        if self.pt._size > 1 and self.pt._rank != 0:
            while self._loss_function(None): pass
            return

        self.output = fs.output
        self.initial_x = np.array(self.reaxff_io.values)
        bounds = self.config.sections['REAXFF'].parameter_bounds
        lb, ub = map(np.array, zip(*bounds))
        self._iteration = 1
        self.io_executor = ThreadPoolExecutor(max_workers=1)

        import logging
        logging.getLogger("nevergrad").setLevel(logging.INFO)

        #param = ng.p.Array(init=self.initial_x, mutable_sigma=False)
        #param.set_mutation(sigma=((ub-lb)/3))
        #param.set_bounds(lb, ub)

        param = ng.p.Dict(**{
            name: ng.p.Scalar(init=val, lower=lo, upper=hi)
            for name, val, (lo, hi) in zip(
                self.reaxff_io.parameter_names,
                self.reaxff_io.values,
                bounds
            )
        })

        optimizer = ng.optimizers.NGOpt(parametrization=param, budget=2**31-1, num_workers=1)
        optimizer.parametrization.random_state = np.random.RandomState(12345)
        #self.hsic_logger = ng.callbacks.HSICLoggerCallback()
        #optimizer.register_callback("tell", self.hsic_logger)
        optimizer.register_callback("tell", self._log)
        self._constraints = self._build_constraints()
        best = optimizer.minimize(
            self._loss_function,
            constraint_violation=self._constraints,
            verbosity=0
        )
        best_x = best._value
        best_loss = best.loss
        self.fit = self.reaxff_io.change_parameters_string(best_x)
        self._log_best(self.initial_x, best_x, best_loss, optimizer.num_ask)
        self.io_executor.shutdown(wait=True)
        self.pt._comm.bcast(None, root=0)
