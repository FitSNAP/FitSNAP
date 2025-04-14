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
        if self.pt._rank == 0:
            self._hsic_data.append([self._iteration, *x, *total.tolist()])
            return total[-1]
        return True

    # --------------------------------------------------------------------------------------------

    def perform_fit(self, fs):
        self.reaxff_calculator = fs.calculator
        if self.pt._rank != 0:
            while self._loss_function(None): pass
            return

        self.output = fs.output
        self.initial_x = np.array(self.reaxff_io.values)
        bounds = self.config.sections['REAXFF'].parameter_bounds
        lb, ub = map(np.array, zip(*bounds))
        self._iteration = 1
        self.io_executor = ThreadPoolExecutor(max_workers=1)
        self.constraints = self._build_constraints()

        def feasible(x): return all(c(x) >= 0 for c in self.constraints)

        def project(x):
            x = np.clip(x, lb, ub)
            if feasible(x): return x
            xf = lb + np.random.rand(*x.shape) * (ub - lb)
            while not feasible(xf):
                xf = lb + np.random.rand(*x.shape) * (ub - lb)
            lo, hi = 0.0, 1.0
            for _ in range(50):
                mid = (lo + hi) / 2.0
                xmid = np.clip(mid * x + (1 - mid) * xf, lb, ub)
                if feasible(xmid): hi = mid
                else: lo = mid
                if hi - lo < 1e-8: break
            return np.clip(mid * x + (1 - mid) * xf, lb, ub)

        def safe_loss(x): return self._loss_function(project(np.array(x)))

        import logging
        logging.getLogger("nevergrad").setLevel(logging.INFO)

        #param = ng.p.Array(init=self.initial_x, mutable_sigma=True)
        #param.set_mutation(sigma=((ub-lb)/3).astype(float))
        #param.set_bounds(lb, ub)

        #param = ng.p.Array(init=self.initial_x, mutable_sigma=True).set_bounds(lb, ub)
        #sigma = (ub - lb) / 3
        #param.sigma.value = sigma.astype(float)  # must be 1D float array

        # Create Array with fixed sigma first
        param = ng.p.Array(init=self.initial_x).set_bounds(lb, ub)
        # Overwrite the sigma with a vector-valued parameter
        sigma = (ub - lb) / 3
        param.parameters["sigma"] = ng.p.Array(init=sigma, mutable_sigma=False)

        opt = ng.optimizers.NGOpt(parametrization=param, budget=1000000, num_workers=1)
        opt.register_callback("tell", self._log)
        rec = opt.minimize(self._loss_function, verbosity=0)
        best = project(rec.value)
        best_x = best._value
        best_loss = best.loss
        self.fit = self.reaxff_io.change_parameters_string(best_x)
        self._log_best(self.initial_x, best_x, best_loss, opt.num_ask)
        self.io_executor.shutdown(wait=True)
        self.pt._comm.bcast(None, root=0)

    # --------------------------------------------------------------------------------------------

    def _log(self, opt, candidate, loss):

        self._iteration = opt.num_ask + 1
        now = time.time()

        from pprint import pprint
        #print(f"*** num_ask {opt.num_ask} num_tell {opt.num_tell}")
        #print(f"*** {type(opt.recommend())}")
        #pprint(vars(opt.recommend()))
        #print(f"***")

        if opt.num_tell==1:
            best = opt.recommend()
            self._log_best(self.initial_x, best._value, best.loss, opt.num_ask)

        if (now - self._last_log_time) >= 1*60:
            best = opt.recommend()
            best_x = best._value.copy()
            best_loss = best.loss
            hsic_data_copy = self._hsic_data
            self._hsic_data = []
            self._last_log_time = now

            def do_logging():
                self._log_best(self.initial_x, best_x, best_loss, opt.num_ask)
                current_fit = self.reaxff_io.change_parameters_string(best_x)
                df = pd.DataFrame(hsic_data_copy, columns=self._hsic_header)
                self.output.output(current_fit, df)

            self.io_executor.submit(do_logging)

    # --------------------------------------------------------------------------------------------

    def _log_best(self, x0, x_best, f_best, iteration):
        print(f"------------------------- {iteration:<7} {f_best:11.9g} -------------------------")
        print(self.config.sections['CALCULATOR'].charge_fix)
        print("PARAMETER_NAME           INITIAL  LOWER_BOUND        NOW    UPPER_BOUND")
        for p, x0i, xbi in zip(self.reaxff_io.parameter_names, x0, x_best):
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
            return lambda x: x[lhs_idx] - x[rhs_idx] - delta

        def greater_than_or_equal(lhs_idx, rhs_idx):
            return lambda x: x[lhs_idx] - x[rhs_idx]

        def greater_than_or_equal_factor(lhs_idx, rhs_idx, factor):
            return lambda x: x[lhs_idx] - factor * x[rhs_idx]

        def greater_than_or_equal_abs(lhs_idx, rhs_idx):
            return lambda x: x[lhs_idx] - abs(x[rhs_idx])

        def bcut_acks2_ge_max_rs(i_bcut, rs_indices):
            return lambda x: x[i_bcut] - max(x[i] for i in rs_indices)

        def ofd_alpha_ge_max_atm(ofd_idx, atm1_idx, atm2_idx):
            return lambda x: x[ofd_idx] - max(x[atm1_idx], x[atm2_idx])

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
                constraints_desc.append(f"(5) {key_eta} ≥ {factor} * {key_gamma}")

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

        hierarchy = ['F', 'O', 'Cl', 'N', 'S', 'C', 'P', 'H', 'Mg', 'Ca', 'Na', 'K']
        present = [e for e in hierarchy if f'ATM.{e}.chi' in param_idx]

        for e1, e2 in zip(present[:-1], present[1:]):
            k1 = f'ATM.{e1}.chi'
            k2 = f'ATM.{e2}.chi'
            constraints.append(greater_than(param_idx[k1], param_idx[k2]))
            constraints_desc.append(f"(7) {k1} > {k2}")

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
                        constraints_desc.append(f"(17) {pname} > {key_rs}")

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
