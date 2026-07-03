#!/usr/bin/env python3
"""
General class for running and caching numerical ORM Jacobian calculations.

Each unique combination of parameters is stored as a numbered case under
outputs/<lattice_name>/. An index.json maps case numbers to their parameters,
so re-running with the same parameters loads the cached result instead of
recomputing.
"""

import json
import copy
from pathlib import Path

import numpy as np
import at

from ACalORAT import numerical
from ACalORAT import read

# Resolved once at import time; notebooks must set ROOT explicitly via
# NumericalCalculation.ROOT before instantiating, or pass root= to __init__.
_DEFAULT_ROOT = Path(__file__).resolve().parent.parent


class NumericalCalculation:
    """
    Manages numerical computation and caching of ORM Jacobians.

    Parameters
    ----------
    lattice_name : str
        Name of the lattice file (without extension), e.g. "ring_a2".
        Results are stored under outputs/<lattice_name>/.
    params : dict
        Calculation parameters. Any key not provided falls back to the
        defaults defined in DEFAULT_PARAMS.
    root : Path, optional
        Project root directory. Required when instantiating from a notebook.
    """

    DEFAULT_PARAMS = {
        "step"        : 1e-6,   # Finite difference step
        "direction"   : "v",    # "h", "v", or "both"
        "linearize"   : -1,     # Max PolynomB index kept (-1 = no change)
        "fringes"     : True,   # Keep entrance/exit angles
        "feedback"    : "none", # "none", "Cor_SVD", "Full_SVD"
        "elements"    : "CFD",  # Elements to vary: "CFD", "quad", "dip"
        "n_elements"  : None,   # How many elements to use (None = all)
        "dispersion"  : True,   # Include dispersion in horizontal ORM
    }

    def __init__(self, lattice_name: str, params: dict, root: Path = None):
        self.lattice_name = lattice_name
        self.params = {**self.DEFAULT_PARAMS, **params}
        self.root = root or _DEFAULT_ROOT

        self.output_dir = self.root / "outputs" / lattice_name
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.index_path = self.output_dir / "index.json"
        self._index = self._load_index()

        self.case_id = self._find_or_register_case()
        self.case_dir = self.output_dir / str(self.case_id)
        self.case_dir.mkdir(exist_ok=True)

        self.ring = None
        self.ind  = None
        self.results = {}

    # ------------------------------------------------------------------
    # Index management
    # ------------------------------------------------------------------

    def _load_index(self) -> dict:
        if self.index_path.exists():
            with open(self.index_path) as f:
                return json.load(f)
        return {}

    def _save_index(self):
        with open(self.index_path, "w") as f:
            json.dump(self._index, f, indent=2)

    def _find_or_register_case(self) -> int:
        """Return existing case ID if params match, else register a new one."""
        for case_id, stored_params in self._index.items():
            if stored_params == self.params:
                print(f"Found existing case {case_id} matching parameters.")
                return int(case_id)
        new_id = max((int(k) for k in self._index), default=-1) + 1
        self._index[str(new_id)] = self.params
        self._save_index()
        print(f"Registered new case {new_id}.")
        return new_id

    # ------------------------------------------------------------------
    # Lattice preparation
    # ------------------------------------------------------------------

    def prepare_lattice(self):
        """Load the lattice and apply all parameter-driven modifications."""
        lattice_path = self.root / "data" / f"{self.lattice_name}.mat"
        reader = read.ALBA if self.lattice_name == "THERING" else read.ALBAII
        self.ring, self.ind = reader(lattice_path)

        if not self.params["fringes"]:
            for i in self.ind["dip"]:
                self.ring[i].EntranceAngle = 0.0
                self.ring[i].ExitAngle = 0.0

        if self.params["linearize"] >= 0:
            cutoff = self.params["linearize"]
            for el in filter(at.checkattr("PolynomB"), self.ring):
                for i in range(cutoff + 1, len(el.PolynomB)):
                    el.PolynomB[i] = 0.0

        for i in self.ind["cor"]["h"]: self.ring[i].KickAngle = np.array([0.0, 0.0])
        for i in self.ind["cor"]["v"]: self.ring[i].KickAngle = np.array([0.0, 0.0])

        if not self.params["dispersion"]:
            self.ring.disable_6d()

    # ------------------------------------------------------------------
    # Numerical computation
    # ------------------------------------------------------------------

    def _array_path(self, name: str) -> Path:
        return self.case_dir / f"{name}.npy"

    def _is_cached(self, *names) -> bool:
        return all(self._array_path(n).exists() for n in names)

    def compute(self, force: bool = False):
        """
        Run the numerical calculation for the configured case.

        If results are already cached on disk, they are loaded instead of
        recomputed unless force=True.
        """
        if self.ring is None:
            self.prepare_lattice()

        direction = self.params["direction"]
        elements  = self.params["elements"]
        feedback  = self.params["feedback"]
        ind_el    = self._get_element_indices(elements)

        if feedback != "none":
            cache_keys = ["numerical_h", "numerical_v", "energy", "dfreq",
                          "dkicks_h", "dkicks_v", "x_sex", "dx_sex", "ddisp"]
        elif direction == "both":
            cache_keys = ["numerical_h", "numerical_v"]
            if elements == "dip":
                cache_keys += ["energy", "di_dk"]
        else:
            cache_keys = [f"numerical_{direction}"]
            if elements == "dip":
                cache_keys += ["energy", "di_dk"]

        if self._is_cached(*cache_keys) and not force:
            print(f"Loading cached results for case {self.case_id}.")
            self._load_results(cache_keys)
            return

        print(f"Computing numerical Jacobian for case {self.case_id}...")

        if feedback == "none":
            dirs = ["h", "v"] if direction == "both" else [direction]
            for d in dirs:
                if elements == "dip":
                    num, energy, dispersion = numerical.dORM_dbend(
                        self.ring, self.ind["bpm"], self.ind["cor"][d],
                        ind_el, self.params["step"], d
                    )
                    self.results["energy"] = energy
                    np.save(self._array_path("energy"), energy)
                    if "di_dk" not in self.results:
                        self.results["di_dk"] = dispersion
                        np.save(self._array_path("di_dk"), dispersion)
                else:
                    num = numerical.dORM_dq(
                        self.ring, self.ind["bpm"], self.ind["cor"][d],
                        ind_el, self.params["step"], d
                    )
                self.results[f"numerical_{d}"] = num
                np.save(self._array_path(f"numerical_{d}"), num)

        else:
            # dORM_dCFD computes both directions and handles the feedback loop
            results_cfd = numerical.dORM_dCFD(
                self.ring, self.ind, self.params["step"],
                num=self.params["n_elements"], method=feedback
            )
            num_h, num_v, dfreq, dkicks_h, dkicks_v, x_sex, dx_sex, energy, ddisp = results_cfd

            self.results.update({
                "numerical_h" : num_h,
                "numerical_v" : num_v,
                "dfreq"       : dfreq,
                "dkicks_h"    : dkicks_h,
                "dkicks_v"    : dkicks_v,
                "x_sex"       : x_sex,
                "dx_sex"      : dx_sex,
                "energy"      : energy,
                "ddisp"       : ddisp,
            })
            for name, arr in self.results.items():
                np.save(self._array_path(name), arr)

        print(f"Done. Results saved to case {self.case_id}/")

    def _get_element_indices(self, elements: str) -> list:
        key_map = {"CFD": "CFD", "quad": "quad", "dip": "dip"}
        ind_all = self.ind[key_map[elements]]
        n = self.params["n_elements"]
        return ind_all[:n] if n is not None else ind_all

    def _save_results(self, *names):
        for name in names:
            np.save(self._array_path(name), self.results[name])
        print(f"Saved: {', '.join(names)} → case {self.case_id}/")

    def _load_results(self, names: list):
        for name in names:
            self.results[name] = np.load(self._array_path(name))

    def numerical(self, direction: str) -> np.ndarray:
        """Shortcut to access a numerical result by direction after compute()."""
        key = f"numerical_{direction}"
        if key not in self.results:
            raise RuntimeError(f"No result '{key}'. Run compute() first.")
        return self.results[key]
