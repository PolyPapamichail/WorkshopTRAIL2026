# Code based on N.L. Ricker TEP two-phase reactor simulator


from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Optional

import numpy as np

# Constant values (see Ricker's paper)
CV = np.array([3.3046, 0.2246, 0.00352, 0.0417], dtype=float)
TAU_VLV = 2.77e-3
VT = 122.0
VL_MAX = 30.0
RGAS = 8.314
TGAS = 373.0
LDEN = 8.3


@dataclass
class TEParams:
    # Init values are nominal values from Table 1 or constants
    tdelay: float = 0.1  # Sampling delay for gas composition
    ya1: float = 0.485  # Feed 1 mole fraction of A
    yb1: float = 0.005  # Feed 1 mole fraction of B
    # Maximum possible valve position, Feed 2 (Useful for Scenario III)
    u2max: float = 100.0
    kc_vl: float = -1.4  # Level controller gain (%/%)
    u4bar: float = 47.03024823457651  # Nominal steady-state for product valve
    kpar: float = 0.00117  # Pre-exponential (see Equation 5)
    ncpar: float = 0.4  # Exponent on Pc (see Equation 5)
    memory: np.ndarray = field(
        # Variable used to store intermediate results
        default_factory=lambda: np.zeros(41, dtype=float))

    @property
    def yc1(self) -> float:
        # Mole fraction C in feed 1
        return 1.0 - self.ya1 - self.yb1

    def copy(self) -> 'TEParams':
        return TEParams(
            tdelay=self.tdelay,
            ya1=self.ya1,
            yb1=self.yb1,
            u2max=self.u2max,
            kc_vl=self.kc_vl,
            u4bar=self.u4bar,
            kpar=self.kpar,
            ncpar=self.ncpar,
            memory=self.memory.copy(),
        )


class TEPlant:
    def __init__(self, params: Optional[TEParams] = None):
        self.params = params or TEParams()
        self.reset_memory()

    def reset_memory(self, t0: float = 0.0, x: Optional[Iterable[float]] = None) -> None:
        # Initialize certain parameters that change from one call to the next
        # Time of next purge sample (delayed measurement)
        self.tpurge = t0 + self.params.tdelay - 1.0e-4
        if x is None:
            # last measured composition in the purge gas
            self.ylast = np.zeros(3, dtype=float)
            # current measured composition delayed by one sample period
            self.ymeas = np.zeros(3, dtype=float)
        else:
            x = np.asarray(x, dtype=float)
            _, y3 = self._flows_pressures_composition(x)
            self.ylast = y3.copy()
            self.ymeas = y3.copy()

    @staticmethod
    def default_initial_state() -> np.ndarray:
        # see state variables nominal values of Table 1
        return np.array([
            4.449999958429348e01,
            1.353296996509594e01,
            3.664788062995841e01,
            1.100000000000000e02,
            6.095327313484253e01,
            2.502232231706676e01,
            3.925777017606444e01,
            4.703024823457651e01,
        ], dtype=float)

    def _flows_pressures_composition(self, x: np.ndarray):
        nl = x[3]  # Total liquid moles [kmol]
        vl = nl / LDEN  # Liquid volume [m^3]
        vv = VT - vl  # Vapor volume [m^3]
        nv = x[0] + x[1] + x[2]  # Total vapor moles [kmol]
        pt = nv * RGAS * TGAS / vv  # Total pressure [kPa]
        vlpct = vl * 100.0 / VL_MAX  # Liquid volume as a percentage of capacity

        # Flow rates (Equations 12 and 13)
        f1 = CV[0] * x[4]
        f2 = CV[1] * x[5]
        root_term = np.sqrt(max(pt - 100.0, 0.0))
        f3 = CV[2] * x[6] * root_term
        f4 = CV[3] * x[7] * root_term
        if x[3] <= 0.0:
            f4 = 0.0
        y3 = np.array([x[0], x[1], x[2]], dtype=float) / nv
        return (f1, f2, f3, f4, pt, vlpct, vl, vv, nv), y3

    def _update_measurement_delay(self, t: float, y3: np.ndarray) -> None:

        if t <= 0.0 and not np.any(self.ymeas):
            self.ylast = y3.copy()
            self.ymeas = y3.copy()
            return

        while t > self.tpurge:
            self.tpurge += self.params.tdelay
            # last measure
            self.ymeas = self.ylast.copy()
            # current measure
            self.ylast = y3.copy()

    def outputs(self, t: float, x: Iterable[float], u: Iterable[float], update_memory: bool = True) -> np.ndarray:
        # compute output variables according current state variables
        x = np.asarray(x, dtype=float)
        (f1, f2, f3, f4, pt, vlpct, *_), y3 = self._flows_pressures_composition(x)

        if update_memory:
            # purge concentrations are obtained with a delay
            self._update_measurement_delay(t, y3)
        return np.array([
            f1,
            f2,
            f3,
            f4,
            pt,
            vlpct,
            self.ymeas[0] * 100.0,
            self.ymeas[1] * 100.0,
            self.ymeas[2] * 100.0,
            f3 * (self.ymeas[0] * 2.206 + self.ymeas[2]
                  * 6.177) / f4,  # cost/mole product
        ], dtype=float)

    def derivatives(self, t: float, x: Iterable[float], u: Iterable[float]) -> np.ndarray:
        # Express state derivative according current state and u applied
        x = np.asarray(x, dtype=float)
        u = np.asarray(u, dtype=float)
        (f1, f2, f3, f4, pt, vlpct, *_), y3 = self._flows_pressures_composition(x)
        self._update_measurement_delay(t, y3)
        # can use y3 here because the delay is in the measurement not in the actual process dynamics,
        # the reactor responds to the current composition
        ya3, yb3, yc3 = y3
        kpar = self.params.kpar if self.params.kpar > 0 else 0.00117
        ncpar = self.params.ncpar if self.params.ncpar > 0 else 0.4
        pa = ya3 * pt
        pc = yc3 * pt
        rr1 = kpar * (pa ** 1.2) * (pc ** ncpar)
        # See Equations 1-4
        dxdt = np.zeros(8, dtype=float)
        dxdt[0] = self.params.ya1 * f1 + f2 - ya3 * f3 - rr1
        dxdt[1] = self.params.yb1 * f1 - yb3 * f3
        dxdt[2] = self.params.yc1 * f1 - yc3 * f3 - rr1
        dxdt[3] = rr1 - f4
        # See Equations 14-15
        u_eff = np.asarray(u, dtype=float).copy()
        u4 = self.params.u4bar + self.params.kc_vl * (u[3] - vlpct)
        u_eff = np.append(u_eff[:3], u4)
        for i in range(4):
            val = u_eff[i]
            if val <= 1.0:
                val = np.exp(val - 1.0)
            elif val >= 99.0:
                val = 100.0 - np.exp(99.0 - val)
            if i == 1:
                val = min(val, self.params.u2max)
            dxdt[i + 4] = (val - x[i + 4]) / TAU_VLV

        return dxdt
