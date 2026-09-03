# CONVERSION OF THE TEPLANT OBJECT IN TORCH, ALLOWING BATCHED (OPEN-LOOP) TRAJECTORIES INTEGRATION
# RUN THE SCRIPT TO GET COMPARISON WITH ORIGINAL (NUMPY) TRAJECTORIES AND OUTPUTS

from math import sqrt
from typing import Optional

import numpy as np
from TEPlant import CV, LDEN, RGAS, TAU_VLV, TGAS, VL_MAX, VT, TEParams
from torch import Tensor, cat, float64, stack, tensor, zeros, zeros_like

DEFAULT_DTYPE = float64
EMPTY_TENSOR = tensor([])

# Constant values
CV = tensor(CV, dtype=DEFAULT_DTYPE)


class TorchTEPlant:
    """
    PyTorch version of the TEP (Tennessee Eastman) two-phase reactor plant model.

    This class converts the original numpy-based TEPlant to PyTorch, enabling:
    - Batched trajectory integration (processing multiple scenarios simultaneously)
    - GPU acceleration for faster computation
    - Differentiability for optimization and control applications

    The plant models:
    - Two competing first-order reactions in an isothermal vapor-liquid reactor
    - Vapor-liquid equilibrium
    - Valve dynamics
    - Measurement delays for composition sensing
    """

    def __init__(self, params: Optional[TEParams] = None):
        """
        Initialize the TorchTEPlant with optional parameters.

        Args:
            params: TEParams object containing plant parameters. If None, uses defaults.
        """
        self.params = params or TEParams()
        self.reset_memory()

    def reset_memory(self, t0: float = 0.0, x: Tensor = None) -> None:
        """
        Reset internal state for measurement delay tracking.

        Args:
            t0: Initial time
            x: Initial state tensor (batch_size x 8). If None, initializes to zeros.
        """
        # Initialize time for next purge sample (accounts for measurement delay)
        self.tpurge = tensor([t0 + self.params.tdelay - 1.0e-4])
        if x is None:
            # last measured composition in the purge gas
            self.ylast = zeros((1, 3), dtype=DEFAULT_DTYPE)
            # current measured composition delayed by one sample period
            self.ymeas = zeros((1, 3), dtype=DEFAULT_DTYPE)
        else:
            _, y3 = self._flows_pressures_composition(x)
            self.ylast = y3.clone()
            self.ymeas = y3.clone()

    @staticmethod
    def default_initial_state() -> Tensor:
        """
        Return the nominal steady-state initial state vector (See Table 1 in Ricker paper).

        Returns:
            Tensor of shape (8,) containing nominal initial state values:
            [vapor_moles_A, vapor_moles_B, vapor_moles_C, liquid_moles D, 
             valve1_pos, valve2_pos, valve3_pos, valve4_pos]
        """
        return tensor([
            4.449999958429348e01,
            1.353296996509594e01,
            3.664788062995841e01,
            1.100000000000000e02,
            6.095327313484253e01,
            2.502232231706676e01,
            3.925777017606444e01,
            4.703024823457651e01,
        ], dtype=DEFAULT_DTYPE)

    def _flows_pressures_composition(self, x: Tensor):
        """
        Compute flow rates, pressures, and composition from state variables.

        Args:
            x: State tensor of shape (batch_size, 8) containing:
               [vapor_A, vapor_B, vapor_C, liquid, valve1, valve2, valve3, valve4]

        Returns:
            Tuple of:
            - (f1, f2, f3, f4, pt, vlpct, vl, vv, nv): flow rates, pressure, volumes and vapor moles
            - y3: composition in vapor phase [y_A, y_B, y_C]
        """
        nl = x[:, 3]                 # Total liquid moles [kmol]
        vl = nl / LDEN               # Liquid volume [m^3]
        vv = VT - vl                 # Vapor volume [m^3]
        nv = x[:, :3].sum(dim=1)   # Total vapor moles [kmol]
        pt = nv * RGAS * TGAS / vv   # Total pressure [kPa]
        vlpct = vl * 100.0 / VL_MAX  # Liquid volume as a percentage of capacity

        # Flow rates (Equations 12 and 13 from Ricker's paper)
        # Feed 1 flow rate
        f1 = CV[0] * x[:, 4]
        # Feed 2 flow rate
        f2 = CV[1] * x[:, 5]

        # Pressure drop across purge valve
        root_term = pt - 100.0
        root_term[root_term < 0.0] = 0.
        root_term = root_term.sqrt()

        # Feed 3 (product) flow rate
        f3 = CV[2] * x[:, 6] * root_term
        # Product (purge) flow rate
        f4 = CV[3] * x[:, 7] * root_term

        # No flow if no liquid
        f4[x[:, 3] <= 0.0] = 0.0

        # Vapor composition (mole fractions)
        y3 = x[:, :3] / nv[:, None]

        return (f1, f2, f3, f4, pt, vlpct, vl, vv, nv), y3

    def _update_measurement_delay(self, t: Tensor, y3: Tensor) -> None:
        """
        Update the delayed composition measurement.

        Implements the sampling delay for gas composition measurements.
        The measurement at time t depends on the composition at time t - tdelay.

        Args:
            t: Current time(s) - tensor of shape (batch_size,)
            y3: Current vapor composition [y_A, y_B, y_C] of shape (batch_size, 3)
        """
        # Handle batch size matching for time-dependent operations
        if t.size(0) > 1:
            # Expand memory if processing multiple trajectories
            if self.ylast.size(0) == 1:
                self.ylast = self.ylast.repeat([t.size(0), 1])
            if self.ymeas.size(0) == 1:
                self.ymeas = self.ymeas.repeat([t.size(0), 1])
            if self.tpurge.size(0) == 1:
                self.tpurge = self.tpurge.repeat([t.size(0)])

        # Handle initial condition at t <= 0
        negative_time_mask = t <= 0.
        self.ylast[negative_time_mask] = y3[negative_time_mask].clone()
        self.ymeas[negative_time_mask] = y3[negative_time_mask].clone()

        # Handle delayed measurements
        # tpurge should already be expanded if t.size(0) > 1

        # Find trajectories where we need to update delayed measurement
        purge_time_mask = (t > self.tpurge).logical_and(
            negative_time_mask.logical_not())

        # Update delayed measurement when sampling time is reached
        while purge_time_mask.any():
            self.tpurge[purge_time_mask] += self.params.tdelay
            # last measure becomes previous measurement
            self.ymeas[purge_time_mask] = self.ylast[purge_time_mask].clone()
            # current measure becomes latest composition
            self.ylast[purge_time_mask] = y3[purge_time_mask].clone()
            # Check for next sampling event
            purge_time_mask = t > self.tpurge

    def outputs(self, t: Tensor, x: Tensor, u: Tensor, update_memory: bool = True) -> Tensor:
        """
        Compute plant outputs for given state and control inputs.
        In practice, in this case u does not directly affects outputs,
        it affects outputs indirectly through the states via the 
        dynamics x_dot = f(x, u).

        Args:
            t: Current time(s)
            x: State tensor of shape (batch_size, 8)
            u: Control input tensor of shape (batch_size, 4) [valve positions]
            update_memory: Whether to update delayed measurements

        Returns:
            Output tensor of shape (batch_size, 10) containing:
            [f1, f2, f3, f4, pt, vlpct, yA%, yB%, yC%, cost_per_mole]
        """
        # Compute flows, pressures, and composition
        (f1, f2, f3, f4, pt, vlpct, *_), y3 = self._flows_pressures_composition(x)

        # Debug assertions
        for _ in (f1, f2, f3, f4, pt, vlpct):
            assert _.ndim == 1
        assert y3.ndim == 2

        # Update delayed measurements if requested
        if update_memory:
            self._update_measurement_delay(t, y3)

        # Assemble output vector
        return stack([
            f1,                          # Feed 1 flow rate
            f2,                          # Feed 2 flow rate
            f3,                          # Feed 3 (product) flow rate
            f4,                          # Product (purge) flow rate
            pt,                          # Total pressure
            vlpct,                       # Liquid volume percentage
            self.ymeas[:, 0] * 100.0,    # A in purge (mole %)
            self.ymeas[:, 1] * 100.0,    # B in purge (mole %)
            self.ymeas[:, 2] * 100.0,    # C in purge (mole %)
            f3 * (self.ymeas[:, 0] * 2.206 + self.ymeas[:, 2]
                  * 6.177) / f4  # cost/mole product
        ], dim=-1)

    def derivatives(self, t: Tensor, x: Tensor, u: Tensor) -> Tensor:
        """
        Compute state derivatives for the reactor plant model.

        Implements the dynamic equations for the two-phase reactor:
        - Material balances for A, B, C and D in vapor and liquid phases
        - Reaction rate kinetics
        - Valve dynamics

        Args:
            t: Current time(s)
            x: State tensor of shape (batch_size, 8)
            u: Control input tensor of shape (batch_size, 4) [valve positions]

        Returns:
            dxdt: State derivative tensor of shape (batch_size, 8)
        """
        # Compute flows, pressures, and composition
        (f1, f2, f3, f4, pt, vlpct, *_), y3 = self._flows_pressures_composition(x)
        self._update_measurement_delay(t, y3)

        # Reactor composition (current, not delayed - dynamics respond to current state)
        ya3, yb3, yc3 = y3[:, 0], y3[:, 1], y3[:, 2]

        # Kinetic parameters with safety checks
        kpar = self.params.kpar if self.params.kpar > 0 else 0.00117
        ncpar = self.params.ncpar if self.params.ncpar > 0 else 0.4

        # Partial pressures
        pa = ya3 * pt
        pc = yc3 * pt

        # Reaction rate (Equation 5 from Ricker's paper)
        # r = k * (p_A)^1.2 * (p_C)^n where n = ncpar
        rr1 = kpar * (pa ** 1.2) * (pc ** ncpar)

        # Material balances (Equations 1-4)
        dxdt = zeros_like(x)
        # Vapor A balance: feed A -> consumption in reaction and product
        dxdt[:, 0] = self.params.ya1 * f1 + f2 - ya3 * f3 - rr1
        # Vapor B balance: feed B -> consumption in product
        dxdt[:, 1] = self.params.yb1 * f1 - yb3 * f3
        # Vapor C balance: feed C -> consumption in reaction and product
        dxdt[:, 2] = self.params.yc1 * f1 - yc3 * f3 - rr1
        # Liquid D balance: reaction produces liquid, product removes liquid
        dxdt[:, 3] = rr1 - f4

        # Valve dynamics (Equations 14-15)
        # Effective valve position includes level controller feedback
        u_eff = u.clone()
        u4 = self.params.u4bar + self.params.kc_vl * (u[:, 3] - vlpct)
        u_eff = cat((u_eff[:, :3], u4[:, None]), dim=-1)

        # Nonlinear valve characteristic
        val = u_eff.clone()
        # For valve position <= 1: exponential rise from 0 to ~2.7
        val[u_eff <= 1.0] = (u_eff[u_eff <= 1.0] - 1.0).exp()
        # For valve position >= 99: exponential approach to 100
        val[u_eff >= 99.0] = 100.0 - ((99.0 - u_eff[u_eff >= 99.0])).exp()
        # Apply maximum limit for Feed 2 valve
        val[val[:, 1] > self.params.u2max, 1] = self.params.u2max

        # First-order valve dynamics
        dxdt[:, 4:8] = (val - x[:, 4:8]) / TAU_VLV

        return dxdt


if __name__ == '__main__':
    # Test configuration
    TEST_STATE = False  # Set to True to test state integration
    TEST_OUTPUT = True  # Set to True to test output computation

    # Import reference implementations for comparison
    import os
    import sys

    from closed_loop_scenarios import run_closed_loop_scenario
    from integrators import integrate_wcontrol
    from plot_utils import plotly_output_comparison, plotly_state_comparison
    from plotly.subplots import make_subplots
    from TEPlant import TEParams, TEPlant
    from torch import cat, from_numpy, stack

    class HiddenPrints:
        """
        Context manager to suppress print output.
        Temporarily redirects sys.stdout to /dev/null.
        """

        def __enter__(self):
            self._original_stdout = sys.stdout
            sys.stdout = open(os.devnull, 'w')

        def __exit__(self, exc_type, exc_val, exc_tb):
            sys.stdout.close()
            sys.stdout = self._original_stdout

    # Generate reference data from numpy implementation
    # Using 'ramp_production' scenario to test tracking performance
    with HiddenPrints():
        TE_RESULTS = run_closed_loop_scenario(scenario='ramp_production')

    # Extract parameters from reference results
    PARAMS = TE_RESULTS['params']

    # Initialize both plant models
    PLANT = TorchTEPlant(params=PARAMS)  # PyTorch version for testing
    NP_PLANT = TEPlant(params=PARAMS)    # NumPy version as reference

    # Load reference trajectory data
    T = from_numpy(TE_RESULTS['t'].T)      # Time points
    X = from_numpy(TE_RESULTS['x'].T)      # State trajectories
    U = from_numpy(TE_RESULTS['u'].T)      # Control inputs
    Y = from_numpy(TE_RESULTS['y'].T)      # Outputs

    # Test 1: Verify derivatives match between numpy and torch implementations
    DERIVS = PLANT.derivatives(T, X, U)

    for IDX in range(T.size(0)):
        # Compute derivatives using numpy reference
        NP_DERIVS = from_numpy(NP_PLANT.derivatives(
            TE_RESULTS['t'].T[IDX],
            TE_RESULTS['x'].T[IDX],
            TE_RESULTS['u'].T[IDX]
        ))

        # Compute maximum absolute error
        DERIV_ERROR = (DERIVS[IDX] - NP_DERIVS).abs().max().item()

        if DERIV_ERROR > 1e-10:
            print(f"Derivative mismatch at index {IDX}: error = {DERIV_ERROR}")

    # Test 2: Verify state integration (optional)
    if TEST_STATE:
        PLANT.reset_memory()

        # Integrate plant dynamics over each time step
        # integrate_wcontrol returns (batch_size, n_times, n_states)
        # We need only the final state [0, -1, :] with shape (1, n_states)
        X1 = integrate_wcontrol(
            PLANT.derivatives,      # Dynamics function
            X[:-1],                 # Initial states for each step
            stack((T[:-1], T[1:]), dim=-1),  # Time intervals
            U[:-1],                 # Control inputs
            rtol=1e-6, atol=1e-9    # Integration tolerances
        )[:, -1, :]  # Extract final state

        # Concatenate initial state with integrated states
        X_out = cat((X[0][None, :], X1), dim=0)

        # Plot state comparison using plot_utils function
        plotly_state_comparison(T, X, X_out).show(renderer="browser")

    # Test 3: Verify output computation (default test)
    if TEST_OUTPUT:
        PLANT.reset_memory()

        # Compute outputs along trajectory
        Y_out = zeros_like(Y)
        Y_out[0] = PLANT.outputs(T[0].reshape(
            [1]), X[0][None, :], U[0][None, :])
        X_t = X[0][None, :]

        for idx in range(T.size(0) - 1):
            # Integrate dynamics to get next state

            X_t = integrate_wcontrol(
                PLANT.derivatives,
                X_t,
                T[idx:idx+2][None, :],
                U[idx][None, :],
                rtol=1e-6, atol=1e-9
            )[:, -1, :]  # Extract final state

            # Compute output at next time point
            Y_out[idx+1] = PLANT.outputs(
                T[idx+1].reshape([1]),
                X_t,
                U[idx][None, :]
            )

        # Plot output comparison using plot_utils function
        plotly_output_comparison(T, Y, Y_out).show(renderer="browser")
