from typing import Literal, Optional

import numpy as np
from scipy.integrate import solve_ivp
from TEPlant import CV, LDEN, RGAS, TGAS, VT, TEParams, TEPlant

# List of scenarios (runtime tuple)
_SCENARIOS = ('feed_A_composition', 'feed_B_composition', 'loss_f2',
              'kinetic_drift', 'ramp_production', 'low_cost_ya3', 'nominal')

# Type alias for scenario type hints
_SCENARIOS_TYPE = Literal['feed_A_composition', 'feed_B_composition', 'loss_f2',
                          'kinetic_drift', 'ramp_production', 'low_cost_ya3', 'nominal']


# Constant values (see Ricker's paper)
TAUIS = np.array([1.0,  1.5, 3.0])  # integral times for PI controllers
KCS = np.array([0.1, -0.25, 2.0])  # controller gains for PI controllers

ICV = [3, 4, 6]  # indices of controlled outputs (F4, Pressure, A in Purge)
IMV = [0, 2, 1]  # indices of manipulated variables (valves)

ERRN1_PC_INIT = 0.0
KC_PC_INIT = 0.7
TAUI_PC_INIT = 3.0
F4SP_ADJ_INIT = 0.0


def steady_states(ya3: float, f4: float, p: float, params: Optional[TEParams] = None):
    # Steady state computation (state derivatives are 0)
    params = params or TEParams()
    pa = p * ya3
    pc = (f4 / (params.kpar * (pa ** 1.2))) ** (1.0 / params.ncpar)
    yc3 = pc / p
    pb = p - pa - pc
    yb3 = pb / p
    # Equations 1-4 with derivatives = 0
    a = np.array([
        [params.ya1, 1.0, -ya3],
        [params.yb1, 0.0, -yb3],
        [params.yc1, 0.0, -yc3],
    ], dtype=float)
    b = np.array([f4, 0.0, f4], dtype=float)
    f = np.linalg.solve(a, b)
    x = np.zeros(8, dtype=float)
    # Equations 12-13
    x[4] = f[0] / CV[0]
    x[5] = f[1] / CV[1]
    x[6] = f[2] / (CV[2] * np.sqrt(p - 100.0))
    x[7] = f4 / (CV[3] * np.sqrt(p - 100.0))
    nd = 110.0
    vl = nd / LDEN
    x[3] = nd
    vv = VT - vl
    nv = p * vv / (RGAS * TGAS)
    x[0] = nv * ya3
    x[1] = nv * yb3
    x[2] = nv * yc3
    return x, vl


def run_closed_loop_scenario(
    tend: float = 30.0,
    dt: float = 0.1,
    integration_step: float = 0.002,
    scenario: _SCENARIOS_TYPE = 'feed_A_composition',
):
    base = TEParams()
    # Get steady-state x0 based on nominal values of
    # base parameters and controlled variables
    x0, vl = steady_states(ya3=0.47, f4=100.0, p=2700.0, params=base)
    vlpct = vl * 100.0 / 30.0  # nominal liquid inventory

    # Get steady-state u0 based on x0 and nominal liquid inventory
    # u(1:3) = x(1:3) since derivatives are 0 and x4=x4bar (nominal)
    # in equations 14-15
    u0 = np.array([x0[4], x0[5], x0[6], vlpct], dtype=float)
    base.u4bar = x0[7]

    # Define disturbance scenarios
    params = base.copy()
    del_kpar = 0.0
    del_ncpar = 0.0
    tdrift = 48.0

    if scenario == 'feed_A_composition':
        params.ya1 = 0.45
    elif scenario == 'feed_B_composition':
        params.yb1 = 0.01
    elif scenario == 'loss_f2':
        params.u2max = 0.0
    elif scenario == 'kinetic_drift':
        del_kpar = -0.0002
        del_ncpar = -0.1
    elif scenario == 'ramp_production':
        pass
    elif scenario == 'low_cost_ya3':
        pass
    elif scenario == 'nominal':
        pass
    else:
        raise ValueError(f'Unknown scenario: {scenario}')

    plant = TEPlant(params)
    plant.reset_memory(t0=0.0, x=x0)
    # Get outputs at the steady-state initial condition
    y0 = plant.outputs(0.0, x0, u0, update_memory=True)
    icv = np.array([3, 4, 6])  # Indices of controlled outputs
    imv = np.array([0, 2, 1])  # Indices of manipulated outputs
    setpts = y0[icv].copy()  # setpoint are initialized to measurements
    if scenario == 'ramp_production':
        setpts[1] = 2850
        setpts[2] = 63

    # Prepare control loops

    xp = x0.copy()
    up = u0.copy()

    errn1 = np.zeros_like(KCS)

    nstep = int(round(tend / dt)) + 1
    times = np.linspace(0., tend, nstep)

    Y = np.zeros([y0.shape[0], nstep])
    Y[:, 0] = y0

    X = np.zeros([x0.shape[0], nstep])
    X[:, 0] = x0

    U = np.zeros([up.shape[0], nstep])

    F4sp = np.zeros(nstep)
    F4sp[0] = setpts[0]

    kpar_hist = np.zeros([nstep])
    kpar_hist[0] = params.kpar

    ncpar_hist = np.zeros([nstep])
    ncpar_hist[0] = params.ncpar

    kc_pc = KC_PC_INIT
    taui_pc = TAUI_PC_INIT
    errn1_pc = ERRN1_PC_INIT
    f4sp_adj = F4SP_ADJ_INIT

    # Number of sampling periods in simulation

    for ii, t in enumerate(times[1:]):
        t = ii * dt
        if scenario == 'kinetic_drift':
            if t < tdrift:
                params.kpar = base.kpar + del_kpar * (t / tdrift)
                params.ncpar = base.ncpar + del_ncpar * (t / tdrift)
            else:
                params.kpar = base.kpar + del_kpar
                params.ncpar = base.ncpar + del_ncpar
        elif scenario == 'ramp_production':
            # Scenario 2: Ramp F4 setpoint from 100 to 130 kmol/h
            # Ramp time from Ricker's paper Figure 8 is approximately 8-10 hours
            ramp_time = 1.0  # hours to complete ramp
            if t < ramp_time:
                ramp_fraction = t / ramp_time
            else:
                ramp_fraction = 1.0
            # Target F4 = 130 kmol/h
            f4_ramp_target = 130.0
            # Update F4 setpoint based on ramp
            setpts[0] = 100.0 + (f4_ramp_target - 100.0) * ramp_fraction
        elif scenario == 'low_cost_ya3':
            # Scenario: Move from nominal ya3=47% to ya3=63% to reduce operating costs
            # Operating costs are less than half those at the base point when all other factors are nominal
            ramp_time = 10.0  # hours to complete ramp
            if t < ramp_time:
                ramp_fraction = t / ramp_time
            else:
                ramp_fraction = 1.0
            # Target ya3 in purge = 63% (from nominal 47%)
            ya3_target = 63.0
            # Update A in Purge setpoint based on ramp
            setpts[2] = 47.0 + (ya3_target - 47.0) * ramp_fraction
        # Get current plant outputs
        yp = plant.outputs(t, xp, up, update_memory=True)
        # Pressure control override loop
        errn_pc = 2900.0 - yp[4]
        f4sp_adj = min(f4sp_adj + kc_pc * (errn_pc -
                       errn1_pc + dt * errn_pc / taui_pc), 0.0)
        errn1_pc = errn_pc
        setpts_save = setpts.copy()
        setpts[0] = setpts[0] + f4sp_adj
        # PI multi-loop control
        errn = setpts - yp[icv]
        delu = KCS * (errn - errn1 + dt * errn / TAUIS)
        up[imv] = up[imv] + delu
        errn1 = errn
        up[:3] = np.minimum(np.maximum(up[:3], 0.0), 100.0)

        # register values for future plots

        Y[:, ii+1] = yp
        U[:, ii] = up
        F4sp[ii+1] = setpts[0]
        kpar_hist[ii+1] = params.kpar
        ncpar_hist[ii+1] = params.ncpar
        setpts = setpts_save

        if ii == nstep - 1:
            break

        # Simulate plant to next sampling time
        def rhs(tt, xx):
            return plant.derivatives(tt, xx, up)
        sol = solve_ivp(rhs, (t, t + dt), xp, method='RK45',
                        max_step=integration_step, rtol=1e-6, atol=1e-9)
        xp = sol.y[:, -1]

        X[:, ii+1] = xp

    U[:, -1] = up

    return {
        't': times,
        'x': X,
        'y': Y,
        'u': U,
        'f4sp': F4sp,
        'kpar': kpar_hist,
        'ncpar': ncpar_hist,
        'scenario': scenario,
        'params': params
    }


if __name__ == '__main__':

    from plot_utils import plot_ricker_style, plot_scenario_comparison

    # Single scenario plot, matching the original MATLAB layout
    result = run_closed_loop_scenario(scenario=_SCENARIOS[5])
    fig, axs = plot_ricker_style(result, save_path=f'{_SCENARIOS[5]}.png')

    # Multi-scenario comparison plot
    scenarios = {
        name: run_closed_loop_scenario(scenario=name)
        for name in _SCENARIOS
    }
    fig, axs = plot_scenario_comparison(
        scenarios, save_path='scenario_comparison.png')

    """
    from torch import cat, from_numpy

    SCENARIOS = ['feed_A_composition', 'feed_B_composition', 'loss_f2',
                 'kinetic_drift', 'ramp_production', 'low_cost_ya3', 'nominal']

    SOL_DICT = {}

    for scen in SCENARIOS:
        SOL_DICT[scen] = run_closed_loop_scenario(scenario=scen)

    SNAPSHOT_SIZE = 2

    X_VALUES = {key: from_numpy(sol['x'].T).reshape(
        [-1, 2, 8]) for key, sol in SOL_DICT.items()}
    SET_SIZES = {key: val.size(0) for key, val in X_VALUES.items()}

    X_TENSOR = cat([x for x in X_VALUES.values()], dim=0).flatten(start_dim=1)

    print(X_TENSOR.size())

    """
