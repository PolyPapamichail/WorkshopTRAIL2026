# FUNCTIONS RELATIVE TO RUNNING BATCHED CLOSED-LOOP TRAJECTORIES USING TORCH
# AS OF NOW, THESE ARE NOT FUNCTIONNAL, IT IS STILL A DRAFT

class Scenario(ABC):
    def __init__(self):
        super().__init__()
        self.name = None

    def initial_params_update(self, params: TEParams) -> None:
        pass

    def initial_setpts_update(self, setpts: Tensor) -> None:
        pass

    def __call__(self, t: Tensor, x: Tensor, params: TEParams, setpts: Tensor) -> None:
        pass


class FeedComposition(Scenario):
    def __init__(self, ya1: float = None, yb1: float = None):
        super().__init__()
        self.name = 'feed_composition'
        self.ya1, self.yb1 = ya1, yb1

    def initial_params_update(self, params):
        if self.ya1 is not None:
            params.ya1 = self.ya1
        if self.yb1 is not None:
            params.yb1 = self.yb1


class LossF2(Scenario):
    def __init__(self):
        super().__init__()
        self.name = 'loss_f2'

    def initial_params_update(self, params):
        params.u2max = 0.


class RampProduction(Scenario):
    def __init__(self, ramp_time: float = 1., f4_ramp_target: float = 130.):
        super().__init__()
        self.name = 'ramp_production'
        self.ramp_time = ramp_time
        self.f4_ramp_target = f4_ramp_target

    def initial_setpts_update(self, setpts):
        setpts[:, 1] = 2850
        setpts[:, 2] = 63

    def __call__(self, t, x, params, setpts):
        ramp_fraction = ones_like(t)
        ramp_fraction[t < self.ramp_time] = t[t <
                                              self.ramp_time] / self.ramp_time

        setpts[:, 0] = 100. + (self.f4_ramp_target - 100.) * ramp_fraction


class LowCostYa3(Scenario):
    def __init__(self, ramp_time: float = 10., ya3_target: float = 63.):
        super().__init__()
        self.name = 'low_cost_ya3'
        self.ramp_time = ramp_time
        self.ya3_target = ya3_target

    def __call__(self, t, x, params, setpts):
        ramp_fraction = ones_like(t)
        ramp_fraction[t < self.ramp_time] = t[t <
                                              self.ramp_time] / self.ramp_time

        setpts[:, 2] = 47.0 + (self.ya3_target - 47.0) * ramp_fraction


def batched_te4_ss(ya3: Tensor, f4: Tensor, p: float, params: Optional[TEParams] = None):
    # Steady state computation (state derivatives are 0)
    assert ya3.size() == f4.size()
    params = params or TEParams()
    pa = p * ya3
    pc = (f4 / (params.kpar * (pa ** 1.2))) ** (1.0 / params.ncpar)
    yc3 = pc / p
    pb = p - pa - pc
    yb3 = pb / p

    # Equations 1-4 with derivatives = 0
    a = tensor([
        [params.ya1, 1.0, 0.],
        [params.yb1, 0.0, 0.],
        [params.yc1, 0.0, 0.]])[None, :, :].repeat(ya3.size(0), 1, 1)
    a[:, 0, 2] = -ya3
    a[:, 1, 2] = -yb3
    a[:, 2, 2] = -yc3

    b = stack([f4, zeros_like(f4), f4])
    f = solve(a, b)
    x = zeros((ya3.size(0), 8))

    nd = 110.0
    vl = nd / LDEN
    vv = VT - vl
    nv = p * vv / (RGAS * TGAS)

    x[:, 0] = nv * ya3
    x[:, 1] = nv * yb3
    x[:, 2] = nv * yc3

    x[:, 3] = nd
    # Equations 12-13
    x[:, 4] = f[:, 0] / CV[0]
    x[:, 5] = f[:, 1] / CV[1]
    x[:, 6] = f[:, 2] / (CV[2] * sqrt(p - 100.0))
    x[:, 7] = f4 / (CV[3] * sqrt(p - 100.0))
    return x, vl


def torch_run_scenario(fun: Callable, tend: float = 30.,
                       dt: float = .1, integration_step: float = .002,
                       scenario: Scenario = FeedComposition()):

    base = TEParams()
    x0, vl = batched_te4_ss(ya3=.47, f4=100., p=2700., params=base)
    vlpct = vl * 100. / 30.

    u0 = cat((x0[4:7], full_like(x0[:, 0][:, None], vlpct)), dim=-1)
    base.u4bar = x0[:, 7]

    params = base.copy()
    del_kpar = 0.
    del_ncpar = 0.
    tdrift = 48.

    scenario.initial_params_update(params)

    plant = TorchTEPlant(params)
    plant.reset_memory(t0=0., x=x0)
    y0 = plant.outputs(0., x0, u0, update_memory=True)
    setpts = y0[:, ICV].clone()

    scenario.initial_setpts_update(setpts)

    errn1 = zeros_like(KCS)

    nstep = int(round(tend / dt)) + 1

    xp = x0.clone()
    up = u0.clone()

    times = linspace(0., tend, nstep)

    Y = zeros(y0.shape[0, 1] + [nstep])
    Y[..., 0] = y0

    X = zeros(x0.shape[0, 1] + [nstep])
    X[..., 0] = x0

    U = zeros(up[:, :3].shape[0, 1] + [nstep])
    U[..., 0] = u0[:, 3]

    F4sp = zeros([setpts.size(0), nstep])
    F4sp[:, 0] = setpts[:, 0]

    kpar_hist = zeros([nstep])
    kpar_hist[0] = params.kpar

    ncpar_hist = zeros([nstep])
    ncpar_hist[0] = params.ncpar

    kc_pc = KC_PC_INIT
    taui_pc = TAUI_PC_INIT
    errn1_pc = ERRN1_PC_INIT
    f4sp_adj = F4SP_ADJ_INIT

    for ii, t in enumerate(times[1:]):

        scenario(t, xp, params, setpts)

        yp = plant.outputs(t, xp, up, update_memory=True)

        # Pressure control override loop
        errn_pc = 2900. - yp[:, 4]
        f4sp_adj = minimum(f4sp_adj + kc_pc * (errn_pc -
                           errn1_pc + dt * errn_pc / taui_pc), 0.)
        errn1_pc = errn_pc

        setpts_save = setpts.clone()
        setpts[:, 0] = setpts[:, 0] + f4sp_adj

        # PI multi-loop control
        errn = setpts - yp[:, ICV]
        delu = KCS * (errn - errn1 + dt * errn / TAUIS)
        up[:, IMV] = up[:, IMV] + delu
        errn1 = errn
        up[:3] = minimum(maximum(up[:3], tensor([0.]), tensor([100.])))

        # register values for future plots
        Y[..., ii+1] = yp
        U[..., ii+1] = up[:, 3]
        F4sp[..., ii+1] = setpts[0]
        kpar_hist[ii+1] = params.kpar
        ncpar_hist[ii+1] = params.ncpar
        setpts = setpts_save

        if ii == nstep - 1:
            break

        # Simulate plant to next sampling time
        def rhs(tt, xx):
            return plant.derivatives(tt, xx, up)

        sol = integrate(rhs, x0, times[ii+1:ii+2], rtol=1e-6, atol=1e-9)
        xp = sol[..., -1]

        X[..., ii+1] = xp

        return {'t': times,
                'x': X,
                'y': Y,
                'u': U,
                'f4sp': F4sp,
                'kpar': kpar_hist,
                'ncpar': ncpar_hist,
                'scenario': scenario}
