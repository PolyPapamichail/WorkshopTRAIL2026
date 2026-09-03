from typing import Callable

import torchode as to
from torch import Tensor, compile


def integrate(fun: Callable, x0: Tensor, tspan: Tensor,
              atol: float = 1e-6, rtol: float = 1e-3):
    """
    Integrate ODE dy/dt = fun(t, y) using torchode Dopri5 solver.

    Args:
        fun: ODE function with signature fun(t, y) -> dy/dt
        x0: Initial state tensor of shape (batch_size, n_states)
        tspan: Time span tensor of shape (batch_size, n_times)
        atol: Absolute tolerance
        rtol: Relative tolerance

    Returns:
        y_out: Solution at all time points, shape (batch_size, n_times, n_states)
    """
    term = to.ODETerm(fun)
    step_method = to.Dopri5(term=term)
    step_size_controller = to.IntegralController(
        atol=atol, rtol=rtol, term=term)
    solver = to.AutoDiffAdjoint(step_method, step_size_controller)
    jit_solver = compile(solver)

    return jit_solver.solve(to.InitialValueProblem(y0=x0, t_eval=tspan)).ys


def integrate_wcontrol(fun: Callable, x0: Tensor, tspan: Tensor, u: Tensor,
                       atol: float = 1e-6, rtol: float = 1e-3):
    """
    Integrate ODE dy/dt = fun(t, y, args) using torchode Dopri5 solver
    with control inputs passed as args.

    Args:
        fun: ODE function with signature fun(t, y, args) -> dy/dt where args=u
        x0: Initial state tensor of shape (batch_size, n_states)
        tspan: Time span tensor of shape (batch_size, n_times)
        u: Control input tensor of shape (batch_size, n_controls)
        atol: Absolute tolerance
        rtol: Relative tolerance

    Returns:
        y_out: Solution at all time points, shape (batch_size, n_times, n_states)
    """
    term = to.ODETerm(fun, with_args=True)
    step_method = to.Dopri5(term=term)
    step_size_controller = to.IntegralController(
        atol=atol, rtol=rtol, term=term)
    solver = to.AutoDiffAdjoint(step_method, step_size_controller)
    jit_solver = compile(solver)

    return jit_solver.solve(to.InitialValueProblem(y0=x0, t_eval=tspan), args=u).ys
