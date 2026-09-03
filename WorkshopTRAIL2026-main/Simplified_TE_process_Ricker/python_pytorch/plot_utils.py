from itertools import product

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from plotly.subplots import make_subplots

DEFAULT_LAYOUT_1 = {'horizontal_spacing': 0.05,
                    'vertical_spacing': 0.1}  # (10, 7)
DEFAULT_LAYOUT_2 = {'horizontal_spacing': 0.05,
                    'vertical_spacing': 0.1}  # (12, 8)


class Loc():
    def __init__(self, row: int, col: int):
        self.row, self.col = row, col

    def __enter__(self):
        return {'row': self.row, 'col': self.col}

    def __exit__(self, exc_type, exc, tb):
        pass


def subplot_size(rows: int, cols: int, horizontal_spacing: float, vertical_spacing: float):
    subplot_height = (1 - (rows - 1) * vertical_spacing) / rows
    subplot_width = (1 - (cols - 1) * horizontal_spacing) / cols
    return subplot_height, subplot_width


def plot_ricker_style(result, figsize=(10, 7), save_path=None):
    """Reproduce the 2x2 MATLAB-style plot layout from te_4_pi.m.

    Parameters
    ----------
    result : dict
        Output from run_closed_loop_scenario(...)
    figsize : tuple
        Matplotlib figure size.
    save_path : str or Path or None
        Optional output path.
    """
    t = result['t']
    y = result['y'].T  # Transpose from [10, nstep] to [nstep, 10]
    u = result['u'].T  # Transpose from [4, nstep] to [nstep, 4]
    f4sp = result['f4sp']

    fig, axs = plt.subplots(2, 2, figsize=figsize, constrained_layout=True)

    axs[0, 0].plot(t, y[:, 3], label='F4')
    axs[0, 0].plot(t, f4sp, '--', label='F4 setpoint')
    axs[0, 0].set_title('Product Rate and Setpoint [kmol/h]')
    axs[0, 0].set_xlabel('Time [h]')
    axs[0, 0].legend()
    axs[0, 0].grid(True, alpha=0.25)

    axs[0, 1].plot(t, y[:, 4], color='tab:orange')
    axs[0, 1].set_title('Pressure [kPa]')
    axs[0, 1].set_xlabel('Time [h]')
    axs[0, 1].grid(True, alpha=0.25)

    axs[1, 0].plot(t, y[:, 6], color='tab:green')
    axs[1, 0].set_title('A in Purge [mole %]')
    axs[1, 0].set_xlabel('Time [h]')
    axs[1, 0].grid(True, alpha=0.25)

    axs[1, 1].plot(t, u[:, 0], label='u1')
    axs[1, 1].plot(t, u[:, 1], label='u2')
    axs[1, 1].plot(t, u[:, 2], label='u3')
    axs[1, 1].set_title('Manipulated Variables [%]')
    axs[1, 1].set_xlabel('Time [h]')
    axs[1, 1].legend()
    axs[1, 1].grid(True, alpha=0.25)

    if save_path is not None:
        fig.savefig(save_path, dpi=180, bbox_inches='tight')
    return fig, axs


def plot_scenario_comparison(results, figsize=(12, 8), save_path=None):
    """Plot multiple scenarios on the same 2x2 summary figure.

    Parameters
    ----------
    results : dict[str, dict]
        Mapping of scenario name to run_closed_loop_scenario output.
    """
    # Get consistent colors for each scenario
    colors = list(mcolors.TABLEAU_COLORS.values())[:len(results)]
    scenario_colors = {name: color for name,
                       color in zip(results.keys(), colors)}

    fig, axs = plt.subplots(2, 3, figsize=figsize, constrained_layout=True)
    for name, result in results.items():
        t = result['t']
        y = result['y'].T  # Transpose from [10, nstep] to [nstep, 10]
        u = result['u'].T  # Transpose from [4, nstep] to [nstep, 4]
        color = scenario_colors[name]
        axs[0, 0].plot(t, y[:, 3], label=name, color=color)
        axs[0, 0].plot(t, result['f4sp'], '--', color=color, alpha=0.35)
        axs[0, 1].plot(t, y[:, 4], label=name, color=color)
        axs[0, 2].plot(t, u[:, 0], label=name,
                       color=color)  # u1 - Feed 1 valve
        axs[1, 0].plot(t, y[:, 6], label=name, color=color)
        axs[1, 1].plot(t, u[:, 1], label=name,
                       color=color)  # u2 - Feed 2 valve
        axs[1, 2].plot(t, u[:, 2], label=name, color=color)  # u3 - Purge valve

    axs[0, 0].set_title('Product Rate F4 and Setpoint')
    axs[0, 0].set_xlabel('Time [h]')
    axs[0, 0].set_ylabel('kmol/h')
    axs[0, 0].grid(True, alpha=0.25)

    axs[0, 1].set_title('Pressure')
    axs[0, 1].set_xlabel('Time [h]')
    axs[0, 1].set_ylabel('kPa')
    axs[0, 1].grid(True, alpha=0.25)

    axs[0, 2].set_title('Feed 1 Valve u1 [%]')
    axs[0, 2].set_xlabel('Time [h]')
    axs[0, 2].set_ylabel('%')
    axs[0, 2].grid(True, alpha=0.25)

    axs[1, 0].set_title('A in Purge')
    axs[1, 0].set_xlabel('Time [h]')
    axs[1, 0].set_ylabel('mole %')
    axs[1, 0].grid(True, alpha=0.25)

    axs[1, 1].set_title('Feed 2 Valve u2 [%]')
    axs[1, 1].set_xlabel('Time [h]')
    axs[1, 1].set_ylabel('%')
    axs[1, 1].grid(True, alpha=0.25)

    axs[1, 2].set_title('Purge Valve u3 [%]')
    axs[1, 2].set_xlabel('Time [h]')
    axs[1, 2].set_ylabel('%')
    axs[1, 2].grid(True, alpha=0.25)

    handles, labels = axs[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right')

    if save_path is not None:
        fig.savefig(save_path, dpi=180, bbox_inches='tight')
    return fig, axs


def plotly_ricker_style(result, layout=DEFAULT_LAYOUT_1, save_path=None):
    """Reproduce the 2x2 MATLAB-style plot layout from te_4_pi.m.

    Parameters
    ----------
    result : dict
        Output from run_closed_loop_scenario(...)
    figsize : tuple
        Matplotlib figure size.
    save_path : str or Path or None
        Optional output path.
    """
    t = result['t']
    y = result['y'].T  # Transpose from [10, nstep] to [nstep, 10]
    u = result['u'].T  # Transpose from [4, nstep] to [nstep, 4]
    f4sp = result['f4sp']

    fig = make_subplots(rows=2, cols=2,
                        subplot_titles=['Product Rate and Setpoint [kmol/h]', 'Pressure [kPa]',
                                        'A in Purge [mole %]', 'Manipulated Variables [%]'],
                        **layout)

    with Loc(1, 1) as loc:
        fig.add_scatter(x=t, y=y[:, 3], mode='lines',
                        name='F4', **loc, legendgroup='legend2')
        fig.add_scatter(x=t, y=f4sp,    mode='lines', name='F4_setpoint', line={
                        'dash': 'dash'}, **loc, legendgroup='legend2')
        fig.update_xaxes(title_text='Time [h]', **loc)

    with Loc(1, 2) as loc:
        fig.add_scatter(x=t, y=y[:, 4], mode='lines', **loc,
                        showlegend=False, legendgroup='legend3')  # orange
        fig.update_xaxes(title_text='Time [h]', **loc)

    with Loc(2, 1) as loc:
        fig.add_scatter(x=t, y=y[:, 6], mode='lines', **loc,
                        showlegend=False, legendgroup='legend4')  # green
        fig.update_xaxes(title_text='Time [h]', **loc)

    with Loc(2, 2) as loc:
        fig.add_scatter(x=t, y=u[:, 0], mode='lines',
                        name='u1', **loc, legendgroup='legend5')
        fig.add_scatter(x=t, y=u[:, 1], mode='lines',
                        name='u2', **loc, legendgroup='legend5')
        fig.add_scatter(x=t, y=u[:, 2], mode='lines',
                        name='u3', **loc, legendgroup='legend5')
        fig.update_xaxes(title_text='Time [h]', **loc)

    if save_path is not None:
        fig.savefig(save_path, dpi=180, bbox_inches='tight')

    return fig


def plotly_scenario_comparison(results, layout=DEFAULT_LAYOUT_2, save_path=None):
    """Plot multiple scenarios on the same 2x2 summary figure.

    Parameters
    ----------
    results : dict[str, dict]
        Mapping of scenario name to run_closed_loop_scenario output.
    """
    # Define consistent colors for each scenario
    colors = list(mcolors.TABLEAU_COLORS.values())[:len(results)]
    scenario_colors = {name: color for name,
                       color in zip(results.keys(), colors)}

    fig = make_subplots(rows=2, cols=3,
                        subplot_titles=['Product Rate F4 and Setpoint', 'Pressure', 'Feed 1 Valve u1 [%]',
                                        'A in Purge', 'Feed 2 Valve u2 [%]', 'Purge Valve u3 [%]'],
                        **layout)

    for i, (name, result) in enumerate(results.items()):
        t = result['t']
        y = result['y'].T  # Transpose from [10, nstep] to [nstep, 10]
        u = result['u'].T  # Transpose from [4, nstep] to [nstep, 4]
        color = scenario_colors[name]

        # F4_setpoint - no legend
        fig.add_scatter(x=t, y=result['f4sp'], mode='lines', name='F4_setpoint',
                        line={'dash': 'dash', 'color': color}, row=1, col=1, showlegend=False)

        # F4 - Product Rate (legend entry)
        fig.add_scatter(x=t, y=y[:, 3], mode='lines', name=name, row=1, col=1,
                        line={'color': color}, showlegend=True, legendgroup=name)

        # Pressure - no legend (same group)
        fig.add_scatter(x=t, y=y[:, 4], mode='lines', row=1, col=2,
                        line={'color': color}, showlegend=False, legendgroup=name)

        # u1 - Feed 1 Valve (same group)
        fig.add_scatter(x=t, y=u[:, 0], mode='lines', row=1, col=3,
                        line={'color': color}, showlegend=False, legendgroup=name)

        # A in Purge (same group)
        fig.add_scatter(x=t, y=y[:, 6], mode='lines', row=2, col=1,
                        line={'color': color}, showlegend=False, legendgroup=name)

        # u2 - Feed 2 Valve (same group)
        fig.add_scatter(x=t, y=u[:, 1], mode='lines', row=2, col=2,
                        line={'color': color}, showlegend=False, legendgroup=name)

        # u3 - Purge Valve (same group)
        fig.add_scatter(x=t, y=u[:, 2], mode='lines', row=2, col=3,
                        line={'color': color}, showlegend=False, legendgroup=name)

    for row in range(3):
        for col in range(2):
            fig.update_xaxes(title_text='Time [h]', row=row + 1, col=col + 1)

    fig.update_yaxes(title_text='kmol/h', row=1, col=1)
    fig.update_yaxes(title_text='kPa',    row=1, col=2)
    fig.update_yaxes(title_text='%',      row=1, col=3)
    fig.update_yaxes(title_text='mole %', row=2, col=1)
    fig.update_yaxes(title_text='%',      row=2, col=2)
    fig.update_yaxes(title_text='%',      row=2, col=3)

    if save_path is not None:
        fig.savefig(save_path, dpi=180, bbox_inches='tight')

    return fig


# Comparison plotting functions for TorchTEPlant integration verification

def plotly_state_comparison(T, X_true, X_integrated, save_path=None):
    """Plot state comparison between true and integrated trajectories.

    Parameters
    ----------
    T : Tensor
        Time points, shape (n_steps,)
    X_true : Tensor
        True state trajectory, shape (n_steps, 8)
    X_integrated : Tensor
        Integrated state trajectory, shape (n_steps, 8)
    save_path : str or Path or None
        Optional output path.
    """
    state_names = ['vapor_A', 'vapor_B', 'vapor_C', 'liquid_%',
                   'valve1', 'valve2', 'valve3', 'valve4']

    fig = make_subplots(rows=2, cols=4, shared_xaxes=True,
                        vertical_spacing=0.08, horizontal_spacing=0.08)
    fig.update_layout(
        height=800,
        width=1400,
        title_text="TorchTEPlant State Integration Comparison",
        showlegend=True
    )

    for row in range(2):
        for col in range(4):
            idx = row * 4 + col

            # Add True line (legend entry only on first subplot)
            fig.add_scatter(
                x=T, y=X_true[:, idx],
                name="True",
                legendgroup="true",
                showlegend=(row == 0 and col == 0),
                line={'color': 'blue'},
                row=row+1, col=col+1
            )

            # Add Integrated line (dashed, legend entry only on first subplot)
            fig.add_scatter(
                x=T, y=X_integrated[:, idx],
                name="Integrated",
                legendgroup="integrated",
                showlegend=(row == 0 and col == 0),
                line={'color': 'red', 'dash': 'dash'},
                row=row+1, col=col+1
            )

            fig.update_xaxes(title_text="Time", row=row+1, col=col+1)
            fig.update_yaxes(
                title_text=state_names[idx], row=row+1, col=col+1)

    if save_path is not None:
        fig.savefig(save_path, dpi=180, bbox_inches='tight')
    return fig


def plotly_output_comparison(T, Y_true, Y_computed, save_path=None):
    """Plot output comparison between true and computed outputs.

    Parameters
    ----------
    T : Tensor
        Time points, shape (n_steps,)
    Y_true : Tensor
        True output trajectory, shape (n_steps, 10)
    Y_computed : Tensor
        Computed output trajectory, shape (n_steps, 10)
    save_path : str or Path or None
        Optional output path.
    """
    output_names = ['f1', 'f2', 'f3', 'f4', 'pt',
                    'vlpct', 'yA%', 'yB%', 'yC%', 'cost/mole']

    fig = make_subplots(rows=2, cols=5, shared_xaxes=True,
                        vertical_spacing=0.08, horizontal_spacing=0.08)
    fig.update_layout(
        height=700,
        width=1500,
        title_text="TorchTEPlant Output Comparison",
        showlegend=True
    )

    for row in range(2):
        for col in range(5):
            idx = row * 5 + col  # 2 rows x 5 cols = idx = row * 5 + col

            # Only plot subplots for valid output indices (0-9)
            if idx < len(output_names):
                # Add True line (legend entry only on first subplot)
                fig.add_scatter(
                    x=T, y=Y_true[:, idx],
                    name="True",
                    legendgroup="true",
                    showlegend=(row == 0 and col == 0),
                    line={'color': 'blue'},
                    row=row+1, col=col+1
                )

                # Add Computed line (dashed, legend entry only on first subplot)
                fig.add_scatter(
                    x=T, y=Y_computed[:, idx],
                    name="Computed",
                    legendgroup="computed",
                    showlegend=(row == 0 and col == 0),
                    line={'color': 'red', 'dash': 'dash'},
                    row=row+1, col=col+1
                )

                fig.update_xaxes(title_text="Time", row=row+1, col=col+1)
                fig.update_yaxes(
                    title_text=output_names[idx], row=row+1, col=col+1)

    if save_path is not None:
        fig.savefig(save_path, dpi=180, bbox_inches='tight')
    return fig


if __name__ == '__main__':

    from closed_loop_scenarios import _SCENARIOS, run_closed_loop_scenario

    # Single scenario plot, matching the original MATLAB layout, this time in plotly
    # which are interactive plots. It will open the figures in the browser

    TE_RESULTS = run_closed_loop_scenario()
    fig = plotly_ricker_style(TE_RESULTS)

    # Print the type to verify
    print('Figure type:', type(fig))
    print('Has show:', hasattr(fig, 'show'))

    fig.show(renderer="browser")

    # Multi-scenario comparison plot
    scenarios = {
        name: run_closed_loop_scenario(scenario=name)
        for name in _SCENARIOS
    }
    fig2 = plotly_scenario_comparison(scenarios)
    fig2.show(renderer="browser")
