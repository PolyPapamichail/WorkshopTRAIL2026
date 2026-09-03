# TE reactor Python/PyTorch conversion

This package is a modernized SciPy-based Python/PyTorch port of the attached Fortran/MATLAB TE reactor example.

Files:
- `TEPlant.py`: numpy-python plant model
- `closed_loop_scenarios.py`: numpy-python steady-state computation, PI control and different control scenarios
- `TorchTEPlant.py`: PyTorch translation of TEPlant model
- `closed_loop_scenarios_tirch.py`: PyTorch translation of closed-loop scenarios and steady-state computation (still a draft for now)
- `integrators.py` : integrators for dynamical system derivatives in PyTorch code
- `plot_utils.py` : helpers for plots
- `generate_TEP_data.ipynb` : Jupyter notebook that explain how to go from limited closed-loop scenarios available to the generation of new time series in open-loop setting
- `env.yml` : configuration file to set up an Anaconda virtual environment (see the notebook for setup instructions)

Quick start:

Follow the notebook to setup your code environment and to have a full demonstration of the code.
