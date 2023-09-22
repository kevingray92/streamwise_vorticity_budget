# streamwise_vorticity_budget

BackwardsTrajectories.py:
Code used to produe backwards trajectories. We first created CM1 output files with 2 second output frequency via a restart of CM1. This code then calculates backwards trajectories using the 2 second output file.

ForwardTrajectories.py:
Code used to produce forward trajectories to confirm SVCs were tilted by, and became part of, the low-level mesocyclone. These used the 1 minute output files from CM1.

filter.py : 
Code used to filter the backwards trajectories before they can be used in the budget.py code.

budget.py : 
Code used to calculate streamwise, crosswise, and vertical vorticity and the forcing terms along a trajectory. The forcing terms at each time step are saved in a .npy file for each term.
