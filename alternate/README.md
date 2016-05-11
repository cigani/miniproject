TODO: Burst Current Simulation
      Gaussian White Noise Simulation

OPTIONAL: Implement afterhyperpolarization adaptation

2016-05-06
Implemented a single-compartment model with an additional
M-type current, i.e. a slow voltage-dependent potassium
current, inducing spike-frequency adaptation.

Differential equation for the M-type current channel used
in order to avoid long optimization of the parameters.

Rewrote plotting scripts to be compatible with the code in
the bmnn directory.

2016-05-11
Added a module to plot the steady state values of the
gating variables.

Increased ylim of current plot to better show the step
current used in the problem.