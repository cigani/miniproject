TODO: Burst Current Simulation
      Gaussian White Noise Simulation

2016-05-15
Implemented the afterhyperpolarization adaptation.
May need to debug as it appears to do little by itself;
however, with the M-type current (ex3.py), effects are
easily seen.  Compare ex2.py (M-type) with
ex3.py (AHP and M-ytpe).

Also implemented adaptive HH model from the following paper2:

1) The Effects of Spike Frequency Adaptation and Negative
Feedback on the Synchronization of Neural Oscillators

2) The contribution of spike-frequency adaptation to the
variability of spike responses in a sensory neuron

New modules are found in the bmnn/ref.  Check ref-ex2.py
and ref-ex3.py.  Check for errors later.

2016-05-11
Added a module to plot the steady state values of the
gating variables.

Increased ylim of current plot to better show the step
current used in the problem.

2016-05-06
Implemented a single-compartment model with an additional
M-type current, i.e. a slow voltage-dependent potassium
current, inducing spike-frequency adaptation.

Differential equation for the M-type current channel used
in order to avoid long optimization of the parameters.

Rewrote plotting scripts to be compatible with the code in
the bmnn directory.
