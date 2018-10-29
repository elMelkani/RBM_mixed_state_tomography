# RBM_mixed_state_tomography
Tomography using the NQS ansatz and a procedure of step-wise extraction of pure eigenstates of the in-general mixed states 

A short description of the files:

fullRhoMain3 is the main file that is to be run to perform tomography.

rhoRBM3 is the class file for the RBMs. It contains all the functionality of the RBMs.

tomoHelper has helper functions to generate random density matrices for testing or to generate random POVMs/Unitaries etc.

plotter is used to plot the results of the main file run

costAnalysis was used to analyze which the performance of the distance measures KL, KL2, L2, etc.

Entropies was used to make the plot of the entropy of measurements of mixed state compared with corresponding measurements on the pure eigenstate.
