#The RBM-class functions have grad-descent functions called fullStepCloser that return an object called gradStat. Pass that to this.
import matplotlib.pyplot as plt;
import numpy as np;

def plotter(gradStat, stri):
	lab = stri;
	length = gradStat.shape[1];
	t = np.linspace(0,length,num=length);
	plt.figure(1)
	plt.subplot(211)
	plt.plot(t,gradStat[0], label='Cost_'+lab);
	axes = plt.gca()
	axes.set_xlim([0,length]);
	plt.xlabel('Iterations', fontsize=14)
	plt.ylabel('Cost', fontsize=14)
	plt.grid(True);
	plt.legend();

	plt.subplot(212)
	plt.plot(t,gradStat[1], label='Overlap_'+lab);
	plt.plot(t,gradStat[2], label='P1_bound_'+lab);
	axes = plt.gca()
	axes.set_xlim([0,length]);
	#axes.set_ylim([0,1]);
	plt.xlabel('Iterations', fontsize=14)
	plt.ylabel('Overlap', fontsize=14)
	plt.grid(True);
	plt.legend();