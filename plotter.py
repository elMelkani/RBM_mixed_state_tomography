#The RBM-class functions have grad-descent functions called fullStepCloser that return an object called gradStat. Pass that to this.
import matplotlib.pyplot as plt;
import numpy as np;

def plotter(gradStat):
	length = gradStat.shape[1];
	t = np.linspace(0,3*length,num=length);
	plt.figure(1)
	plt.subplot(211)
	plt.plot(t,gradStat[0], label='mom');
	axes = plt.gca()
	axes.set_xlim([0,3*length]);
	plt.xlabel('Iterations', fontsize=14)
	plt.ylabel('Cost', fontsize=14)

	plt.subplot(212)
	plt.plot(t,gradStat[1], label='mom');
	plt.plot(t,gradStat[2], label='mom');
	axes = plt.gca()
	axes.set_xlim([0,3*length]);
	#axes.set_ylim([0,1]);
	plt.xlabel('Iterations', fontsize=14)
	plt.ylabel('Overlap', fontsize=14)