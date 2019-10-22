#For psi1
import rhoRBM; #the class with the RBM
import numpy as np;
import tomoHelper; #tomographyHelper generates measurements statistics and reads rho from text file etc
import plotter; #plots data of gradientDescent
import pickle

#XXX """SECTION: PARAMETERS"""

nVisible = 6;
nHidden = nVisible; # usually alpha = nV/nH = 1
nTMeasurements = 3**nVisible; #total possible Pauli measurements
nComp = 2**nVisible; #dimension of Hilbert space
nMeasurements = int((1.5**nVisible)*nVisible*3); #int((1.5**nVisible)*nVisible*3) + 70;
#nMeasurements = nTMeasurements;

#XXX """SECTION: HELPERS AND PREPARATION"""

tom = tomoHelper.tomoHelper(nVisible, nHidden);
NQS = rhoRBM.rhoRBM(nVisible, nHidden);
eValueMatrix = NQS.getEigenValueMatrix();

rho = tom.readRho(nVisible); #read from paper's rho
evs, eVs = np.linalg.eig(rho);#eigenvalues and eigenvectors
actualWF = eVs.T[0];
aWFKet = actualWF.reshape([nComp, 1]);
aWFBra = actualWF.reshape([1, nComp]).conj();

#oldRes = pickle.load( open( "save.p", "rb" ) ) #syntax to load
oldRes = pickle.load( open( "save6_9976_6899.p", "rb" ) )

#XXX """SECTION: RUN THE MAIN"""

tommy = tom.makeUnitaries(nVisible, nTMeasurements, nMeasurements, rho);
Unitaries = tommy[0]; #[nMeasurements,nComp,nComp] Unitaries corresponding to the measurements
measProbs = tommy[1]; #[nMeasurements,nComp] statistics of the density matrix

NQS.rWeights = oldRes['rWeights'];
NQS.iWeights = oldRes['iWeights'];
print("Started\n") 
gradStat = NQS.fullStepCloser(measProbs, Unitaries, learningRate = 10, iterations = 5, actualWF = actualWF, whichCost = "L1.5", initFraction = 0.4);
#initfraction is the fraction of measurements that are considered in one iteration of batchwise grad-descent.
#Learning rate of around 10 for 4qub, 30 for 6qub is nice for KL2 and around 40 for L2. KL2 is much faster better use that

WF = NQS.getFullWF(eValueMatrix);

c = np.zeros(nMeasurements);
for i in range(nMeasurements):
	c[i] = 0.5*NQS.getAbsSumCost(WF, measProbs[i], Unitaries[i]);

evTraceestimate = 1 - max(c);

WFKet = WF.reshape([nComp, 1]);
WFBra = WF.reshape([1, nComp]).conj();

rho2 = np.dot(WFKet, WFBra);
diff = rho - rho2;
evs2, eVs2 = np.linalg.eig(diff);
TraceRBMandRho = 0.5*sum(abs((evs2))); #real trace distance bw RBM pure state and density matrix

WFMeasProbs = np.zeros([nMeasurements, nComp]);
allFracs = np.zeros([nMeasurements, nComp]);
for i in range(nMeasurements):
	Unitary = Unitaries[i];
	WFRotated = np.dot(Unitary, WFKet);
	WFMeasProbs[i] = (abs(WFRotated.conj()*WFRotated)).reshape(nComp,);

newP = 1;
for i in range(nMeasurements):
	for j in range(nComp):
		if WFMeasProbs[i,j] == 0:
			continue;
		frac = measProbs[i,j]/WFMeasProbs[i,j];
		allFracs[i,j] = frac;
		if frac < newP:
			newP = frac;
			print('['+str(i) + ',' + str(j) + ']: ' + str(measProbs[i,j]))

plotter.plotter(gradStat, "xyz");


print("emperical: "+ str(evTraceestimate));
print("bound: " + str(newP));


#XXX """SAVE THE RESULTS"""
results = {};
results['WF'] = WF;
results['rWeights'] = NQS.rWeights;
results['iWeights'] = NQS.iWeights;
results['measProbs'] = measProbs;
results['WFMeasProbs'] = WFMeasProbs;
results['EVbound'] = newP;
results['gradStat'] = gradStat;
results['actualWF'] = eVs.T[0];
results['RBMactualWFoverlap'] = np.abs(np.dot(WF.conj(), eVs.T[0]))**2;
results['estimatedTraceRBMandRho'] = evTraceestimate;
results['actualEigenValue'] = evs[0];
results['TraceRBMandRho'] = TraceRBMandRho;

#pickle.dump( results, open( "save8_9942_5049.p", "wb" ) )

oldWf = np.array(oldRes['WF'])[0];
W = tom.getWState(nVisible);
overlap2 = np.abs(np.dot(W, oldWf)**2);