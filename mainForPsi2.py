#For psi2
import rhoRBM; #the class with the RBM
import numpy as np;
import tomoHelper; #tomographyHelper generates measurements statistics and reads rho from text file etc
import plotter; #plots data of gradientDescent
import pickle;

#XXX """SECTION: PARAMETERS"""

nVisible = 8;
nHidden = nVisible; # usually alpha = nV/nH = 1
nTMeasurements = 3**nVisible; #total possible Pauli measurements
nComp = 2**nVisible; #dimension of Hilbert space
nMeasurements = int((1.5**nVisible)*nVisible*3); #int((1.5**nVisible)*nVisible*3) + 70; 
#nMeasurements = nTMeasurements;

def checkMeas(Unitaries, measProbs, WFMeasProbs, P): #returns the Unitaries and corresponding probs that are useful for next run
	i = 0;
	measProbs3 = (measProbs - P*WFMeasProbs);
	tCount = nMeasurements;
	newMeasProbs = [];
	Unitaries2 = [];
	while i < tCount:
		flag = 0;
		for j in range(nComp):
			if measProbs3[i,j] < 0.000:
				flag = 1;
				break;
		if flag == 0:
			newMeasProbs.append(measProbs3[i]/(1-P));
			Unitaries2.append(Unitaries[i]);
		i = i + 1;
		#newMeasProbs = newMeasProbs/(1 - P);
	return [newMeasProbs, Unitaries2];

#XXX """SECTION: HELPERS AND PREPARATION"""

tom = tomoHelper.tomoHelper(nVisible, nHidden);
NQS = rhoRBM.rhoRBM(nVisible, nHidden);
eValueMatrix = NQS.getEigenValueMatrix();

rho = tom.readRho(nVisible); #read from paper's rho
evs, eVs = np.linalg.eig(rho);#eigenvalues and eigenvectors
actualWF = eVs.T[1];
aWFKet = actualWF.reshape([nComp, 1]);
aWFBra = actualWF.reshape([1, nComp]).conj();

oldRes = pickle.load( open( "save8_9942_5049.p", "rb" ) ) #syntax to load

#XXX """SECTION: RUN THE MAIN"""

#tommy = tom.makeUnitaries(nVisible, nTMeasurements, nMeasurements, rho);
tommy = oldRes['tommy']
Unitaries = tommy[0]; #[nMeasurements,nComp,nComp] Unitaries corresponding to the measurements
measProbs = tommy[1]; #[nMeasurements,nComp] statistics of the density matrix

priorWF = oldRes['WF'];

priorWFKet = priorWF.reshape([nComp, 1]);
priorWFBra = priorWF.reshape([1, nComp]).conj();

priorWFMeasProbs = np.zeros([nMeasurements, nComp]);
for i in range(nMeasurements):
	Unitary = Unitaries[i];
	WFRotated = np.dot(Unitary, priorWFKet);
	priorWFMeasProbs[i] = (abs(WFRotated.conj()*WFRotated)).reshape(nComp,);

PP = oldRes['EVbound'];
x = checkMeas(Unitaries, measProbs, priorWFMeasProbs, PP);
Unitaries2 = np.asarray(x[1]); #[nMeasurements,nComp,nComp] Unitaries corresponding to the measurements
newMeasProbs = np.asarray(x[0]); #[nMeasurements,nComp] statistics of the density matrix

NQS.oldWFs.append(priorWF);

#useOld = pickle.load( open( "save5_2_7335.p", "rb" ) ) #syntax to load
#NQS.iWeights = useOld['iWeights'];
#NQS.rWeights = useOld['rWeights'];

gradStat = NQS.fullStepCloser(newMeasProbs, Unitaries2, learningRate = 45, iterations = 501, actualWF = eVs.T[1], whichCost = "L1.5", initFraction = 0.4, orthogonalFactor = 1);

WF2 = NQS.getFullWF(eValueMatrix);
ortho = np.dot(priorWF.conj(), WF2);

c2 = np.zeros(newMeasProbs.shape[0]);

for i in range(newMeasProbs.shape[0]):
	c2[i] = 0.5*NQS.getAbsSumCost(WF2, newMeasProbs[i], Unitaries[i]);
evTraceestimate2 = 1 - max(c2);

WFKet2 = WF2.reshape([nComp, 1]);
WFBra2= WF2.reshape([1, nComp]).conj();

WFMeasProbs2 = np.zeros([newMeasProbs.shape[0], nComp]);
allFracs2 = np.zeros([newMeasProbs.shape[0], nComp]);
for i in range(newMeasProbs.shape[0]):
	Unitary = Unitaries2[i];
	WFRotated = np.dot(Unitary, WFKet2);
	WFMeasProbs2[i] = (abs(WFRotated.conj()*WFRotated)).reshape(nComp,);

allFracs = np.zeros([nMeasurements, nComp]);
newP2 = 1;
for i in range(newMeasProbs.shape[0]):
	for j in range(nComp):
		if WFMeasProbs2[i,j] == 0:
			continue;
		frac = newMeasProbs[i,j]/WFMeasProbs2[i,j];
		allFracs[i,j] = frac;
		if frac < newP2:
			print('['+str(i) + ',' + str(j) + ']: ' + str(newMeasProbs[i,j]))
			newP2 = frac;

plotter.plotter(gradStat, "xyz");

#XXX """SAVE THE RESULTS"""
results = {};
results['WF'] = WF2;
results['rWeights'] = NQS.rWeights;
results['iWeights'] = NQS.iWeights;
results['priorWF'] = priorWF;
results['measProbs'] = newMeasProbs;
results['WFMeasProbs'] = WFMeasProbs2;
results['EVbound'] = newP2;
results['gradStat'] = gradStat;
results['actualWF'] = eVs.T[1];
results['RBMactualWFoverlap'] = np.abs(np.dot(WF2.conj(), eVs.T[1]))**2;
results['actualEigenValue'] = evs[1];
results['estimatedTraceRBMandRho'] = evTraceestimate2;

#pickle.dump( results, open( "save8_2_2457.p", "wb" ) )

