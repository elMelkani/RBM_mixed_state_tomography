#Most up to date main file to run for density matrix tomography. Works with rhoRBM3
import rhoRBM3; #the class with the RBM
import numpy as np;
import tomoHelper; #tomographyHelper generates measurements statistics and reads rho from text file etc
import plotter; #plots data of gradientDescent
import pickle
#PARAMETERS:
nVisible = 4;
nHidden = 4; # usually alpha = nV/nH = 1
nTMeasurements = 3**nVisible; #total possible Pauli measurements
nComp = 2**nVisible; #dimension of Hilbert space
nMeasurements = int((1.5**nVisible)*nVisible*3) + 70; #int((1.5**nVisible)*nVisible*3) + 70; # we will only work with a smaller number than nTMeasurements
nMeasurements = nTMeasurements;

def getFidelity(rho, rho2):
	from scipy import linalg;
	rhoRoot = linalg.sqrtm(rho);
	mat = np.dot(rhoRoot, rho2);
	mat = np.dot(mat, rhoRoot);
	mat2 = linalg.sqrtm(mat);
	tr = np.trace(mat2);
	return tr**2;

def checkMeas(Unitaries, measProbs, probs2, P):
	i = 0;
	measProbs3 = (measProbs - P*probs2)/(1 - P);
	tCount = nMeasurements;
	measProbs2 = [];
	Unitaries2 = [];
	while i < tCount:
		flag = 0;
		for j in range(nComp):
			if measProbs3[i,j] < 0:
				flag = 1;
				break;
		if flag == 0:
			measProbs2.append(measProbs3[i]);
			Unitaries2.append(Unitaries[i]);
			#measProbs = np.delete(measProbs, i, 0);
			#Unitaries = np.delete(Unitaries, i, 0);
		i = i + 1;
	return [measProbs2, Unitaries2];

def newRho(rho2, p):
	rho3 = (rho - p*rho2)/(1-p);
	return rho3;

tom = tomoHelper.tomoHelper(nVisible, nHidden);
NQS = rhoRBM3.rhoRBM3(nVisible, nHidden);
eValueMatrix = NQS.getEigenValueMatrix();

results = {};
results['rho'] = [];
results['WF'] = [];
results['measProbs'] = [];
results['probs2'] = [];
results['newP'] = [];
results['newP2'] = [];
results['newP2c'] = [];
results['gradStat'] = [];
results['allDistances'] =[];
results['empiricalP'] = [];
results['empiricalDist'] = [];
results['actualWF'] = [];
results['actualP'] = [];
results['overlap'] = [];
results['realDist'] = [];
results['rWeights'] = [];
results['iWeights'] = [];

KK = 1; #How many eigenvectors are we interested in? Usually 1

rho = tom.readRho(nVisible); #read from paper's rho
evs, eVs = np.linalg.eig(rho);#eigenvalues and eigenvectors
ind = np.argmax(np.real(evs));
actualWF = eVs.T[ind];
aWFKet = actualWF.reshape([nComp, 1]);
aWFBra = actualWF.reshape([1, nComp]).conj();
#rho2 = np.dot(aWFKet, aWFBra);
#RUN FROM HERE

#oldRes = pickle.load( open( "save.p", "rb" ) ) #syntax to load

for runs in range(KK):
	results['rho'].append(rho);
	tommy = tom.makeUnitaries(nVisible, nTMeasurements, nMeasurements, rho);
	Unitaries = tommy[0]; #[nMeasurements,nComp,nComp] Unitaries corresponding to the measurements
	measProbs = tommy[1]; #[nMeasurements,nComp] statistics of the density matrix

	results['measProbs'].append(measProbs);

	evs, eVs = np.linalg.eig(rho);#eigenvalues and eigenvectors
	ind = np.argmax(np.real(evs));
	actualWF = eVs.T[ind]; # eigenvector corresponding to largest eigenvalue
	results['actualWF'].append(actualWF);
	results['actualP'].append(np.real(evs[ind]));

	NQS = rhoRBM3.rhoRBM3(nVisible, nHidden); #reInitialize weights randomly
	#NQS.rWeights = oldRes['rWeights'][0];
	#NQS.iWeights = oldRes['iWeights'][0];
	gradStat = NQS.fullStepCloser(measProbs, Unitaries, learningRate = 10, iterations = 2000, actualWF = actualWF, whichCost = "KL2", initFraction = 0.5);
	#initfraction is the fraction of measurements that are considered in one iteration of batchwise grad-descent.
	#Learning rate of around 10 for 4qub, 30 for 6qub is nice for KL2 and around 40 for L2. KL2 is much faster better use that
	results['gradStat'].append(gradStat);

	WF = NQS.getFullWF(eValueMatrix);
	results['WF'].append(WF);
	results['overlap'].append(np.abs(np.dot(WF.conj(), actualWF))**2); #overlap bw RBM state and actual eigenstate = 1 - \epsilon
	results['rWeights'].append(NQS.rWeights);
	results['iWeights'].append(NQS.iWeights);

	c = np.zeros(nMeasurements);
	for i in range(nMeasurements):
		c[i] = 0.5*NQS.getAbsSumCost(WF, measProbs[i], Unitaries[i]);
	results['allDistances'].append(c);

	argMax = np.argmax(c);
	results['empiricalP'].append(1 - c[argMax]); #an old procedure. ignore
	results['empiricalDist'].append(c[argMax]); #estimated trace distance bw RBM pure state and density matrix

	WFKet = WF.reshape([nComp, 1]);
	WFBra = WF.reshape([1, nComp]).conj();

	rho2 = np.dot(WFKet, WFBra);
	diff = rho - rho2;
	evs2, eVs2 = np.linalg.eig(diff);
	results['realDist'].append(0.5*sum(abs((evs2)))); #real trace distance bw RBM pure state and density matrix

	probs2 = np.zeros([nMeasurements, nComp]);
	allFracs = np.zeros([nMeasurements, nComp]);
	for i in range(nMeasurements):
		Unitary = Unitaries[i];
		WFRotated = np.dot(Unitary, WFKet);
		probs2[i] = (abs(WFRotated.conj()*WFRotated)).reshape(nComp,);
	results['probs2'].append(probs2);

	newP = 1 - c[argMax] - 0.06;
	results['newP'].append(newP); #useless. ignore

	for i in range(nMeasurements):
		for j in range(nComp):
			if probs2[i,j] == 0:
				continue;
			frac = measProbs[i,j]/probs2[i,j];
			allFracs[i,j] = frac;
			if frac < newP:
				newP = frac;
				print(i); print(j);
	fracMins = np.zeros(nMeasurements);
	for i in range(nMeasurements):
		fracMins[i] = min(allFracs[i]);
	results['newP2'].append(newP); # estimated p by new procedure. This is important result
	#now preparing rho for next run...
	#rho = (rho - newP2*rho2)/(1 - newP2);
gradStat2 = np.asarray([gradStat[0][900:gradStat.shape[1]-1], gradStat[1][900:gradStat.shape[1]-1]])
#plotter.plotter(gradStat);

#reconstruct the density matrix:
solution = 0;
normalizer = 1;
for runs in range(KK):
	WF = results['WF'][runs];
	p = results['newP2'][runs];
	WFKet = WF.reshape([nComp, 1]);
	WFBra = WF.reshape([1, nComp]).conj();
	solution = solution + p*np.dot(WFKet, WFBra)*normalizer;
	normalizer = normalizer - p;
solution = solution/np.trace(solution);
results['solution'] = solution;

rhoOriginal = results['rho'][0];
diff = rhoOriginal - solution;
evs2, eVs2 = np.linalg.eig(diff);
finalConvergence = 0.5*sum(abs((evs2)));
print(finalConvergence); #trace distance bw reconstructed DM and actual DM. Can also use fidelity function

results['finalConvergence'] = finalConvergence;


pickle.dump( results, open( "save2.p", "wb" ) ) #always save results to look at later
#favorite_color = pickle.load( open( "save2.p", "rb" ) ) #syntax to load

print(results['actualP']);
print(results['empiricalP']);
print(results['newP2'] );

PP = np.real(evs[ind]);
x = checkMeas(Unitaries, measProbs, probs2, PP);
Unitaries2 = np.asarray(x[1]); #[nMeasurements,nComp,nComp] Unitaries corresponding to the measurements
measProbs2 = np.asarray(x[0]); #[nMeasurements,nComp] statistics of the density matrix
NQS = rhoRBM3.rhoRBM3(nVisible, nHidden); #reInitialize weights randomly
gradStat = NQS.fullStepCloser(measProbs2, Unitaries2, learningRate = 10, iterations = 2000, actualWF = eVs.T[1], whichCost = "KL2", initFraction = 1);

def getFids():
	fids = np.zeros(nMeasurements);
	for i in range(nMeasurements):
		for j in range(nComp):
			fids[i] += np.sqrt(measProbs[i,j]*probs2[i,j]);
		fids[i] = fids[i]*fids[i];
	return fids;
