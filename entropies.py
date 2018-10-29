#Cost analyzer
import numpy as np;
import matplotlib.pyplot as plt
from scipy import stats;
import pickle;
import tomoHelper;

nVisible = 4;
nTMeasurements = nMeasurements = 3**nVisible;

def entropyMeasProbs(measProbs):
	ent = np.zeros(nMeasurements);
	for i in range(nMeasurements):
		ent[i] = sum([-p*np.log(p) for p in measProbs[i]]);
	return ent;

tom = tomoHelper.tomoHelper(nVisible, nVisible);
rho = tom.readRho(nVisible);
evs, eVs = np.linalg.eig(rho);
ind = np.argmax(np.real(evs));
actualWF = eVs.T[ind];
nComp = 2**nVisible;


store = pickle.load( open( "WFStore.p", "rb" ) );
#7 is cheating 9,10 are KL2, 8 is KL, 11 is from machine L2
store = np.asarray(store);

POVMs = tom.makePOVMs( nVisible, nTMeasurements, nMeasurements);

measProbs = np.zeros([nMeasurements, nComp]);
for i in range(nMeasurements):
	for j in range(nComp):
		measProbs[i,j] = np.real(np.trace(np.dot(POVMs[i,j],rho)));

entsRho = entropyMeasProbs(measProbs);

pureProbs = tom.getPureProbStat(POVMs, actualWF);
entsPure = entropyMeasProbs(pureProbs);

WFKL = store[8];
klProbs = tom.getPureProbStat(POVMs, WFKL);
entsKL = entropyMeasProbs(klProbs);

y = np.arange(nMeasurements)

fig = plt.figure(233)
ax = fig.add_subplot(111)
fSize = 25;

ax.bar(y, entsRho, align='center', alpha=0.4, facecolor='g', label="Experimental");
ax.bar(y, entsPure, align='center', alpha=0.4, facecolor='r', label="Pure");
#ax.bar(y, entsKL, align='center', alpha=0.4, facecolor='b', label="KLMinimized");

ax.set_xlabel('Measurement Index')
ax.set_ylabel('Entropies')
ax.set_title('Comparison of Entropies of the Probability Distributions')
#plt.text(60, .025, r'$\mu=100,\ \sigma=15$')
#ax.xlim([0, nMeasurements])
ax.grid(True);
ax.legend();
fig.show();
