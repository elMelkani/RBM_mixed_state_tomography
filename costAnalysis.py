#Cost analyzer
import numpy as np;
import matplotlib.pyplot as plt
from scipy import stats;
import pickle;
import tomoHelper;

nVisible = 4;
nTMeasurements = nMeasurements = 3**nVisible;
nCosts = 4;

tom = tomoHelper.tomoHelper(nVisible, nVisible);
rho = tom.readRho(nVisible);
evs, eVs = np.linalg.eig(rho);
ind = np.argmax(np.real(evs));
actualWF = eVs.T[ind];
nComp = 2**nVisible;

store = pickle.load( open( "WFStore.p", "rb" ) );
#0.997694  ,  0.99737124store[6] for L1
store = [store[8], store[7] ,store[3], store[9], store[10], store[11] ]; #7 is cheating 9,10 are KL2, 8 is KL, 11 is from machine L2
#store = [store[8], store[10]];
store = np.asarray(store);

def LWhatDistance(mixedPs, otherPs, what):
	cost = 0;
	for i in range(mixedPs.shape[0]):
		cost = cost + (abs(mixedPs[i] - otherPs[i]))**what;
	return cost;

def HuberDistance(mixedPs, otherPs, delta):
	cost = 0;
	for i in range(mixedPs.shape[0]):
		if abs(mixedPs[i] - otherPs[i]) > delta:
			cost = cost + delta*(abs(mixedPs[i] - otherPs[i]) - 0.5*delta);
		else:
			cost = cost + 0.5*(mixedPs[i] - otherPs[i])**2;
	return cost;

def KLDivergence(mixedPs, otherPs):
	cost = 0;
	for i in range(mixedPs.shape[0]):
		cost = cost + mixedPs[i]*np.log(mixedPs[i]/otherPs[i]);
	return cost;

def KLReversed(mixedPs, otherPs):
	cost = 0;
	for i in range(mixedPs.shape[0]):
		cost = cost + otherPs[i]*np.log(otherPs[i]/mixedPs[i]);
	return cost;

def dPOVM(mixedPs, otherPs):
	cost = 0;
	for i in range(nMeasurements):
		k = sum(abs(mixedPs[i] - otherPs[i]));
		if k > cost:
			cost = k;
	return cost;

overlaps = np.zeros(store.shape[0]);
for i in range(store.shape[0]):
	overlaps[i] = np.abs(np.dot(store[i].conj(), actualWF))**2

for i in range(store.shape[0]):
	for j in range(store.shape[0] - 1):
		if overlaps[j] < overlaps[j + 1]:
			temp = overlaps[j];
			overlaps[j] = overlaps[j+1];
			overlaps[j+1] = temp;
			temp = store[j].copy();
			store[j] = store[j+1].copy();
			store[j+1] = temp.copy();


costsAll = np.zeros([store.shape[0] + 1,nCosts]);

POVMs = tom.makePOVMs( nVisible, nTMeasurements, nMeasurements);

measProbs = np.zeros([nMeasurements, nComp]);
for i in range(nMeasurements):
	for j in range(nComp):
		measProbs[i,j] = np.real(np.trace(np.dot(POVMs[i,j],rho)));
probsMix = measProbs.reshape([nMeasurements*nComp]);

pureProbs = tom.getPureProbStat(POVMs, actualWF)
costsAll[0,2] = dPOVM(measProbs, pureProbs);

probsPure = pureProbs.reshape([nMeasurements*nComp]);

probsAll = np.zeros([store.shape[0] + 1,nMeasurements*nComp]);
probsAll[0] = probsPure;
for i in range(store.shape[0]):
	probsFuck = tom.getPureProbStat(POVMs, store[i]);
	probsAll[i+1] = probsFuck.reshape([nMeasurements*nComp]);
	costsAll[i+1,2] = dPOVM(measProbs, probsFuck);




for i in range(store.shape[0] + 1):
	costsAll[i,0] = LWhatDistance(probsMix, probsAll[i], 1);
	costsAll[i,1] = LWhatDistance(probsMix, probsAll[i], 2);
	#costsAll[i,2] = HuberDistance(probsMix, probsAll[i], 0.1);
	#costsAll[i,2] = dPOVM(probsMix, probsAll[i]);
	costsAll[i,2] = KLDivergence(probsMix, probsAll[i]);
	costsAll[i,3] = KLReversed(probsMix, probsAll[i]);
	#costsAll[i,5] = LWhatDistance(probsMix, probsAll[i], 1.5);

normCosts = stats.zscore(costsAll);
pureCosts = costsAll[0,:];
otherCosts = costsAll[1:,:];

ind = np.arange(nCosts)  # the x locations for the groups
width = 0.1       # the width of the bars

fig = plt.figure(23)
ax = fig.add_subplot(111)

colors = [(0.1,0.2,0.5,0.3), 'g', 'b', 'y', 'black', 'cyan', 'magenta', 'green', 'b', 'y', 'cyan', 'black'];
cols = (0.1, 0.2, 0.5, 0.3)
tRects =[];
for i in range(store.shape[0] + 1):
	rects = ax.bar(ind+width*i, normCosts[i], width, color=(0,0.14*i,0.14*i,0.8));
	tRects.append(rects);

ax.set_ylabel('Costs (z-scored)')
ax.set_xticks(ind+width)
ax.set_xticklabels( ('L2', 'L1', 'KL1', 'KL2') )

LHS = (tRects[0][0],);
RHS = ('0) Pure (1.0)',);


for i in range(1, store.shape[0] + 1):
	LHS = LHS + ( tRects[i][0],);
	str2 = str(i)+') '+"{0:.5f}".format(overlaps[i-1]);
	RHS = RHS + ( str2,);
ax.set_title('Comparison of Cost Functions')
# Shrink current axis by 20%
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

ax.legend( LHS, RHS, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.);
#ax.legend()
fig.show()
