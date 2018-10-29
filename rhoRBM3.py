#Most uptodate double-RBM for learning the closest pure state to a given density matrix.
import numpy as np;
import random;
class rhoRBM3:
	def __init__(self, nVisible, nHidden, GibbsSample = False):
		#r (real) is for mod-amplitude-RBM; i (imaginary) is for phase-RBM
		self.nVisible = nVisible;
		self.nHidden = nHidden;
		self.nComp = 2**self.nVisible;
		self.debugPrint = True; #do we print the error etc during gradient descent

		rand = np.random.RandomState(123);
		WeightRange = 0.2*np.sqrt(6. / (nVisible + nHidden));
		self.rWeights = np.asarray(rand.uniform(low=-WeightRange, high=WeightRange, size=(nVisible+1, nHidden+1)));
		self.rWeights[0,0] = 0;
		#wts is [(num_visible+1) * (num_hidden+1)] of the form:
		#[ -- b1  b2  b3  ...]
		#[ a1 W11 W21 W31 ...]
		#[ a2 W12 W22 W32 ...]
		#[... ... ... ... ...]
		self.iWeights = np.asarray(rand.uniform(low=-WeightRange, high=WeightRange, size=(nVisible+1, nHidden+1)));
		self.iWeights[0,0] = 0;

		self.nWeights = self.rWeights.shape[0]*self.rWeights.shape[1] - 1;
		self.GibbsSample = GibbsSample; #do we exactly compute amplitude/phase or gibbs sample them?

#Each "eigenValue vector" is a vector of eigenvalues. It is NOT an eigenvector.
#Note that we start with (1,1) which is eigenvalue for |Z+Z+> ie \psi = [1,0,0,0]. then (1,-1) which is ev for psi = 0100 etc.
	def getEigenValueMatrix(self):
		toBinary = lambda x : ''.join(reversed( [str((x >> i) & 1) for i in range(self.nVisible)] ) );
		eBasis = np.zeros([2**self.nVisible, self.nVisible]);
		for i in range(2**self.nVisible): #2**nVisible number of states each having unique eigenValueBasis
			basisString = toBinary(i); #Convert to binary representation in the form of string
			basis = [int(j) for j in list(basisString)]; #finally in the form required by RBM
			basis = np.asarray(basis);
			basis = 1 - 2*basis;
			eBasis[i] = basis;
		return eBasis;

	def getNormalizedPhase(self, EM): #gives you all the pMu's takes care of the singularities so use this
		if self.GibbsSample == True:
			state = self.getGibbsStat(rORi = 'i');
			for i in range(state.shape[0]):
				if state[i] == 0:
					state[i] = 1;
		else:
			state = np.ones(self.nComp);
			for i in range(self.nComp):
				sigma = EM[i];
				biasedSigma = np.insert(sigma, 0, 1);
				for j in range(1, self.nHidden+1):
					act = sum(self.iWeights[:,j]*biasedSigma);
					state[i] = state[i]*2*np.cosh(act);
				biasedSigma[0] = 0; #getting ready for ai's part
				act = sum(self.iWeights[:,0]*biasedSigma);
				state[i] = state[i]*np.exp(act);
		exponent = 1j*0.5*np.log(state);#Note that log and exp of a vector is log and exp of each entry of the vector.
		PhiMuFactor = np.exp(exponent);
		return PhiMuFactor;

	def getBigAmplitude(self, EM): #gives you all the pLambda's. For Gibbs implementation borrow from above function
		if self.GibbsSample == True:
			state = self.getGibbsStat(rORi = 'r'); return state;
		state = np.ones(self.nComp);
		for i in range(self.nComp):
			sigma = EM[i];
			biasedSigma = np.insert(sigma, 0, 1);
			for j in range(1, self.nHidden+1):
				act = sum(self.rWeights[:,j]*biasedSigma);
				state[i] = state[i]*2*np.cosh(act);
			biasedSigma[0] = 0; #getting ready for ai's part
			act = sum(self.rWeights[:,0]*biasedSigma);
			state[i] = state[i]*np.exp(act);
		return state;

	def getNormalizedAmplitude(self, EM): #gives you all the pLambda's/Z
		PLambda = self.getBigAmplitude(EM);
		Z = sum(PLambda);
		PLambda = PLambda/Z;
		return PLambda;

	def getFullWF(self, EM): #gives the complete WF with amplitude and phase
		PhiMuFactor = self.getNormalizedPhase(EM);
		PLambda = self.getNormalizedAmplitude(EM);
		WF = np.sqrt(PLambda)*PhiMuFactor; #for np arrays this is element-wise multiplication.
		return WF;

	def getPhaseDerivativeSigma(self, sigma_b):
		#returns a row of derivatives of all weights of phaseRBM(sigma_b). The first entry of sigma_b is 1 (one) - the bias
		sigma_b = np.asarray(sigma_b);
		sigma_b = sigma_b.reshape(sigma_b.shape[0],1); #it becomes a column
		M = np.dot(self.iWeights.T, sigma_b); #to get as each entry = b_i + <sum>W_ij.sigma_j . Ignore first entry
		M = M.reshape(M.shape[0],1);
		M = self.tanh(M);
		M[0] = 1;
		M = np.dot(sigma_b, M.T); #derivatives (nV+1)*(nH+1) (ignore first entry)
		M[0,0] = 0;
		M = M.reshape(1,M.shape[0]*M.shape[1]); #become a row
		M=M[0][1:];# get rid of useless first entry of the weights
		return M;

	def getAmpDerivativeSigma(self, sigma_b):
		#returns a row of derivatives of all weights of amplitudeRBM(sigma_b). The first entry of sigma_b is 1 (one) - the bias
		sigma_b = np.asarray(sigma_b);
		sigma_b = sigma_b.reshape(sigma_b.shape[0],1); #it becomes a column
		M = np.dot(self.rWeights.T, sigma_b); #to get as each entry = b_i + <sum>W_ij.sigma_j . Ignore first entry
		M = M.reshape(M.shape[0],1);
		M = self.tanh(M);
		M[0] = 1;
		M = np.dot(sigma_b, M.T); #derivatives (nV+1)*(nH+1) (ignore first entry)
		M[0,0] = 0;
		M = M.reshape(1,M.shape[0]*M.shape[1]); #become a row
		M=M[0][1:];# get rid of useless first entry of the weights
		return M;

	def getAmpDerivativesMatrix(self, sigmaMatrix): #[nComp * rNWeights] no more useless weight from now on
		sigmaMatrix = np.insert(sigmaMatrix, 0, 1, axis = 1); #insert bias states of 1 into the first column
		D = np.zeros([2**self.nVisible, self.nWeights]);
		for j in range(2**self.nVisible):
			sigma = sigmaMatrix[j,:]; #take the jth row
			D[j] = self.getAmpDerivativeSigma(sigma);
		return D;

	def getPhaseDerivativesMatrix(self, sigmaMatrix): #[nComp * iNWeights] no more useless weight from now on
		sigmaMatrix = np.insert(sigmaMatrix, 0, 1, axis = 1); #insert bias states of 1 into the first column
		D = np.zeros([2**self.nVisible, self.nWeights]);
		for j in range(2**self.nVisible):
			sigma = sigmaMatrix[j,:]; #take the jth row
			D[j] = self.getPhaseDerivativeSigma(sigma);
		return D;

	def getAbsSumCost(self,WF, measProbs, Unitary):
		cost = 0;
		WFKet = WF.reshape([2**self.nVisible, 1]);
		WFRotated = np.dot(Unitary, WFKet);
		expectations = abs(WFRotated.conj()*WFRotated);
		expectDiff = expectations - measProbs.reshape([self.nComp, 1]);
		cost = cost + sum(abs(expectDiff));
		return np.asscalar(cost);

	def findSumDistance(self, WF, nPOVMSets, fullMeasProbs, fullPOVMs):
		distances = np.zeros(nPOVMSets);
		for i in range(nPOVMSets):
			distances[i] = self.getAbsSumCost(WF, fullMeasProbs[i], fullPOVMs[i]);
		su = sum(distances);
		return su;

	def getExactDistance(self, rho, rho2):
		diff = rho - rho2;
		evs2, eVs2 = np.linalg.eig(diff);
		realDist = 0.5*sum(abs((evs2)));
		return realDist;

	def fullStepCloser(self, fullMeasProbs, fullUnitaries, learningRate = 40, iterations = 1600, actualWF = 0, whichCost = "KL2", initFraction = 0.3):
		nBases = fullMeasProbs.shape[0];
		eValueMatrix = self.getEigenValueMatrix() #matrix of sigmas as rows
		gradStat = np.zeros([3,int(iterations/3)+1]);
		fraction = initFraction;

		for iters in range(iterations):
			cost = 0;
			WF = self.getFullWF(eValueMatrix);
			WFKet = WF.reshape([self.nComp, 1]);
			#The following are (1/Pi)*der(Pi) also called Si
			ampDerivatives = self.getAmpDerivativesMatrix(eValueMatrix); #[nComp * rNWeights] first index is amplitude
			phaseDerivatives = self.getPhaseDerivativesMatrix(eValueMatrix); #[nComp * iNWeights] first index is amplitude
			gradLambda = np.zeros(self.nWeights);
			gradMu = np.zeros(self.nWeights);

			PLambda = self.getBigAmplitude(eValueMatrix); #[nComp]
			Z = sum(PLambda);
			ZDerivativeTerm = np.dot(PLambda, ampDerivatives)/Z;  #[rNWeights]
			#play around with these parameters.
			if iters%3 == 0:
				#Take large factor for 3-divisible iterations so that the cost is accurate(3-div iters are plotted)
				factor = nBases;
				if iters > 200 and iters%180 == 0:
					#if fraction < 0.8:
						#fraction = fraction + 0.08;
					cost = 0;
					#learningRate = 0.95*learningRate;
			else:
				factor = int(fraction*nBases);
			randIndices = random.sample(range(nBases), int(factor));
			#New addition begins:
			indicesAll = np.arange(nBases);
			valuesAll = np.ones(nBases);
			RBMprobsAll = np.zeros([nBases, self.nComp]);
			for i in range(nBases):
				Unitary = fullUnitaries[i];
				WFRotated = np.dot(Unitary, WFKet);
				RBMprobsAll[i] = (abs(WFRotated.conj()*WFRotated)).reshape(self.nComp,);
			for i in range(nBases):
				for j in range(self.nComp):
					if RBMprobsAll[i,j] == 0:
						continue;
					frac = fullMeasProbs[i,j]/RBMprobsAll[i,j];
					if frac < valuesAll[i]:
						valuesAll[i] = frac;
			for i in range(nBases):
				for j in range(nBases - 1 - i):
					if valuesAll[j] > valuesAll[j + 1]:
						temp = valuesAll[j];
						valuesAll[j] = valuesAll[j+1];
						valuesAll[j+1] = temp;
						temp = indicesAll[j];
						indicesAll[j] = indicesAll[j+1];
						indicesAll[j+1] = temp;
			randIndices = indicesAll[0:int(factor)];
			#
			for p in randIndices:
				measProbs = fullMeasProbs[p].reshape([self.nComp, 1]);
				Unitary = fullUnitaries[p];
				WFRotated = np.dot(Unitary, WFKet);
				expectations = abs(WFRotated.conj()*WFRotated);
				expectations = expectations.reshape([self.nComp, 1]);
				if whichCost == "L2":
					expectDiff = expectations - measProbs;
					cost = cost + sum((expectDiff**2));
				elif whichCost == "KL2":
					logExpectations = np.log(expectations);
					logMeasProbs = np.log(measProbs);
					cost = cost + sum(expectations*(logExpectations - logMeasProbs));
					expectDiff = 1 + logExpectations - logMeasProbs;
				elif whichCost == "L1.5":
					expectDiff = np.sign(expectations - measProbs)*np.abs(expectations - measProbs)**(0.5);
					cost = cost + sum(np.abs(expectations - measProbs)**(1.5));
				zFactor = sum(expectDiff*expectations);

				for k in range(self.nWeights):
					scaledAmpDer = ampDerivatives[:,k];
					scalerAmp = WF*scaledAmpDer;
					scalerAmp = scalerAmp.reshape([self.nComp, 1]);
					scaledPhaDer = phaseDerivatives[:,k];
					scalerPha = WF*scaledPhaDer;
					scalerPha = scalerPha.reshape([self.nComp, 1]);

					realTerm = np.dot(Unitary, scalerAmp);
					realTerm = np.real(realTerm*WFRotated.conj());
					realTerm = expectDiff*realTerm;
					gradLambda[k] = gradLambda[k] + sum(realTerm);

					imagTerm = np.dot(Unitary, scalerPha);
					imagTerm = -np.imag(imagTerm*WFRotated.conj());
					imagTerm = expectDiff*imagTerm;
					gradMu[k] = gradMu[k] + sum(imagTerm);

					gradLambda[k] = gradLambda[k] - zFactor*ZDerivativeTerm[k];

			factor = factor*fullMeasProbs.shape[1];
			deltaLambda = (learningRate/factor)*gradLambda;
			deltaLambda = np.insert(deltaLambda, 0, 0); #trash value to keep indices matched
			deltaMu = (learningRate/factor)*gradMu;
			deltaMu = np.insert(deltaMu, 0, 0);

			if iters%3 == 0:
				cost = np.asscalar(cost);
				o = np.abs(np.dot(WF.conj(), actualWF))**2;
				print("iter: "+str(iters)+" cost: "+str(cost) + " overlap:" +str(o));
				gradStat[0, int(iters/3)] = cost;
				gradStat[1, int(iters/3)] = o;
				gradStat[2, int(iters/3)] = valuesAll[0];

			self.rWeights = self.rWeights - deltaLambda.reshape(self.rWeights.shape);
			self.iWeights = self.iWeights - deltaMu.reshape(self.iWeights.shape);
		return gradStat;

	def tanh(self, x):
		return np.sinh(x)/np.cosh(x);

	def _logistic(self, x):
		return 1.0 / (1 + np.exp(-2*x));

	def getGibbsStat(self, rORi = 'r', nSamples = 5000, matSize = 500):
		#matSize is the number of parallel gibbsSampling chains you are running.
		#So small matSize means faster run but more correlation (because sampling every skip^th entry of every chain to collect nSamples)
		if not nSamples%matSize == 0:
			print("please give nSamples a multiple of matSize(or 500) in this implementation");
			return;
		samps = 2*np.random.rand(matSize, self.nVisible) - 1;
		samps = np.insert(samps, 0, 1, axis = 1); #insert bias states of 1 into the first column
		throw = 25;
		skip = 2;
		run = int(nSamples/matSize);
		ultaBina = lambda x : int("".join([str(int(i)) for i in x]) , 2);
		#converts the given nVisible size list of output nodes to a number
		for i in range(throw):
			if rORi == 'r':
				hids = self.generateRHidden(samps, matSize);
				samps = self.generateRVisible(hids, matSize);
			elif rORi == 'i':
				hids = self.generateIHidden(samps, matSize);
				samps = self.generateIVisible(hids, matSize);
		count = np.zeros(2**self.nVisible);
		for i in range(int(run)):
			states = -(0.5*(samps[:,1:]-1)); #get rid of first entry which is bias 1
			#then convert 1 to 0 and -1 to 1 as per convention followed in getCompleteState function etc.
			for j in range(states.shape[0]):
				index = ultaBina(states[j]);
				count[index] = count[index] + 1;
			for j in range(skip):
				if rORi == 'r':
					hids = self.generateRHidden(samps, matSize);
					samps = self.generateRVisible(hids, matSize);
				elif rORi == 'i':
					hids = self.generateIHidden(samps, matSize);
					samps = self.generateIVisible(hids, matSize);
		return np.asarray(count);

	def generateRHidden(self, visible, num_examples):
		hidden_activations = np.dot(visible, self.rWeights);
		hidden_probs = self._logistic(hidden_activations);
		hidden_probs[:,0] = 1; # Fix the bias unit.
		hidden_states = hidden_probs > np.random.rand(num_examples, self.nHidden + 1);
		hidden_states = 2*hidden_states - 1;#TRANSFORM ZEROES TO -1'S AND 1'S TO 1'S.
		return hidden_states;

	def generateRVisible(self, hidden, num_examples):
		visible_activations = np.dot(hidden, self.rWeights.T);
		visible_probs = self._logistic(visible_activations);
		visible_probs[:,0] = 1; # Fix the bias unit.
		visible_states = visible_probs > np.random.rand(num_examples, self.nVisible + 1);
		visible_states = 2*visible_states - 1; #TRANSFORM ZEROES TO -1'S AND 1'S TO 1'S.
		return visible_states;

	def generateIHidden(self, visible, num_examples):
		hidden_activations = np.dot(visible, self.iWeights);
		hidden_probs = self._logistic(hidden_activations);
		hidden_probs[:,0] = 1; # Fix the bias unit.
		hidden_states = hidden_probs > np.random.rand(num_examples, self.nHidden + 1);
		hidden_states = 2*hidden_states - 1;#TRANSFORM ZEROES TO -1'S AND 1'S TO 1'S.
		return hidden_states;

	def generateIVisible(self, hidden, num_examples):
		visible_activations = np.dot(hidden, self.iWeights.T);
		visible_probs = self._logistic(visible_activations);
		visible_probs[:,0] = 1; # Fix the bias unit.
		visible_states = visible_probs > np.random.rand(num_examples, self.nVisible + 1);
		visible_states = 2*visible_states - 1; #TRANSFORM ZEROES TO -1'S AND 1'S TO 1'S.
		return visible_states;