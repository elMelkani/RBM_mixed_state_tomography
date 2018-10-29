#Helper functions etc for main file. to read/generate rho or to generate measurements etc.
import numpy as np;
from scipy import linalg;
import random;

class tomoHelper:
	def __init__(self, nVisible, nHidden):
		self.nVisible = nVisible;
		self.nHidden = nHidden;
		self.nComp = 2**self.nVisible;
		self.rho = [];

	def str_base(self, number, base):
		for i in range(self.nVisible):
			number, digit = divmod(number, base)
			yield digit;

	def readRho(self, nVisible):
		mat = np.zeros([2**nVisible,2**nVisible]);
		lines = [line.rstrip('\n') for line in open('real'+str(nVisible)+'.txt')];
		count = 0;
		for i in range(2**nVisible):
			j = 0;
			count = 0;
			while j < len(lines[i]):
				if lines[i][j] == ' ':
					while j < len(lines[i]) and lines[i][j] == ' ':
						j=j+1;
				if j < len(lines[i]) and not lines[i][j] == ' ':
					if lines[i][j:].find(' ') == -1:
						ind2 = len(lines[i]);
					else:
						ind2 = j+lines[i][j:].find(' ');
					mat[i,count] = float(lines[i][j:ind2]);
					count = count + 1;
					j = ind2;
		mat2 = np.zeros([2**nVisible,2**nVisible]);
		lines = [line.rstrip('\n') for line in open('imag'+str(nVisible)+'.txt')];
		count = 0;
		for i in range(2**nVisible):
			j = 0;
			count = 0;
			while j < len(lines[i]):
				if lines[i][j] == ' ':
					while j < len(lines[i]) and lines[i][j] == ' ':
						j=j+1;
				if j < len(lines[i]) and not lines[i][j] == ' ':
					if lines[i][j:].find(' ') == -1:
						ind2 = len(lines[i]);
					else:
						ind2 = j+lines[i][j:].find(' ');
					mat2[i,count] = float(lines[i][j:ind2]);
					count = count + 1;
					j = ind2;
		rho = mat + 1j*mat2
		self.rho = rho;
		return rho;

	def generateRho(self, actualState, nVisible):
		nComp = 2**nVisible;
		if actualState == "notSoRandom":
			H = np.zeros([nComp, nComp], dtype = complex);
			mat = np.random.rand(nComp, nComp) + 1j*np.random.rand(nComp, nComp);
			H = H + mat + mat.T.conj();
			U = linalg.expm(1j*H);

			d = np.ones(nComp, dtype = complex);
			d = (1 - 0.78 - 0.15)/(nComp - 2)*d;
			d[0] = 0.78;
			d[2] = 0.15;
			rho = np.dot(np.dot(U, np.diag(d)), U.T.conj());
		elif actualState == "randomDM":
			H = np.zeros([nComp, nComp], dtype = complex);
			mat = np.random.rand(nComp, nComp) + 1j*np.random.rand(nComp, nComp);
			H = H + mat + mat.T.conj();
			rho = linalg.expm(0.1*H);
			rho = rho/np.trace(rho);
		elif actualState == "mixedRotaBell":
			p1 = 0.6; p2 = 0.2; p3 = 0.2;
			bell1 = np.asarray([1,0,0,1j])/np.sqrt(2); #Bell = 1/norm * |00> + |11>
			rhoBell1 = np.dot(bell1.reshape([4,1]), bell1.reshape([4,1]).T.conj()); #outer product
			bell2 = np.asarray([0,1j,1,0])/np.sqrt(2); #Bell = 1/norm * |01> + |10>
			rhoBell2 = np.dot(bell2.reshape([4,1]), bell2.reshape([4,1]).T.conj()); #outer product
			rho = p1*rhoBell1 + p2*rhoBell2 +p3*0.25*np.eye(4);
		elif actualState == "mixedBell":
			p1 = 0.65; p2 = 0.3; p3 = 0.05;
			bell1 = np.asarray([1,0,0,1])/np.sqrt(2); #Bell = 1/norm * |00> + |11>
			rhoBell1 = np.dot(bell1.reshape([4,1]), bell1.reshape([4,1]).T.conj()); #outer product
			bell2 = np.asarray([0,1,1,0])/np.sqrt(2); #Bell = 1/norm * |01> + |10>
			rhoBell2 = np.dot(bell2.reshape([4,1]), bell2.reshape([4,1]).T.conj()); #outer product
			rho = p1*rhoBell1 + p2*rhoBell2 +p3*0.25*np.eye(4);
		self.rho = rho;
		return rho;

	def makePOVMs(self, nVisible, nTMeasurements, nMeasurements):
		nComp = 2**nVisible;

		#get #basic stuff ready:
		#Unitary for getting from Z-basis to X-basis
		UX = np.ones([2,2], dtype = complex);
		UX[1,1] = -1;
		UX = (1.0/np.sqrt(2))*UX; # now as X-basis is UX.zBasis and zBasis is Identity this is also the xBasis

		#Unitary for getting from Z-basis to Y-basis
		UY = np.ones([2,2], dtype = complex);
		UY[1,1] = -1j; UY[1,0] = 1j;
		UY = (1.0/np.sqrt(2))*UY;

		U = [UX, UY, np.eye(2)];

		POVMs = np.zeros([nMeasurements,nComp,nComp,nComp], dtype = complex);#as many POVM sets as there are measurement bases
		zProjections = np.zeros([nComp,nComp,nComp], dtype = complex);#as many projections as there are measurement bases
		for i in range(nComp):
			zProjections[i,i,i] = 1;

		randIndices = random.sample(range(nTMeasurements), nMeasurements);
		count = 0;
		for i in randIndices:
			liz = list(self.str_base(i, 3));
			Unitary = 1;
			for j in liz:
				Unitary = np.kron(Unitary, U[j]);
			#Unitaries[i] = Unitary;
			for k in range(nComp):
				mat = np.dot(Unitary, zProjections[k]); #projection matrix M_k
				POVMs[count, k] = np.dot(mat, mat.T.conj()); #POVM matrix M_k.M_k\dagger
			count = count + 1;
		return POVMs;

	def getPureProbStat(self, POVMs, WF):
		nComp = WF.shape[0];
		probs = np.zeros([POVMs.shape[0], POVMs.shape[1]]);
		WFKet = WF.reshape([nComp, 1]);
		WFBra = WF.reshape([1, nComp]).conj();
		for j in range(POVMs.shape[0]):
			for k in range(POVMs.shape[1]):
				probs[j,k] = np.asscalar(abs((np.dot(np.dot(WFBra, POVMs[j,k]), WFKet))));
		return probs;

	def makeUnitaries(self, nVisible, nTMeasurements, nMeasurements, rho): #also makes corresponding measurements on rho
		nComp = 2**nVisible;
		#get #basic stuff ready:
		#Unitary for getting from Z-basis to X-basis
		UX = np.ones([2,2], dtype = complex);
		UX[1,1] = -1;
		UX = (1.0/np.sqrt(2))*UX; # now as X-basis is UX.zBasis and zBasis is Identity this is also the xBasis

		#Unitary for getting from Z-basis to Y-basis
		UY = np.ones([2,2], dtype = complex);
		UY[1,1] = -1j; UY[1,0] = 1j;
		UY = (1.0/np.sqrt(2))*UY;

		U = [UX, UY, np.eye(2)];

		#POVMs = np.zeros([nMeasurements,nComp,nComp,nComp], dtype = complex);#as many POVM sets as there are measurement bases
		Unitaries = np.zeros([nMeasurements,nComp,nComp], dtype = complex);#as many Unitaries sets as there are measurement bases
		zProjections = np.zeros([nComp,nComp,nComp], dtype = complex);#as many projections as there are measurement bases
		measProbs = np.zeros([nMeasurements, nComp]);
		for i in range(nComp):
			zProjections[i,i,i] = 1;

		randIndices = random.sample(range(nTMeasurements), nMeasurements);
		count = 0;
		for i in randIndices:
			liz = list(self.str_base(i, 3));
			Unitary = 1;
			for j in liz:
				Unitary = np.kron(Unitary, U[j]);
			Unitaries[count] = Unitary.T.conj();
			for k in range(nComp):
				mat = np.dot(Unitary, zProjections[k]); #projection matrix M_k
				mat = np.dot(mat, mat.T.conj()); #POVM matrix M_k.M_k\dagger
				measProbs[count,k] = np.real(np.trace(np.dot(mat,rho)));
			count = count + 1;
		return [Unitaries, measProbs];
