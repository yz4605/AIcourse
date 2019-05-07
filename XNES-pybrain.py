from scipy.linalg import expm
from scipy import dot, array, randn, eye, outer, exp, trace, floor, log, sqrt, ones
from pybrain.optimization import XNES

class DyXnes(XNES):
    """ NES with exponential parameter representation. """
    def __init__(self, batchSize=2, maxEvaluation=10, batchSizeIncrease=0):
        self.batchSize = batchSize
        self.maxEvaluations  = maxEvaluation
        self.batchSizeIncrease = batchSizeIncrease
        input_dim = len(config["curDict"])
        outputDim = config["outputDim"]
        lenWeights = (input_dim+1+outputDim)*outputDim
        super().__init__(fitness, -1*ones(lenWeights))
    
    #Here is the main part of the modify of the
    def _learnStep(self):
        """ Main part of the algorithm. """    
        #initialize before learn
        print('==========================Train start')
        preLearn()
        input_dim = len(config["curDict"])
        outputDim = config["outputDim"]
        lenWeights = input_dim*outputDim + outputDim*outputDim + outputDim # Length of all weight parameters that need to be train

        #XNES train
        eta = 0.0001 # eta is an arbitrarily small real number 
        orgLenA = len(self._A) # original number of columns/rows of matrix A
        addNum = lenWeights - orgLenA
        keepLen = outputDim + outputDim*outputDim
        orgwLen = input_dim * outputDim

        self._modifyMatrix(True, lenWeights, keepLen, eta)
        self._modifyMatrix(False, lenWeights, keepLen, eta)

        # Update self.center
        #print(108)
        centerBias = self._center[-keepLen:]
        for i in range(keepLen):
            self._center = np.delete(self._center,-1,axis=0)       
        for i in range(addNum):
            self._center = np.append(self._center,0)
        for i in range(keepLen):
            self._center = np.append(self._center,centerBias[i])
        # Upadate self.numParameters
        #self.numParameters = lenWeights

        #print("116"

        # Upadate self.numParameters
        self.numParameters = lenWeights
        I = eye(self.numParameters)
        #print("134")
        
        self._produceSamples()
        
        #print("144")
        #Update for All evaluated
        utilities = self.shapingFunction(self._currentEvaluations)
        utilities /= sum(utilities)  # make the utilities sum to 1
        if self.uniformBaseline:
            utilities -= 1./self.batchSize
        samples = array(list(map(self._base2sample, self._population)))
        #print(172)
        dCenter = dot(samples.T, utilities)
        covGradient = dot(array([outer(s,s) - I for s in samples]).T, utilities)
        covTrace = trace(covGradient)
        covGradient -= covTrace/self.numParameters * I
        dA = 0.5 * (self.scaleLearningRate * covTrace/self.numParameters * I
                    +self.covLearningRate * covGradient)
        self._lastLogDetA = self._logDetA
        self._lastInvA = self._invA
        #print(181)
        self._center += self.centerLearningRate * dot(self._A, dCenter)
        self._A = dot(self._A, expm(dA))
        self._invA = dot(expm(-dA), self._invA)
        self._logDetA += 0.5 * self.scaleLearningRate * covTrace
        if self.storeAllDistributions:
            self._allDistributions.append((self._center.copy(), self._A.copy()))
        
        postLearn()
        self.batchSize += self.batchSizeIncrease
        print("Done")
    
    def _modifyMatrix(self, isForA ,newWeightSizeghts ,biasSize, eta):
        if isForA:
            matrix = self._A
        else:
            matrix = self._invA
        
        orgLenA = len(matrix)
        orgInputSize = orgLenA - biasSize
        addNum = newWeightSizeghts - orgLenA
        if addNum <= 0:
            if isForA:
                self._A = matrix
            else:
                self._invA = matrix
            return        
        if orgInputSize == 0:
            Diag = eta*eye(newWeightSizeghts)
            Diag[addNum:, addNum:] = matrix
            matrix = Diag
        else:
            W = matrix[:orgInputSize, :orgInputSize]
            BC = matrix[:orgInputSize, orgInputSize:]
            BR = matrix[orgInputSize:, :orgInputSize]
            Diag = matrix[orgInputSize:, orgInputSize:]
            BC = np.hstack((np.zeros((orgInputSize,addNum)), BC))
            BR = np.vstack((np.zeros((addNum, orgInputSize)),BR))
            newDiag = eta*np.eye(addNum+biasSize)
            newDiag[addNum:, addNum:] = Diag
            Diag = newDiag
            AR = np.vstack((BC, Diag))
            AL = np.vstack((W, BR))
            matrix = np.hstack((AL, AR))
        if isForA:
            self._A = matrix
        else:
            self._invA = matrix

if __name__ == '__main__':
    runTrain()
    l = DyXnes(5, 100)
    res = l.learn()
    