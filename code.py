import gym
import gym_gvgai
import threading
import numpy as np

config = {}
trainSet = []
updateDict = []

class RNN():
    def __init__(self, w=None , u=None, b=None, input_dim=1, output_dim=1):
        self.output_dim = output_dim
        self.input_dim = input_dim
        self.w = np.random.random((input_dim,output_dim)) if w is None else w
        self.u = np.random.random((output_dim,output_dim)) if u is None else u
        self.b = np.random.random((output_dim,)) if b is None else b
        self.state_t = np.zeros((output_dim,))
        self.successive_outputs = []
        self.maxes = []
        self.end_flag = 0
        self.score = 0
    def sigmoid(self,x):
        s = 1 / (1 + np.exp(-x))
        return s
    def act(self, stateObs):
        if self.input_dim <= 1:
            return np.random.randint(0,self.output_dim)
        input_t = stateObs
        output_t = self.sigmoid(np.dot(self.w.T,input_t)+np.dot(self.u,self.state_t)+self.b)
        self.successive_outputs.append(output_t)
        self.state_t = output_t
        max_t = np.argmax(output_t, axis=0)
        self.maxes.append(max_t)
        action_id = max_t
        return action_id

def getNetwork(dist,lenCode,outputDim):
    if len(dist) == outputDim**2+outputDim*lenCode+outputDim:
        b = dist[-outputDim:]
        u = dist[-outputDim**2-outputDim : -outputDim]
        w = dist[:-outputDim**2-outputDim]
        b = np.array(b)
        u = np.reshape(u,(outputDim, outputDim))
        w = np.reshape(w,(lenCode, outputDim))
        r = RNN(w,u,b,lenCode,outputDim)
    else:
        r = RNN(input_dim=lenCode,output_dim=outputDim)
        print("Initialize Network Structure")
    return r

def downSample(stateObs):
    stateObs = stateObs[::2,::2]
    stateObs = np.sum(stateObs.astype("int32")**2,axis=2)
    stateObs = stateObs.flatten()
    return stateObs

def runGame(agent,result,curDict):
    global trainSet
    env = gym_gvgai.make(config["game"])
    env.reset()
    current_score = 0
    step = 1000
    pick = np.random.randint(step)
    for t in range(step):
        stateObs = env.render("rgb_array")
        stateObs = downSample(stateObs)
        code = DRSC(stateObs,curDict)
        action_id = agent.act(code)
        _, increScore, done, debug = env.step(action_id)
        current_score += increScore
        if done:
            trainSet.append(stateObs)
            print("Game tick " + str(t+1) + ", increScore: " + str(increScore) + " player " + debug['winner'] + ", total score: " + str(current_score))
            break
        if t == pick:
            trainSet.append(stateObs)
    result.append(current_score)
    env.close()
    return

def fitness(dist=[]):
    threadNum = config["threadNum"]
    outputDim = config["outputDim"]
    curDict = updateDict
    lenCode = len(curDict)
    r = getNetwork(dist,lenCode,outputDim)
    tlist,result = [],[]
    for x in range(threadNum):
        t = threading.Thread(target=runGame, args=(r,result,curDict))
        tlist.append(t)
        t.start()
    for x in tlist:
        x.join()
    result.sort()
    if threadNum > 2:
        score = sum(result[1:-1])
    else:
        score = sum(result)
    return score

def DRSC(x, curDict, epsilon=255):
    p,w,code = x,0,[0]*len(curDict)
    if code == []:
        return code
    omega = max(1,len(curDict)/2)
    while np.sum(p) > epsilon and w < omega:
        delta = np.array([])
        for d in curDict:
            eq = p==d
            sim = eq.sum()
            delta = np.append(delta,sim)
        msc = np.argmax(delta)
        code[msc] = 1
        w = w+1
        p = p-curDict[msc]
        p = p.clip(min=0)
    return code

def IDVQ(trainSet, delta=255):
    global updateDict
    curDict = updateDict[:]
    for x in trainSet:
        p = x
        c = DRSC(x, curDict)
        p_new = np.dot(c,curDict)
        R = p-p_new
        R = R.clip(min=0)
        if np.sum(R) > delta:
            curDict.append(R)
    updateDict = curDict
    return

def trainDict():
    global trainSet
    localSet = trainSet[:]
    trainSet = []
    IDVQ(localSet)

def initConfig(game):
    global config
    config = {"game":game,"threadNum":4}
    env = gym_gvgai.make(game)
    config["outputDim"] = len(env.env.GVGAI.actions())
    env.reset()
    stateObs = env.render("rgb_array")
    #print("row length: ",len(stateObs))
    stateObs = downSample(stateObs)
    updateDict.append(stateObs)

def runTrain(game,batch=10):
    initConfig(game)
    for i in range(batch):
        fitness()
    trainDict()
    #result = [np.sum(i) for i in updateDict]
    #print("Dictionary: ",result)

def runTest(game,dist):
    config["game"] = game
    env = gym_gvgai.make(game)
    env.reset()
    stateObs = env.render("rgb_array")
    stateObs = downSample(stateObs)
    updateDict[0]=stateObs
    score = fitness(dist)
    return score

def computeUtilities(fitnesses):
    L = len(fitnesses)
    ranks = rankdata(fitnesses,method='ordinal')-1
    utilities = np.array([max(0., x) for x in log(L / 2. + 1.0) - log(L - np.array(ranks))])
    utilities /= sum(utilities)
    utilities -= 1. / L
    return utilities

if __name__ == "__main__":
    runTrain("gvgai-testgame1-lvl0-v0",1)
    runTest("gvgai-testgame1-lvl1-v0",[])
    
'''
from matplotlib import pyplot as plt
for i in code.updateDict:
    p = i.reshape((45,-1)).astype(int)
    plt.figure()
    plt.imshow(p)
    plt.savefig(str(i.sum())+".png")
plt.show()
'''


