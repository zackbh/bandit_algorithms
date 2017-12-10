import numpy as np

class aucb(Strategy):


    def __init__(self, bandit, turns=10, trend = 'sigmoid'):
       Strategy.__init__(self, bandit, turns)
       self._trend = trend
       if (self._trend == "sigmoid"):
            self._D = [.037 * np.exp(1.15) for i in range(self._bandit.k)] 
       if (self._trend == "decreasing"):
        	self._D = [9.57 for i in range(self._bandit.k)]
       if self._trend == "gaussian":
            self._D = [np.exp(-(-20)^2/40) for i in range(self._bandit.k)]
       self._P = [0 for i in range(self._bandit.k)]

    def initialize(self):
    	self._avgreward = [0
                           for i in range(0, len(self._bandit))]
        self._choice = [1 for i in range(self._bandit.k)]

    def run(self):
    	for t in range(1, self._turns):
        # Choose the arm with the highest history
            arm = np.argmax(np.add(self._avgreward, self._P) * self._D)
            self._choice[arm] += 1
            self._sequence.append(arm)
            reward = self._bandit.pull(arm)  # Generate a rewad from that arm
            self._cumreward[arm] += reward
            self._avgreward[arm] = self._cumreward[arm] / self._choice[arm]
            self._P = np.sqrt((2 * np.log(t)) / (self._choice))
            if self._trend == "sigmoid":
            	self._D[arm] = .037 * np.exp(1.15 * self._choice[arm]) 
        #print(reward)
        self._totalregret = sum(self._cumreward) - max(self._bandit._mu) * self._turns

class trendbandit(Bandit):

    def __init__(self, arms=1, shock=False, murange=(0, 1), trend = "sigmoid"):
        Bandit.__init__(self, arms, shock)
        self._mu = np.random.uniform(min(murange), max(murange), self.k)
        self._trend = trend
        self._n = [0 for i in range(self.k)]
        if (self._trend == "sigmoid"):
            self._D = [.037 * np.exp(1.15) for i in range(self.k)]
        if (self._trend == "decreasing"):
            self._D = [9.57 for i in range(self.k)]
        self._bestarm = np.argmax(self._mu)

    def pull(self, i):
        self._n[i] += 1
        if (self._trend == "sigmoid"):
            self._D[i] = .037 * np.exp(1.15 * self._n[i])  # Increases the trend for arm i
        if (self._trend == "decreasing"):
        	self._D[i] = -6.65*np.log(self._n[i]) + 9.57
        return np.random.poisson(self._mu[i]) * self._D[i]  # Returns random value * trend


j = trendbandit(arms=5, murange=(2,10), trend = "decreasing")
y = aucb(bandit = j, turns = 20, trend = 'decreasing')
y.run()

print("Bandit arms averages are:")
print(y._bandit._mu)
print("Bandit Chose:")
print(y._choice)


