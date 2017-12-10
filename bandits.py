# Zack Barnett-Howell
# April 20, 2017
# Bandit holds objects with different reward distributions
# Strategy holds strategies with different algorithms

import numpy as np  # For basic math operations


class Bandit(object):
    """
Construct a bandit with K different arms

 parameters:
    k: number of arms to initialize. Takes integers > 1
    shock: types of random shocks to the reward vector
methods:
    pull:generates reward from bandit i in K
"""

    def __init__(self, arms=1, shock=False):
        self.k = arms  # Number of bandits
        self._shock = shock

    def __len__(self):
        return self.k

    def pull(self, i):
        return self._reward[i]


class normalbandit(Bandit):
    """
    Bandit class with rewards drawn from the normal distribution

parameters:
    k: number of bandits to initialize. Takes integers > 1
    mu range: range in which means are drawn: min to max
    var range: range in which variance is drawn: 0 to max
methods:
    pull:generates reward from bandit i in k
    """

    def __init__(self, arms=1, shock=False, murange=(-1, 1), varmax=(1)):
        Bandit.__init__(self, arms, shock)
        self._armsmu = np.random.uniform(min(murange), max(murange), self.k)
        self._armsvar = np.random.uniform(0, varmax, self.k)
        self._bestarm = np.argmax(self._armsmu)

    def pull(self, i):
        return np.random.normal(self._armsmu[i], self._armsvar[i])


class poissonbandit(Bandit):
    """
    Bandit class with rewards drawn from the poisson distribution

parameters:
    k: number of bandits to initialize. Takes integers > 1
    mu range: range in which means are drawn: min to max
    var range: range in which variance is drawn: 0 to max
methods:
    pull:generates reward from bandit i in k
    """

    def __init__(self, arms=1, shock=False, murange=(0, 1)):
        Bandit.__init__(self, arms, shock)
        self._mu = np.random.uniform(min(murange), max(murange), self.k)
        self._bestarm = np.argmax(self._mu)

    def pull(self, i):
        return np.random.poisson(self._mu[i])


class Strategy(object):

    """
General strategy object

 parameters:
    turns: number of turns the strategy will play for. Takes integers > 1
    history: initialize empty vector to store bandit arm choices for each turn
    rewards: initialize an empty vector ot store reward for each turn
"""

    def __init__(self, bandit, turns=10):
        self._turns = turns
        self._bandit = bandit
        self._choice = [0 for i in range(self._bandit.k)]
        self._cumreward = [0 for i in range(self._bandit.k)]
        self._avgreward = [0 for i in range(self._bandit.k)]
        self._sequence = []
        self._totalreward = 0
        self._totalregret = 0

    def totalreward(self):
        return sum(self._reward)

    def totalregret(self):
    	return  sum(self._reward)- max(self._bandit._armsmu) * self._turns


class epsilongreedy(Strategy):
    """
Epsilon-Greedy strategy

 parameters:
    epsilon: probability of choosing a random arm. Takes values in (0,1)
    annealing: whether epsilon decreases over time. Takes boolean
    bandit: which bandit object it will play 
"""

    def __init__(self, bandit, turns=10, epsilon=.01, annealing=True):
        Strategy.__init__(self, bandit, turns)
        self._epsilon = epsilon
        self._annealing = annealing

    def initialize(self):
        self._avgreward = [self._bandit.pull(i) for i in range(0, len(self._bandit))]

    def anneal(self):
        self._epsilon *= .95  # Set rate at which epsilon decreases

    def run(self):
        for t in range(1, self._turns):
            if np.random.uniform() > self._epsilon:
                arm = self._avgreward.index(max(self._avgreward))  # Choose the arm with the highest history
                self._choice[arm] += 1
                self._sequence.append(arm)
                reward = self._bandit.pull(arm)  # Generate a rewad from that arm
                self._cumreward[arm] += reward
                self._avgreward[arm] = self._cumreward[arm] / self._choice[arm]
            else:
                arm = np.random.random_integers(0, len(self._bandit) - 1)  # Generate a random arm
                self._choice[arm] += 1
                self._sequence.append(arm)
                reward = self._bandit.pull(arm)
                self._cumreward[arm] += reward
                self._avgreward[arm] = self._cumreward[arm] / self._choice[arm]
            if self._annealing:
                y.anneal()
        self._totalregret = sum(self._cumreward) - max(self._bandit._mu) * self._turns


x=normalbandit(arms=2,murange=(0,1),varmax=(1))
j = poissonbandit(arms=5,murange=(2,10))

y=epsilongreedy(bandit=j, turns=100, epsilon=.1, annealing=True)
#z = ucb1(bandit=j, turns = 50)
#z.initialize()
#z.run()

y.initialize()
y.run()

print("Bandit arms averages are:")
print(y._bandit._mu)
print("Bandit Chose:")
print(y._choice)
print("Bandit received average reward:")
print(y._avgreward)
print("Bandit total regret:")
print(y._totalregret)
