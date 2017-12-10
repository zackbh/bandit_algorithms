import numpy as np

class ucbtuned(Strategy):
    """
UCB-Tuned strategy

 parameters:
    bandit: which bandit object it will play
    turns: how long the strategy will run
"""

    def __init__(self, bandit, turns=10):
        Strategy.__init__(self, bandit, turns)
        self._P = [0 for i in range(self._bandit.k)]
        self._rewardmatrix = np.zeros([1, self._bandit.k], dtype=np.float64)

    def initialize(self):
        self._rewardmatrix[0,] = [self._bandit.pull(i) for i in range(0, len(self._bandit))]
        self._choice = [1 for i in range(self._bandit.k)]

    def run(self):
        for t in range(1, self._turns):
            # Choose the arm with the highest history
            # print("Current reward matrix {} ".format(self._rewardmatrix))
            arm = np.argmax(np.add(np.mean(self._rewardmatrix, axis=0, dtype=np.float64) , self._P))
            self._choice[arm] += 1
            self._sequence.append(arm)
            reward = np.zeros([1,self._bandit.k])
            reward[0,arm] = self._bandit.pull(arm)  # Generate a rewad from that arm
            self._rewardmatrix = np.vstack((self._rewardmatrix,reward))
            V = np.var(self._rewardmatrix, axis=0) + np.sqrt((2 * np.log(t)) / self._choice )
            # print("Value of V {} ".format(V))
            self._P = np.sqrt((np.log(t) / (self._choice)) * V)
            # print("Value of P {} ".format(self._P))
        self._totalregret = sum(self._cumreward) - \
            max(self._bandit._mu) * self._turns

z = ucbtuned(bandit=j, turns = 500)
z.initialize()
z.run()

print("Bandit arms averages are:")
print(z._bandit._mu)
print("Bandit Chose:")
print(z._choice)
print("Bandit received average reward:")
print(np.mean(z._rewardmatrix, axis=0))
print("Bandit total regret:")
print(z._totalregret)

