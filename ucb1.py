import numpy as np

class ucb1(Strategy):
    """
UCB1 strategy

 parameters:
    bandit: which bandit object it will play
    turns: how long the strategy will run
"""

    def __init__(self, bandit, turns=10):
        Strategy.__init__(self, bandit, turns)
        self._P = [0 for i in range(self._bandit.k)]

    def initialize(self):
        self._avgreward = [self._bandit.pull(i)
                           for i in range(0, len(self._bandit))]
        self._choice = [1 for i in range(self._bandit.k)]

    def run(self):
        for t in range(1, self._turns):
            # Choose the arm with the highest history
            arm = np.argmax(np.add(self._avgreward, self._P))
            self._choice[arm] += 1
            self._sequence.append(arm)
            reward = self._bandit.pull(arm)  # Generate a rewad from that arm
            self._cumreward[arm] += reward
            self._avgreward[arm] = self._cumreward[arm] / self._choice[arm]
            self._P = np.sqrt((2 * np.log(t)) / (self._choice))
            print(self._P)
        self._totalregret = sum(self._cumreward) - \
            max(self._bandit._mu) * self._turns

z = ucb1(bandit=j, turns = 50)
z.initialize()
z.run()

print("Bandit arms averages are:")
print(z._bandit._mu)
print("Bandit Chose:")
print(z._choice)
print("Bandit received average reward:")
print(z._avgreward)
print("Bandit total regret:")
print(z._totalregret)
