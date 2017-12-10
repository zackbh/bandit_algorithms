import numpy as np
import scipy


class gblockucl(Strategy):
    def __init___(self, bandit, turns=10, mu, sigma):
        Strategy.__init__(self, bandit, turns)
        self._mu = mu  # Kx1
        self._sigma = sigma  # Kx1
        self._prior = numpy.random.multivariate_normal(self._mu, self._sigma * np.identity(self._bandit.k))
        self._mu0 = mu0
        self._sigma0 = np.sqrt(self._sigma)
        self._K = np.sqrt(2 * np.pi * np.exp(1))
        self._l = np.ceil(np.log2(self._turns))
        self._n = np.zeros(self._k)
        self._mbar = np.zeros(self._k)
        self._delta = self._sigma / self._sigma0
        self._i = np.zeros(self._k)
        self._mit = np.zeros(self._turns)
        self._tau = 0
        self._itau = 1

    def initialize(self):
        self._avgreward = [0
                           for i in range(0, len(self._bandit))]
        self._choice = [0 for i in range(self._bandit.k)]

    def run(self):
        for k in range(1, self._l):
            bk = np.ceil(2^(k - 1) / k)
            for r in range(1, bk):
                self._tau = 2^(k - 1) + (r - 1) * k
                mut = (self._delta^2 * self._mu0 + self._n * self._mbar) / \
                    (self._delta^2 + self._n)
                sigmat = self._sigma / np.sqrt(self._delta^2 + self._n)
                if self._tau == 0:
                    Q = mut + sigmat * scipy.stats.invnorm.cdf(1 - 1 / K)
                else:
                    Q = mut + sigmat * \
                        scipy.stats.invnorm.cdf(1 - 1 / (K * self._tau))
                ihat = np.argmax(Q)
                if 2^k - self._tau >= k:
                    reward = np.zeros(k)
                    for t in range(self._tau, self._tau + k):
                        self._i[t] = ihat
                    for t in range(1, k):
                        reward[t] = self._bandit.pull(ihat)
                    self._mbar[ihat] = (self._n[ihat] * self._mbar[ihat] + np.sum(reward)) / (self._n[ihat] + k)
                    self._n[ihat] += k
                else:
                    reward = np.zeros(2^k - self._tau)
                    for t in range(self._tau, 2 ^ k - 1):
                        self._i[t] = ihat
                    for t in range(1, 2 ^ k - self._tau):
                        reward[t] = self._bandit.pull(ihat)
                    self._mbar[ihat] = (self._n[ihat] * self._mbar[ihat] + \
                        np.sum(reward)) / (self._n[ihat] + 2 ^ k - self._tau)
                    self._n += 2 ^ k - self._tau
