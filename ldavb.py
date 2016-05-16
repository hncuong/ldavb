import re
import sys
import corpus

import numpy as n
from scipy.special import gammaln, psi

n.random.seed(100000001)
meanchangethresh = 0.001
var_maxiter = 100
em_maxiter = 100
em_thresh = 1e-4



def dirichlet_expectation(alpha):
    if len(alpha.shape) == 1:
        return psi(alpha) - psi(n.sum(alpha))
    return psi(alpha) - psi(n.sum(alpha, 1))[:, n.newaxis]


class LDAvb:

    def __init__(self, vocab, K, D, alpha, eta):
        self._vocab = dict()
        for word in vocab:
            word = word.lower()
            word = re.sub(r'[^a-z]', '', word)
            self._vocab[word] = len(self._vocab)
        self._K = K
        self._D = D
        self._W = len(self._vocab)
        self._alpha = alpha
        self._eta = eta
        self._updatect = 0

        self._lambda = 1*n.random.gamma(100.0, 1.0/100.0, (self._K, self._W))
        self._Elogbeta = dirichlet_expectation(self._lambda)
        self._expElogbeta = n.exp(self._Elogbeta)

    def do_e_step(self, wordids, wordcts):
        batchD = len(wordids)

        gamma = n.random.gamma(100.0, 1.0/100.0, (batchD, self._K))
        Elogtheta = dirichlet_expectation(gamma)
        expElogtheta = n.exp(Elogtheta)

        sstats = n.zeros(self._lambda.shape)

        for d in range(0, batchD) :
            ids = wordids[d]
            cts = wordcts[d]

            gammad = gamma[d, :]
            Elogthetad = Elogtheta[d, :]
            expElogthetad = expElogtheta[d, :]
            expElogbetad = self._expElogbeta[:, ids]

            phinorm = n.dot(expElogthetad, expElogbetad) + 1e-100

            for it in range(0, var_maxiter):
                lastgamma = gammad

                gammad = self._alpha + expElogthetad*\
                    n.dot(cts/phinorm, expElogbetad.T)

                Elogthetad = dirichlet_expectation(gammad)
                expElogthetad = n.exp(Elogthetad)
                phinorm = n.dot(expElogthetad, expElogbetad) + 1e-100

                meanchange = n.mean(abs(lastgamma - gammad))
                if meanchange < meanchangethresh :
                    break

                gamma[d, :] = gammad
            sstats[:, ids] += n.outer(expElogthetad.T, cts/phinorm)

        sstats = sstats * self._expElogbeta
        return  ((gamma, sstats))
    def update_lambda(self, wordids, wordcts):

        (gamma, sstats) = self.do_e_step(wordids, wordcts)

        bound = self.approx_bound(wordids, wordcts, gamma)
        self._lambda = self._eta + self._D * sstats / len(wordids)
        self._Elogbeta = dirichlet_expectation(self._lambda)
        self._expElogbeta = n.exp(self._Elogbeta)
        self._updatect += 1

        return (gamma, bound)

    def approx_bound(self, wordids, wordcts, gamma):

        batchD = len(wordids)

        score = 0
        Elogtheta = dirichlet_expectation(gamma)
        expElogtheta = n.exp(Elogtheta)

        # E[log p(docs | theta, z, beta)] + E[log p(z | theta)] - E[log q(z)] ]
        for d in range(0, batchD):
            ids = wordids[d]
            cts = n.array(wordcts[d])
            phinorm = n.zeros(len(ids))
            phinorm = n.dot(expElogtheta[d, :], self._expElogbeta[:, ids])
            score += n.sum(cts * phinorm)

        # E[log p(theta | alpha) - log q(theta | gamma)]
        score += n.sum((self._alpha - gamma) * expElogtheta)
        score += n.sum(gammaln(gamma) - gammaln(self._alpha))
        score += n.sum(gammaln(self._K * self._alpha) - gammaln(n.sum(gamma, 1)))

        # sum(a , i) : the i-th axis run.; sun(a, (i,j)) run on i-th and j-th axis

        score = score * self._D / len(wordids)
        # E[log p(theta | alpha) - log q(theta | gamma)]
        score += n.sum((self._eta - self._lambda) * self._expElogbeta)
        score += n.sum(gammaln(self._lambda) - gammaln(self._eta))
        score += n.sum(gammaln(self._W * self._eta) - gammaln(n.sum(self._lambda, 1)))

        return score
    def run_em(self, wordids, wordcts):
        global var_maxiter
        old_bound = 0
        converged = 1
        for it in range(0, em_maxiter):
            print it
            (gamma, bound) = self.update_lambda(wordids, wordcts)
            print bound
            if it % 10 == 0:
                n.savetxt('lambda%d' % it, self._lambda)
                n.savetxt('gamma%d' %it, gamma.T)
            converged = (old_bound - bound) / (old_bound + 1e-100)
            if converged > 0 and converged < em_thresh:
                break
            if converged < 0:
                var_maxiter *= 2
            old_bound = bound



def main():
    infile = sys.argv[1]
    K = int(sys.argv[2])
    alpha = float(sys.argv[3])
    eta = float(sys.argv[4])

   # docs = corpus()
    corp = corpus.read_data(infile)
    D = corp.num_docs

    vocab = open(sys.argv[5]).readlines()
    model = LDAvb(vocab, K, D, 1./K, 1./K)

    wordids = [d.words for d in corp.docs[:]]
    wordcts = [d.counts for d in corp.docs[:]]
    model.run_em(wordids, wordcts)





if __name__ == '__main__':
    main()

